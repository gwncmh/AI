import sys
import struct
import numpy as np
import pickle
import traceback
import random
import asyncio
from collections import defaultdict
import time
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from MoveSorter import MoveSorter
from Position import Position
from Solver import Solver
from TranspositionTable import TranspositionTable

def play_vs_ai(solver: Solver, human_turn: bool = False):
    position = Position()
    current_turn = human_turn
    ai_player = 2 if human_turn else 1 
    human_player = 1 if human_turn else 2

    player_symbols = {1: 'X', 2: 'O'}
    print(f"Connect Four - Human ({player_symbols[human_player]}) vs AI ({player_symbols[ai_player]})")
    print("Nhập số cột (1-7) để chơi\n")

    while True:
        print(position)
        if position.nb_moves() == Position.WIDTH * Position.HEIGHT:
            print("Hòa!")
            sequence = position.get_played_sequence()
            solver.add_to_book(sequence, 0)
            print("Đã lưu trận hòa vào battles.txt để AI học hỏi!")
            break

        if current_turn:
            while True:
                try:
                    col = int(input("Lượt bạn (1-7): ")) - 1
                    if 0 <= col < Position.WIDTH and position.can_play(col):
                        if position.is_winning_move(col):
                            position.play_col(col)
                            print(position)
                            print("Bạn thắng! Xuất sắc!")
                            sequence = position.get_played_sequence()
                            solver.add_to_book(sequence, human_player)
                            print("Đã lưu trận đấu vào battles.txt để AI học hỏi!")
                            return
                        position.play_col(col)
                        break
                    print("Cột không hợp lệ hoặc đã đầy!")
                except ValueError:
                    print("Vui lòng nhập số từ 1-7")
        else:
            print("\nAI đang suy nghĩ...")
            start_time = time.time()
            book_move = solver.check_book_move(position, ai_player)

            if book_move is not None and book_move >= 0 and position.can_play(book_move):
                best_col = book_move
                print(f"AI sử dụng nước đi từ opening book: cột {best_col + 1}")
            else:
                solver.reset()
                solver.set_timeout(8.0)
                try:
                    scores = solver.analyze(position, weak=False)
                except TimeoutError:
                    print("Solver timed out, using heuristic evaluation")
                    scores = [solver.evaluate_position(position.copy().play_col(col)) if position.can_play(col) else solver.INVALID_MOVE for col in range(Position.WIDTH)]
                best_col = -1
                best_score = -float('inf')

                for col in range(Position.WIDTH):
                    if position.can_play(col) and position.is_winning_move(col):
                        best_col = col
                        break

                if best_col == -1:
                    for col in range(Position.WIDTH):
                        if position.can_play(col) and scores[col] != solver.INVALID_MOVE and scores[col] > best_score:
                            best_score = scores[col]
                            best_col = col

                if best_col == -1:
                    print("No valid solver move, falling back to column order")
                    for col in solver.column_order:
                        if position.can_play(col):
                            best_col = col
                            break
                    if best_col == -1:
                        valid_moves = [col for col in range(Position.WIDTH) if position.can_play(col)]
                        best_col = random.choice(valid_moves) if valid_moves else -1

            if best_col != -1:
                end_time = time.time()
                elapsed = end_time - start_time
                print(f"AI suy nghĩ trong: {elapsed:.2f} giây")
                print(f"AI chọn cột {best_col + 1}")

                if position.is_winning_move(best_col):
                    position.play_col(best_col)
                    print(position)
                    print("AI thắng! Hãy thử lại!")
                    sequence = position.get_played_sequence()
                    solver.add_to_book(sequence, ai_player)  # Save with AI's player number
                    print("Đã lưu chiến thắng vào battles.txt để AI học hỏi!")
                    return

                position.play_col(best_col)
            else:
                print("AI không tìm được nước đi hợp lệ!")
                return

        current_turn = not current_turn
    
class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]
    is_new_game: bool = False

class AIResponse(BaseModel):
    move: int
    is_winning_move: bool = False
    elapsed_time: float = 0.0

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_solver = Solver()

@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        position_bits, mask_bits, moves = Position.convert_to_bitboard(game_state.board, game_state.current_player)
        position = Position(current_position=position_bits, mask=mask_bits, moves=moves)
        
        if not game_state.is_new_game:
            try:
                move_sequence = Position.reconstruct_sequence(game_state.board)
                position._played_sequence = ''.join(map(str, move_sequence))
            except Exception as e:
                position._played_sequence = ''
        else:
            position._played_sequence = ''
        
        ai_player = game_state.current_player
        valid_moves = game_state.valid_moves
        
        if not valid_moves:
            raise ValueError("No valid moves available")

        start_time = time.time()
        
        # 1. Check for book move first
        book_move = api_solver.check_book_move(position, ai_player)
        if book_move is not None and book_move in valid_moves:
            end_time = time.time()
            is_winning = position.is_winning_move(book_move)
            return AIResponse(
                move=book_move,
                is_winning_move=is_winning,
                elapsed_time=end_time - start_time
            )
        
        # 2. Check for immediate winning move
        for col in valid_moves:
            if position.is_winning_move(col):
                end_time = time.time()
                return AIResponse(
                    move=col,
                    is_winning_move=True,
                    elapsed_time=end_time - start_time
                )

        # 3. Try solver with timeout
        api_solver.reset()
        api_solver.set_timeout(6.0)
        
        try:
            scores = await asyncio.wait_for(
                asyncio.to_thread(lambda: api_solver.analyze(position, weak=False)), 
                timeout=6.0
            )
        except asyncio.TimeoutError:
            # If solver times out, initialize scores for non-losing move detection
            scores = [-1] * Position.WIDTH
        
        # 4. Find non-losing moves
        non_losing_moves = []
        for col in valid_moves:
            # Test if move leads to immediate loss
            pos_copy = position.copy()
            pos_copy.play_col(col)
            is_losing = False
            
            for opp_col in range(Position.WIDTH):
                if pos_copy.can_play(opp_col) and pos_copy.is_winning_move(opp_col):
                    is_losing = True
                    break
                    
            if not is_losing:
                score = scores[col] if scores[col] != api_solver.INVALID_MOVE else 0
                non_losing_moves.append((col, score))
        
        # 5. Select best non-losing move
        if non_losing_moves:
            # Sort by score (highest first)
            non_losing_moves.sort(key=lambda x: x[1], reverse=True)
            best_col = non_losing_moves[0][0]
        else:
            # All moves lose, pick center column or follow column priority
            column_priority = [3, 2, 4, 1, 5, 0, 6]  # Prefer center columns
            for col in column_priority:
                if col in valid_moves:
                    best_col = col
                    break
            else:
                # If somehow we get here, pick first available move
                best_col = valid_moves[0]
        
        end_time = time.time()
        is_winning = position.is_winning_move(best_col)
        
        return AIResponse(
            move=best_col,
            is_winning_move=is_winning,
            elapsed_time=end_time - start_time
        )

    except Exception as e:
        traceback.print_exc()
        # Even in exception case, return a non-losing move if possible
        try:
            valid_moves = game_state.valid_moves
            if valid_moves:
                # Try to find a non-losing move even in error recovery
                for col in [3, 2, 4, 1, 5, 0, 6]:  # Center priority
                    if col in valid_moves:
                        return AIResponse(move=col)
                return AIResponse(move=valid_moves[0])
        except:
            pass
        raise HTTPException(status_code=400, detail=str(e))
def main():
    solver = Solver()
    weak = False
    analyze = False
    interactive = False
    args = sys.argv[1:]
    input_from_stdin = False
    api_mode = False
    opening_book_file = None
    
    for i, arg in enumerate(args):
        if arg == '-i':
            interactive = True
        elif arg == '-w':
            weak = True
        elif arg == '-a':
            analyze = True
        elif arg == '-api':
            api_mode = True
        elif arg == '-b':
            if i+1 < len(args) and not args[i+1].startswith('-'):
                opening_book_file = args[i+1]
                print(f"Using opening book: {opening_book_file}")
        elif not arg.startswith('-') and not arg.isdigit():
            input_from_stdin = True

    if opening_book_file:
        if solver.load_book(opening_book_file):
            print("Opening book loaded successfully")
        else:
            print("Failed to load opening book")

    if interactive:
        play_vs_ai(solver)
        return
    
    if api_mode:
        print("Khởi động API Connect Four AI trên cổng 10000...")
        uvicorn.run(app, host="0.0.0.0", port=10000)
        return
    
    if not sys.stdin.isatty() and input_from_stdin:
        for line in sys.stdin:
            line = line.strip()
            if line:
                position = Position()
                try:
                    position.play_sequence(line)
                    if analyze:
                        scores = solver.analyze(position, weak)
                        print(" ".join(map(str, scores)))
                    else:
                        score = solver.solve(position, weak)
                        print(score)
                except ValueError as e:
                    print(f"Lỗi khi xử lý nước đi: {e}")
    else:
        print("Sử dụng các tùy chọn sau:")
        print("  -i: Chơi với AI")
        print("  -api: Khởi động API Connect Four AI")
        print("  -b [file]: Sử dụng opening book từ file")
        print("  -w: Giảm độ mạnh của solver")
        print("  -a: Phân tích vị trí")

if __name__ == "__main__":
    main()
