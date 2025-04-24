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
        print(f"Received game state: {game_state}")
        position_bits, mask_bits, moves = Position.convert_to_bitboard(game_state.board, game_state.current_player)
        position = Position(current_position=position_bits, mask=mask_bits, moves=moves)
        
        # Initialize play sequence
        if hasattr(game_state, 'played_sequence') and game_state.played_sequence:
            position._played_sequence = game_state.played_sequence
        else:
            position._played_sequence = []
        
        print(f"Bitboard: current_position={bin(position.current_position)}, mask={bin(position.mask)}")
        print(f"Converted position: {position}")
        ai_player = game_state.current_player
        print(f"Current player: {ai_player}")

        # Debug valid moves from both sources
        api_valid_moves = game_state.valid_moves
        pos_valid_moves = [col for col in range(Position.WIDTH) if position.can_play(col)]
        print(f"API valid moves: {api_valid_moves}")
        print(f"Position valid moves: {pos_valid_moves}")
        print(f"Played sequence: {position._played_sequence}")
        
        # Trust the game state's valid moves over the Position class calculation
        valid_moves = api_valid_moves if api_valid_moves else pos_valid_moves
        
        print(f"Using valid moves: {valid_moves}")
        if not valid_moves:
            print("No valid moves available")
            raise ValueError("No valid moves available")
        
        # Helper function to get non-losing moves
        def get_non_losing_moves(pos, valid_cols):
            try:
                non_losing_mask = pos.possible_non_losing_moves()
                non_losing_moves = []
                for col in valid_cols:
                    # Check if this column's bottom position is in the non-losing mask
                    col_mask = 1 << (col * (Position.HEIGHT + 1))
                    if (non_losing_mask & col_mask):
                        non_losing_moves.append(col)
                return non_losing_moves
            except Exception as e:
                print(f"Error getting non-losing moves: {str(e)}")
                return []
        
        # Helper function to select best move from a list with center preference
        def select_best_move(moves_list):
            if not moves_list:
                return None
            # Prefer center columns
            return sorted(moves_list, key=lambda c: abs(c - 3))[0]

        start_time = time.time()
        print(f"Checking book move at: {start_time}")
        book_move = api_solver.check_book_move(position, ai_player)
        print(f"Book move result: {book_move}")

        if book_move is not None and book_move in valid_moves and position.is_winning_move(book_move):
            end_time = time.time()
            print(f"Using winning book move: {book_move}")
            return AIResponse(
                move=book_move,
                is_winning_move=True,
                elapsed_time=end_time - start_time,
                played_sequence=position._played_sequence + [book_move]
            )
        
        print("Checking for winning moves before analysis")
        for col in valid_moves:
            if position.is_winning_move(col):
                end_time = time.time()
                print(f"Found immediate winning move: {col}")
                return AIResponse(
                    move=col,
                    is_winning_move=True,
                    elapsed_time=end_time - start_time,
                    played_sequence=position._played_sequence + [col]
                )

        # Always check for non-losing moves
        print("Checking for non-losing moves")
        non_losing_moves = get_non_losing_moves(position, valid_moves)
        print(f"Non-losing moves available: {non_losing_moves}")
        
        # If we have a book move and it's in the non-losing moves, use it
        if book_move is not None and book_move in valid_moves and book_move in non_losing_moves:
            end_time = time.time()
            print(f"Using non-losing book move: {book_move}")
            return AIResponse(
                move=book_move,
                is_winning_move=False,
                elapsed_time=end_time - start_time,
                played_sequence=position._played_sequence + [book_move]
            )

        # Try to use solver for better move selection if we have non-losing moves
        try:
            print(f"Solver state after reset: {api_solver}")
            api_solver.set_timeout(6.0)
            print(f"Analyzing position: {position}")

            scores = await asyncio.wait_for(
                asyncio.to_thread(lambda: api_solver.analyze(position, weak=False)), 
                timeout=6.0
            )
            print(f"Analysis completed, scores: {scores}")
            
            # If we have non-losing moves, only consider those
            candidate_moves = non_losing_moves if non_losing_moves else valid_moves
            
            best_col = -1
            best_score = -float('inf')
            print(f"Selecting best move from scores among {candidate_moves}")
            for col in candidate_moves:
                if scores[col] > best_score:
                    best_score = scores[col]
                    best_col = col
                    print(f"New best move: {best_col} with score: {best_score}")

            if best_col != -1:
                end_time = time.time()
                elapsed = end_time - start_time
                is_winning = position.is_winning_move(best_col)
                print(f"Selected move: {best_col}, is_winning: {is_winning}, elapsed: {elapsed}")
                return AIResponse(
                    move=best_col,
                    is_winning_move=is_winning,
                    elapsed_time=elapsed,
                    played_sequence=position._played_sequence + [best_col]
                )
            else:
                # Fall back to selecting based on preference within non-losing or valid moves
                best_col = select_best_move(candidate_moves)
                end_time = time.time()
                is_winning = position.is_winning_move(best_col)
                print(f"Using preferred move from candidates: {best_col}, is_winning: {is_winning}")
                return AIResponse(
                    move=best_col,
                    is_winning_move=is_winning,
                    elapsed_time=end_time - start_time,
                    played_sequence=position._played_sequence + [best_col]
                )
                
        except Exception as e:
            print(f"Solver error: {str(e)}")
            # Fall back to non-losing moves or center preference
            if non_losing_moves:
                best_col = select_best_move(non_losing_moves)
                print(f"Solver failed but using non-losing move: {best_col}")
            else:
                center_col = 3
                if center_col in valid_moves:
                    best_col = center_col
                else:
                    best_col = select_best_move(valid_moves)
                print(f"Solver failed, no non-losing moves, using preferred move: {best_col}")
            
            end_time = time.time()
            return AIResponse(
                move=best_col,
                is_winning_move=position.is_winning_move(best_col),
                elapsed_time=end_time - start_time,
                played_sequence=position._played_sequence + [best_col]
            )

    except Exception as e:
        print(f"Critical error occurred: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        
        try:
            # Even in critical failures, try to identify non-losing moves
            if 'position' in locals() and 'valid_moves' in locals():
                non_losing_moves = get_non_losing_moves(position, valid_moves)
                if non_losing_moves:
                    col = select_best_move(non_losing_moves)
                    print(f"Using non-losing move in critical error: {col}")
                    return AIResponse(
                        move=col,
                        played_sequence=position._played_sequence + [col]
                    )
            
            # Ultimate fallback
            if 'valid_moves' in locals() and valid_moves:
                col = select_best_move(valid_moves)
                print(f"Falling back to preferred valid move: {col}")
                return AIResponse(
                    move=col,
                    played_sequence=position._played_sequence + [col] if 'position' in locals() else []
                )
            elif 'game_state' in locals() and hasattr(game_state, 'valid_moves') and game_state.valid_moves:
                col = select_best_move(game_state.valid_moves)
                print(f"Falling back to preferred game state move: {col}")
                return AIResponse(
                    move=col,
                    played_sequence=[] # No sequence in error case
                )
        except Exception as inner_e:
            print(f"Even fallback error handling failed: {str(inner_e)}")
            # Try one last attempt with first valid move
            if 'valid_moves' in locals() and valid_moves:
                return AIResponse(move=valid_moves[0])
            elif 'game_state' in locals() and hasattr(game_state, 'valid_moves') and game_state.valid_moves:
                return AIResponse(move=game_state.valid_moves[0])
        
        # If all else fails
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
