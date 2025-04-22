import sys
import struct
import numpy as np
import random
import pickle
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

def play_vs_ai(solver: Solver):
    position = Position()
    human_turn = True  # True for human's turn, False for AI's turn
    import time  # Add time import for measuring AI thinking time
    
    print("Connect Four - Human (O) vs AI (X)")
    print("Nhập số cột (1-7) để chơi\n")
    
    while True:
        print(position)
        
        # Check for draw
        if position.nb_moves() == Position.WIDTH * Position.HEIGHT:
            print("Hòa!")
            # Save drawn game to opening book with neutral score
            sequence = position.get_played_sequence()
            solver.add_to_book(sequence, 0)  # 0 indicates a draw
            print("Đã lưu trận hòa vào battles.txt để AI học hỏi!")
            break
            
        if human_turn:
            # Human's turn
            while True:
                try:
                    col = int(input("Lượt bạn (1-7): ")) - 1
                    if 0 <= col < Position.WIDTH and position.can_play(col):
                        if position.is_winning_move(col):
                            position.play_col(col)
                            print(position)
                            print("Bạn thắng! Xuất sắc!")
                            # Save losing sequence to opening book with human win score
                            sequence = position.get_played_sequence()
                            solver.add_to_book(sequence, 1)  # 1 indicates human win
                            print("Đã lưu trận đấu vào battles.txt để AI học hỏi!")
                            return
                        position.play_col(col)
                        break
                    print("Cột không hợp lệ hoặc đã đầy!")
                except ValueError:
                    print("Vui lòng nhập số từ 1-7")
        else:
            # AI's turn
            print("\nAI đang suy nghĩ...")
            start_time = time.time()
            
            # AI player is player 2 (since human goes first)
            book_move = solver.check_book_move(position, 2)
            
            if book_move is not None and book_move >= 0 and position.can_play(book_move):
                best_col = book_move
                print(f"AI sử dụng nước đi từ opening book: cột {best_col + 1}")
            else:
                # If no book move available, use the solver analysis
                scores = solver.analyze(position)
                    
                # Find the best move
                best_col = -1
                best_score = -float('inf')
                    
                for col in range(Position.WIDTH):
                    if position.can_play(col) and scores[col] > best_score and scores[col] != solver.INVALID_MOVE:
                        best_score = scores[col]
                        best_col = col

                # If AI has a winning move, add a message
                if best_col != -1 and position.is_winning_move(best_col):
                    print("AI đã tìm thấy nước đi chiến thắng!")
            
            if best_col != -1:
                end_time = time.time()
                elapsed = end_time - start_time
                print(f"AI suy nghĩ trong: {elapsed:.2f} giây")
                print(f"AI chọn cột {best_col + 1}")
                
                if position.is_winning_move(best_col):
                    position.play_col(best_col)
                    print(position)
                    print("AI thắng! Hãy thử lại!")
                    
                    # Save winning sequence to opening book with AI win score
                    sequence = position.get_played_sequence()
                    solver.add_to_book(sequence, 2)  # 2 indicates AI win
                    print("Đã lưu chiến thắng vào battles.txt để AI học hỏi!")
                    
                    return
                    
                position.play_col(best_col)
            else:
                print("AI không tìm được nước đi hợp lệ!")
                return
        
        human_turn = not human_turn
    
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]
    is_new_game: bool = False  # Optional parameter to identify new games

class AIResponse(BaseModel):
    move: int
    is_winning_move: bool = False  # Optional info about move quality
    elapsed_time: float = 0.0  # Optional timing information

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize solver for API
api_solver = Solver()

# Time limit parameters
TIME_LIMIT_MS = 8000 

@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        # Validate input
        if not game_state.valid_moves:
            raise ValueError("No valid moves available")
            
        # Convert board to Position object
        position = Position.from_2d_array(game_state.board)
        
        # Check for winning moves first
        for col in game_state.valid_moves:
            if position.is_winning_move(col):
                return AIResponse(move=col, is_winning_move=True)
        
        # If new game, prefer the center column
        if game_state.is_new_game and 3 in game_state.valid_moves:
            return AIResponse(move=3)
            
        # Use solver to analyze the position
        scores = solver.analyze(position)
        
        # Find best move based on scores
        best_col = -1
        best_score = float('-inf')
        
        for col in game_state.valid_moves:
            if scores[col] > best_score:
                best_score = scores[col]
                best_col = col
                
        if best_col != -1:
            return AIResponse(move=best_col)
        
        # Fallback to a random move if analysis fails
        selected_move = random.choice(game_state.valid_moves) 
        return AIResponse(move=selected_move)
        
    except Exception as e:
        # Provide a safe fallback
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
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

    # Load opening book if specified
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
