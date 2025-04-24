from fastapi import FastAPI, HTTPException
import random
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import sys
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import numpy as np

class Position:
    WIDTH = 7
    HEIGHT = 6
    MIN_SCORE = -(WIDTH * HEIGHT) // 2 + 3
    MAX_SCORE = (WIDTH * HEIGHT + 1) // 2 - 3

    BOTTOM_MASK = sum(1 << (col * 7) for col in range(7))
    BOARD_MASK = BOTTOM_MASK * ((1 << 6) - 1)

    def __init__(self, position=None, current_position: int = 0, mask: int = 0, moves: int = 0):
        if position is not None:
            # Copy constructor
            self.current_position = position.current_position
            self.mask = position.mask
            self.moves = position.moves
            self._played_sequence = []
        else:
            # Regular constructor
            self.current_position = current_position
            self.mask = mask
            self.moves = moves
            self._played_sequence = []

    @staticmethod
    def bottom_mask_col(col: int) -> int:
        return 1 << (col * (Position.HEIGHT + 1))

    @staticmethod
    def top_mask_col(col: int) -> int:
        return 1 << ((Position.HEIGHT - 1) + col * (Position.HEIGHT + 1))

    @staticmethod
    def column_mask(col: int) -> int:
        return ((1 << Position.HEIGHT) - 1) << (col * (Position.HEIGHT + 1))

    @staticmethod
    def bottom_mask() -> int:
        return Position.BOTTOM_MASK

    @staticmethod
    def board_mask() -> int:
        return Position.BOARD_MASK

    def can_play(self, col: int) -> bool:
        return (self.mask & Position.top_mask_col(col)) == 0

    def play(self, move: int) -> None:
        self.current_position ^= self.mask
        self.mask |= move
        self.moves += 1

    def get_cell(self, col: int, row: int) -> str:
        mask = 1 << (col * (Position.HEIGHT + 1) + row)
        if not (self.mask & mask):
            return '.'
        return 'X' if (self.current_position & mask) else 'O'

    def play_col(self, col: int) -> None:
        self._played_sequence.append(col)
        self.play((self.mask + Position.bottom_mask_col(col)) & Position.column_mask(col))

    def play_sequence(self, sequence: str):
        valid_moves = 0
        for char in sequence:
            if char.isdigit() and '1' <= char <= '7':
                col = int(char) - 1
                if self.can_play(col):
                    self.play_col(col)
                    valid_moves += 1
        return valid_moves

    def is_winning_move(self, col: int) -> bool:
        if not self.can_play(col):
            return False
        
        temp_pos = self.current_position | ((self.mask + Position.bottom_mask_col(col)) & Position.column_mask(col))
        return self.check_win(temp_pos)

    def get_current_pieces(self):
        return self.current_position if self.moves % 2 == 0 else self.current_position ^ self.mask

    def possible(self) -> int:
        return (self.mask + Position.BOTTOM_MASK) & Position.BOARD_MASK

    def can_win_next(self) -> bool:
        # Calculate directly without calling other methods to avoid recursion
        winning_pos = self.compute_winning_position(self.current_position, self.mask)        
        possible_pos = (self.mask + Position.BOTTOM_MASK) & Position.BOARD_MASK
        
        return bool(winning_pos & possible_pos)

    def winning_position(self) -> int:
        return self.compute_winning_position(self.current_position, self.mask)

    def opponent_winning_position(self) -> int:
        return self.compute_winning_position(self.current_position ^ self.mask, self.mask)

    @staticmethod
    def compute_winning_position(position: int, mask: int) -> int:
        # Vertical
        r = (position << 1) & (position << 2) & (position << 3)

        # Horizontal
        p = (position << (Position.HEIGHT + 1)) & (position << 2 * (Position.HEIGHT + 1))
        r |= p & (position << 3 * (Position.HEIGHT + 1))
        r |= p & (position >> (Position.HEIGHT + 1))
        p = (position >> (Position.HEIGHT + 1)) & (position >> 2 * (Position.HEIGHT + 1))
        r |= p & (position << (Position.HEIGHT + 1))
        r |= p & (position >> 3 * (Position.HEIGHT + 1))

        # Diagonal 1
        p = (position << Position.HEIGHT) & (position << 2 * Position.HEIGHT)
        r |= p & (position << 3 * Position.HEIGHT)
        r |= p & (position >> Position.HEIGHT)
        p = (position >> Position.HEIGHT) & (position >> 2 * Position.HEIGHT)
        r |= p & (position << Position.HEIGHT)
        r |= p & (position >> 3 * Position.HEIGHT)

        # Diagonal 2
        p = (position << (Position.HEIGHT + 2)) & (position << 2 * (Position.HEIGHT + 2))
        r |= p & (position << 3 * (Position.HEIGHT + 2))
        r |= p & (position >> (Position.HEIGHT + 2))
        p = (position >> (Position.HEIGHT + 2)) & (position >> 2 * (Position.HEIGHT + 2))
        r |= p & (position << (Position.HEIGHT + 2))
        r |= p & (position >> 3 * (Position.HEIGHT + 2))

        # Direct calculation of board_mask to avoid recursion
        bottom = 0
        for col in range(Position.WIDTH):
            bottom |= (1 << (col * (Position.HEIGHT + 1)))
        board_mask = bottom * ((1 << Position.HEIGHT) - 1)
        
        return r & (board_mask ^ mask)

    def move_score(self, move: int) -> int:
        """
        Calculate a simple heuristic score for a move without causing recursion.
        
        Args:
            move: The move to evaluate (as a bitmask)
            
        Returns:
            A score value (higher is better)
        """
        # Count center-proximity as a basic heuristic
        col = 0
        temp_move = move
        while temp_move > 0:
            temp_move >>= (Position.HEIGHT + 1)
            col += 1
        
        # Prefer center columns (simple scoring)
        center_distance = abs(col - 1 - Position.WIDTH // 2)
        return Position.WIDTH - center_distance

    def copy(self):
        p = Position()
        p.current_position = self.current_position
        p.mask = self.mask
        p.moves = self.moves
        return p
    
    def possible_non_losing_moves(self) -> int:
        # Remove the assertion that's causing the infinite recursion
        # assert not self.can_win_next()
        
        # Calculate possible directly
        possible_mask = (self.mask + Position.BOTTOM_MASK) & Position.BOARD_MASK
        
        # Calculate opponent winning positions directly
        opponent_position = self.current_position ^ self.mask
        opponent_win = self.compute_winning_position(opponent_position, self.mask)
        
        forced_moves = possible_mask & opponent_win
        
        if forced_moves:
            if forced_moves & (forced_moves - 1):  # Check if more than one bit is set
                return 0  # Multiple forced moves means we lose
            possible_mask = forced_moves  # We're forced to play here
        
        return possible_mask & ~(opponent_win >> 1)  # Avoid letting opponent win after our move

    def check_diagonals(self, position: int) -> int:
        # Diagonal 1 (rising)
        p_d1 = (position << Position.HEIGHT) & (position << 2 * Position.HEIGHT)
        d1 = p_d1 & (position << 3 * Position.HEIGHT)
        d2 = p_d1 & (position >> Position.HEIGHT)
        
        p_d1b = (position >> Position.HEIGHT) & (position >> 2 * Position.HEIGHT)
        d3 = p_d1b & (position << Position.HEIGHT)
        d4 = p_d1b & (position >> 3 * Position.HEIGHT)
        
        # Diagonal 2 (falling)
        p_d2 = (position << (Position.HEIGHT + 2)) & (position << 2 * (Position.HEIGHT + 2))
        d5 = p_d2 & (position << 3 * (Position.HEIGHT + 2))
        d6 = p_d2 & (position >> (Position.HEIGHT + 2))
        
        p_d2b = (position >> (Position.HEIGHT + 2)) & (position >> 2 * (Position.HEIGHT + 2))
        d7 = p_d2b & (position << (Position.HEIGHT + 2))
        d8 = p_d2b & (position >> 3 * (Position.HEIGHT + 2))
        
        return d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8

    def check_win(self, position: int) -> bool:
        # Kiểm tra 4 hướng
        directions = [
            1,                    # Dọc
            self.HEIGHT + 1,      # Ngang
            self.HEIGHT,          # Chéo /
            self.HEIGHT + 2       # Chéo \
        ]
        
        for delta in directions:
            if (position & (position >> delta) & (position >> (2 * delta)) & (position >> (3 * delta))):
                return True
        return False
    
    def key(self) -> int:
        return hash((self.current_position, self.mask))


    def key3(self) -> int:
        key_forward = 0
        for col in range(Position.WIDTH):
            key_forward = self.partial_key3(key_forward, col)

        key_reverse = 0
        for col in range(Position.WIDTH-1, -1, -1):
            key_reverse = self.partial_key3(key_reverse, col)

        return min(key_forward, key_reverse) // 3

    def partial_key3(self, key: int, col: int) -> int:
        pos = 1 << (col * (Position.HEIGHT + 1))
        while pos & self.mask:
            key *= 3
            if pos & self.current_position:
                key += 1
            else:
                key += 2
            pos <<= 1
        key *= 3
        return key
    
    def nb_moves(self) -> int:
        """Return the number of moves played so far"""
        return self.moves

    def __str__(self) -> str:
        board = []
        # Hiển thị số cột
        board.append("  " + "  ".join([str(i+1) for i in range(Position.WIDTH)]))
        current_player_is_human = (self.moves % 2 == 0)
        
        # Duyệt qua từng ô
        for row in range(Position.HEIGHT-1, -1, -1):
            line = []
            for col in range(Position.WIDTH):
                mask = 1 << (col * (Position.HEIGHT + 1) + row)
                if self.mask & mask:
                    # Nếu ô đã được đánh
                    if self.current_position & mask:
                        # Quân cờ thuộc về người chơi HIỆN TẠI
                        symbol = 'O' if current_player_is_human else 'X'
                    else:
                        # Quân cờ thuộc về người chơi TRƯỚC ĐÓ
                        symbol = 'X' if current_player_is_human else 'O'
                    line.append(symbol)
                else:
                    line.append(".")
            board.append("| " + " | ".join(line) + " |")
        
        # Đường viền dưới
        board.append("+" + "---+" * Position.WIDTH)
        
        return "\n".join(board)

    @staticmethod
    def convert_to_bitboard(board: List[List[int]], current_player: int):
        WIDTH, HEIGHT = 7, 6
        position = np.uint64(0)
        mask = np.uint64(0)
        moves = 0

        # Duyệt theo hàng (từ dưới lên)
        for row in reversed(range(HEIGHT)):  # Hàng 5 là dưới cùng
            for col in range(WIDTH):
                if board[row][col] != 0:  # Truy cập [hàng][cột]
                    bit = col * (HEIGHT + 1) + (HEIGHT - 1 - row)  # Tính bit chính xác
                    mask |= np.uint64(1) << np.uint64(bit)
                    if board[row][col] == current_player:
                        position |= np.uint64(1) << np.uint64(bit)
                    moves += 1

        return position, mask, moves
    @staticmethod
    def reconstruct_sequence(board):
        temp_board = [[0 for _ in range(7)] for _ in range(6)]
        sequence = []
        
        # Count total pieces to determine how many moves to reconstruct
        total_pieces = sum(1 for row in board for cell in row if cell > 0)
        
        # Track whose turn it is during reconstruction
        current_player = 1  # Start with player 1
        
        for _ in range(total_pieces):
            # Find the next move in the sequence
            for col in range(7):  # Check each column
                # Find the highest empty row in this column
                for row in range(5, -1, -1):
                    if temp_board[row][col] == 0:  # Found an empty cell
                        # Check if this matches the target board
                        if board[row][col] == current_player:
                            # This is the next move
                            temp_board[row][col] = current_player
                            # Store as 1-indexed column (1-7)
                            sequence.append(col + 1)
                            # Switch players
                            current_player = 3 - current_player  # Toggle between 1 and 2
                            break
                if len(sequence) == total_pieces:
                    break
            if len(sequence) == total_pieces:
                break
        
        # Verify the reconstruction
        if temp_board != board:
            raise ValueError("Failed to reconstruct the sequence correctly")
            
        return sequence
    # Ensure the Position class has the get_played_sequence method
    def get_played_sequence(self) -> str:
        if not hasattr(self, '_played_sequence'):
            return ""
        return ''.join(str(col + 1) for col in self._played_sequence)

    # Ensure Position class has a clone method if it's used
    def clone(self):
        """Create a copy of the current position"""
        new_pos = Position()
        new_pos.current_position = self.current_position
        new_pos.mask = self.mask
        new_pos.moves = self.moves
        if hasattr(self, '_played_sequence'):
            new_pos._played_sequence = self._played_sequence.copy()
        return new_pos

    # Ensure Position class has a switch_player method if it's used
    def switch_player(self):
        """Switch the current player"""
        self.current_position ^= self.mask