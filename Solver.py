from typing import List, Optional
from Position import Position
from MoveSorter import MoveSorter
from TranspositionTable import TranspositionTable
from OpeningBook import OpeningBook

class Solver:
    """
    Connect 4 solver using Negamax algorithm with alpha-beta pruning
    and various optimization techniques.
    """
    INVALID_MOVE = -1000
    # Base column values (center columns are worth more)
    
    def __init__(self):
        """Initialize the solver"""
        self.node_count = 0
        self.trans_table = {}  # Use an empty dict as transposition table
        self.opening_book = OpeningBook()  # Initialize opening book
        self.column_order = [3, 4, 2, 5, 1, 6, 0]
        self.column_values = [-1, 0, 0, 1, 0, 0, -1]
        self.timeout = None
        self.start_time = None
        # Precompute column factors for horizontal windows
        self.horizontal_col_factors = [
            sum(self.column_values[col + i] for i in range(4)) / 4
            for col in range(Position.WIDTH - 3)
        ]

    def load_book(self, book_file: str) -> None:
        """Load opening book from file"""
        return self.opening_book.load_from_file(book_file)

    def save_book(self, book_file: str) -> None:
        """Save opening book to file"""
        return self.opening_book.save_to_file(book_file)

    def check_book_move(self, position: Position, ai_player: int) -> int:
        return self.opening_book.find_next_move(position, ai_player)

    def add_to_book(self, sequence: str, winner: int) -> None:
        """Add a sequence to the opening book"""
        self.opening_book.add_sequence(sequence, winner)

    def negamax(self, position: Position, alpha: int, beta: int, depth: int = 0, max_depth: int = 16) -> int:
        assert alpha < beta
        assert not position.can_win_next()
        
        self.node_count += 1  # Increment counter of explored nodes
        
        # Check max depth if specified
        if max_depth is not None and depth >= max_depth:
            return self.evaluate_position(position)
        
        # Check if there are valid moves
        possible = position.possible_non_losing_moves()
        if possible == 0:  # No possible non-losing moves, opponent wins next move
            return -((Position.WIDTH * Position.HEIGHT - position.nb_moves()) // 2)
        
        # Check for draw game
        if position.nb_moves() >= Position.WIDTH * Position.HEIGHT - 2:
            return 0
        
        # Lower bound of score as opponent cannot win next move
        min_score = -((Position.WIDTH * Position.HEIGHT - 2 - position.nb_moves()) // 2)
        if alpha < min_score:
            alpha = min_score
            if alpha >= beta:
                return alpha  # Prune if window is empty
        
        # Upper bound of our score as we cannot win immediately
        max_score = (Position.WIDTH * Position.HEIGHT - 1 - position.nb_moves()) // 2
        if beta > max_score:
            beta = max_score
            if alpha >= beta:
                return beta  # Prune if window is empty
        
        # Check transposition table
        key = position.key()
        val = self.trans_table.get(key)
        if val is not None:
            if val > Position.MAX_SCORE - Position.MIN_SCORE + 1:  # We have a lower bound
                min_val = val + 2 * Position.MIN_SCORE - Position.MAX_SCORE - 2
                if alpha < min_val:
                    alpha = min_val
                    if alpha >= beta:
                        return alpha  # Prune if window is empty
            else:  # We have an upper bound
                max_val = val + Position.MIN_SCORE - 1
                if beta > max_val:
                    beta = max_val
                    if alpha >= beta:
                        return beta  # Prune if window is empty
        
        # Sort moves
        moves = []
        for i in range(Position.WIDTH):
            col = self.column_order[i]
            move = possible & Position.column_mask(col)
            if move:
                # Add base column preference value for early game
                column_score_bonus = 0
                if position.nb_moves() < 10:  # Early game bonus
                    column_score_bonus = self.column_values[col] * (10 - position.nb_moves())
                
                # Add (move, score, column_order_index) to moves list
                moves.append((move, position.move_score(move) + column_score_bonus, i))
        
        # Sort moves by score (DESCENDING) and then by preference (ASCENDING)
        moves.sort(key=lambda x: (x[1], -x[2]), reverse=True)  # Changed to reverse=True
        
        for move_data in moves:
            move = move_data[0]  # Extract just the move component
            new_pos = position.copy()
            new_pos.play(move)
            score = -self.negamax(new_pos, -beta, -alpha, depth + 1, max_depth)
            
            if score >= beta:
                # Save lower bound
                self.trans_table[key] = score + Position.MAX_SCORE - 2 * Position.MIN_SCORE + 2
                return score  # Prune
            
            if score > alpha:
                alpha = score  # Reduce window for next exploration
        
        # Save upper bound
        self.trans_table[key] = alpha - Position.MIN_SCORE + 1
        return alpha

    def evaluate_position(self, position: Position) -> int:
        # Immediate win check
        if position.can_win_next():
            return (Position.WIDTH * Position.HEIGHT - position.nb_moves()) // 2

        score = 0
        current_player_bits = position.current_position
        opponent_bits = position.mask & ~position.current_position

        # Horizontal patterns
        for row in range(Position.HEIGHT):
            for col in range(Position.WIDTH - 3):
                count_current = 0
                count_opponent = 0
                for i in range(4):
                    pos = (col + i) * (Position.HEIGHT + 1) + row
                    if current_player_bits & (1 << pos):
                        count_current += 1
                    elif opponent_bits & (1 << pos):
                        count_opponent += 1
                if count_opponent == 0 and count_current > 0:
                    score += count_current * count_current * self.horizontal_col_factors[col]
                elif count_current == 0 and count_opponent > 0:
                    score -= count_opponent * count_opponent * self.horizontal_col_factors[col]
                elif count_current >= 2 and count_opponent == 0:
                    score += count_current * 1.5  # Encourage potential threats

        # Vertical patterns
        for col in range(Position.WIDTH):
            count_current = 0
            count_opponent = 0
            for row in range(Position.HEIGHT):
                pos = col * (Position.HEIGHT + 1) + row
                if current_player_bits & (1 << pos):
                    count_current += 1
                elif opponent_bits & (1 << pos):
                    count_opponent += 1
            if count_opponent == 0 and count_current > 0:
                score += count_current * count_current * self.column_values[col]
            elif count_current == 0 and count_opponent > 0:
                score -= count_opponent * count_opponent * self.column_values[col]
            elif count_current >= 2 and count_opponent == 0:
                score += count_current * 1.5

        # Center column bonus
        center_col = Position.WIDTH // 2
        center_mask = sum(1 << (center_col * (Position.HEIGHT + 1) + row) for row in range(Position.HEIGHT))
        score += bin(current_player_bits & center_mask).count('1') * 2
        score -= bin(opponent_bits & center_mask).count('1') * 2

        return score

    def solve(self, position: Position, weak: bool = False) -> int:
        # Check for immediate win
        if position.can_win_next():
            return (Position.WIDTH * Position.HEIGHT + 1 - position.nb_moves()) // 2
        
        # Initialize min-max bounds
        min_score = -(Position.WIDTH * Position.HEIGHT - position.nb_moves()) // 2
        max_score = (Position.WIDTH * Position.HEIGHT + 1 - position.nb_moves()) // 2
        
        if weak:
            min_score = -1
            max_score = 1
        
        # Iteratively narrow the min-max exploration window
        while min_score < max_score:
            med = min_score + (max_score - min_score) // 2
            
            # Adjust median for better performance
            if med <= 0 and min_score // 2 < med:
                med = min_score // 2
            elif med >= 0 and max_score // 2 > med:
                med = max_score // 2
                
            # Use a null depth window to determine if score is greater or smaller than med
            r = self.negamax(position, med, med + 1)
            
            if r <= med:
                max_score = r
            else:
                min_score = r
                
        return min_score

    def analyze(self, position: Position, weak: bool = False) -> List[int]:
        scores = [self.INVALID_MOVE] * Position.WIDTH
        
        # Check each column
        for col in range(Position.WIDTH):
            if position.can_play(col):
                if position.is_winning_move(col):
                    scores[col] = (Position.WIDTH * Position.HEIGHT + 1 - position.nb_moves()) // 2
                else:
                    # Create new position with this move
                    new_pos = Position(position)
                    new_pos.play_col(col)
                    scores[col] = -self.solve(new_pos, weak)
        print(f"Column scores: {scores}")
        return scores

    def set_timeout(self, seconds: float):
        self.timeout = seconds
        self.start_time = None
        
    def reset(self) -> None:
        """Reset the solver state"""
        self.node_count = 0
        self.trans_table = {}

    def get_node_count(self) -> int:
        """Get number of nodes explored in the last search"""
        return self.node_count
