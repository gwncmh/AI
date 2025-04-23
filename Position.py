class Position:
    WIDTH = 7
    HEIGHT = 6
    MIN_SCORE = -(WIDTH * HEIGHT) // 2 + 3
    MAX_SCORE = (WIDTH * HEIGHT + 1) // 2 - 3

    BOTTOM_MASK = sum(1 << (col * 7) for col in range(7))  # Bits 0, 7, 14, 21, 28, 35, 42
    BOARD_MASK = sum((1 << (col * 7 + row)) for col in range(7) for row in range(6))  # All playable bits

    def __init__(self, position=None, current_position: int = 0, mask: int = 0, moves: int = 0):
        if position is not None:
            self.current_position = position.current_position
            self.mask = position.mask
            self.moves = position.moves
            self._played_sequence = position._played_sequence.copy() if hasattr(position, '_played_sequence') else []
        else:
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
        winning_pos = self.compute_winning_position(self.current_position, self.mask)
        possible_pos = (self.mask + Position.BOTTOM_MASK) & Position.BOARD_MASK
        print(f"can_win_next: winning_pos={bin(winning_pos)}, possible_pos={bin(possible_pos)}")
        return bool(winning_pos & possible_pos)

    def winning_position(self) -> int:
        return self.compute_winning_position(self.current_position, self.mask)

    def opponent_winning_position(self) -> int:
        return self.compute_winning_position(self.current_position ^ self.mask, self.mask)

    @staticmethod
    def compute_winning_position(position: int, mask: int) -> int:
        r = (position << 1) & (position << 2) & (position << 3)
        p = (position << (Position.HEIGHT + 1)) & (position << 2 * (Position.HEIGHT + 1))
        r |= p & (position << 3 * (Position.HEIGHT + 1))
        r |= p & (position >> (Position.HEIGHT + 1))
        p = (position >> (Position.HEIGHT + 1)) & (position >> 2 * (Position.HEIGHT + 1))
        r |= p & (position << (Position.HEIGHT + 1))
        r |= p & (position >> 3 * (Position.HEIGHT + 1))
        p = (position << Position.HEIGHT) & (position << 2 * Position.HEIGHT)
        r |= p & (position << 3 * Position.HEIGHT)
        r |= p & (position >> Position.HEIGHT)
        p = (position >> Position.HEIGHT) & (position >> 2 * Position.HEIGHT)
        r |= p & (position << Position.HEIGHT)
        r |= p & (position >> 3 * Position.HEIGHT)
        p = (position << (Position.HEIGHT + 2)) & (position << 2 * (Position.HEIGHT + 2))
        r |= p & (position << 3 * (Position.HEIGHT + 2))
        r |= p & (position >> (Position.HEIGHT + 2))
        p = (position >> (Position.HEIGHT + 2)) & (position >> 2 * (Position.HEIGHT + 2))
        r |= p & (position << (Position.HEIGHT + 2))
        r |= p & (position >> 3 * (Position.HEIGHT + 2))
        return r & (Position.BOARD_MASK ^ mask)

    def move_score(self, move: int) -> int:
        col = 0
        temp_move = move
        while temp_move > 0:
            temp_move >>= (Position.HEIGHT + 1)
            col += 1
        center_distance = abs(col - Position.WIDTH // 2)
        return Position.WIDTH - center_distance

    def copy(self):
        p = Position()
        p.current_position = self.current_position
        p.mask = self.mask
        p.moves = self.moves
        p._played_sequence = self._played_sequence.copy()
        return p

    def possible_non_losing_moves(self) -> int:
        possible_mask = (self.mask + Position.BOTTOM_MASK) & Position.BOARD_MASK
        opponent_position = self.current_position ^ self.mask
        opponent_win = self.compute_winning_position(opponent_position, self.mask)
        forced_moves = possible_mask & opponent_win
        if forced_moves:
            if forced_moves & (forced_moves - 1):
                return 0
            possible_mask = forced_moves
        return possible_mask & ~(opponent_win >> 1)

    def check_diagonals(self, position: int) -> int:
        p_d1 = (position << Position.HEIGHT) & (position << 2 * Position.HEIGHT)
        d1 = p_d1 & (position << 3 * Position.HEIGHT)
        d2 = p_d1 & (position >> Position.HEIGHT)
        p_d1b = (position >> Position.HEIGHT) & (position >> 2 * Position.HEIGHT)
        d3 = p_d1b & (position << Position.HEIGHT)
        d4 = p_d1b & (position >> 3 * Position.HEIGHT)
        p_d2 = (position << (Position.HEIGHT + 2)) & (position << 2 * (Position.HEIGHT + 2))
        d5 = p_d2 & (position << 3 * (Position.HEIGHT + 2))
        d6 = p_d2 & (position >> (Position.HEIGHT + 2))
        p_d2b = (position >> (Position.HEIGHT + 2)) & (position >> 2 * (Position.HEIGHT + 2))
        d7 = p_d2b & (position << (Position.HEIGHT + 2))
        d8 = p_d2b & (position >> 3 * (Position.HEIGHT + 2))
        return d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8

    def check_win(self, position: int) -> bool:
        directions = [
            1,
            self.HEIGHT + 1,
            self.HEIGHT,
            self.HEIGHT + 2
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
        return self.moves

    def __str__(self) -> str:
        board = []
        board.append("  " + "  ".join([str(i+1) for i in range(Position.WIDTH)]))
        # Iterate from bottom (row 0) to top (row 5) to match 2D array
        for row in range(Position.HEIGHT):
            line = []
            for col in range(Position.WIDTH):
                mask = 1 << (col * (Position.HEIGHT + 1) + row)
                if not (self.mask & mask):
                    line.append(".")
                else:
                    line.append('X' if (self.current_position & mask) else 'O')
            board.append("| " + " | ".join(line) + " |")
        board.append("+" + "---+" * Position.WIDTH)
        return "\n".join(board[::-1])  # Reverse to display bottom row at bottom

    @classmethod
    def from_2d_array(cls, board: List[List[int]], current_player: int = 1) -> 'Position':
        position = cls()
        position._played_sequence = []
        position.current_position = 0
        position.mask = 0
        for row in range(cls.HEIGHT):
            for col in range(cls.WIDTH):
                bit_pos = col * (cls.HEIGHT + 1) + row
                if board[row][col] != 0:
                    position.mask |= (1 << bit_pos)
                    if board[row][col] == 1:
                        position.current_position |= (1 << bit_pos)
        position.moves = sum(row.count(1) + row.count(2) for row in board)
        if current_player == 2:
            position.current_position ^= position.mask
        return position

    def get_played_sequence(self) -> str:
        return ''.join(str(col + 1) for col in self._played_sequence)

    def clone(self):
        return self.copy()

    def switch_player(self):
        self.current_position ^= self.mask