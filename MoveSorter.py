class MoveSorter:
    """
    Helper class to sort moves by score for more efficient alpha-beta pruning
    """
    def __init__(self):
        self.moves = []
        self.scores = []
        self.size = 0
    
    def add(self, move: int, score: int) -> None:
        """Add a move with its score to the sorter"""
        # Find the right position for insertion (maintaining descending order)
        pos = 0
        while pos < self.size and score < self.scores[pos]:
            pos += 1
        
        # Insert the move at the found position
        self.moves.insert(pos, move)
        self.scores.insert(pos, score)
        self.size += 1
    
    def get_next(self) -> int:
        """Get the next best move"""
        if self.size == 0:
            return 0
        
        self.size -= 1
        return self.moves[self.size]