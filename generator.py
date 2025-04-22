import random

class Generator:
    def __init__(self, seed=None):
        self.random = random.Random(seed)

    def generate_move(self, valid_moves):
        return self.random.choice(valid_moves) if valid_moves else None