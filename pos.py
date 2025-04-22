import time
import cProfile
from Position import Position
from Solver import Solver
from MoveSorter import MoveSorter
from TranspositionTable import TranspositionTable

def parse_sequence(sequence_str):
    """
    Parse a string of moves (1-7) into a sequence of column indices (0-6)
    """
    return [int(char) - 1 for char in sequence_str if '1' <= char <= '7']

def setup_position_from_sequence(sequence):
    """
    Set up a position by playing a sequence of moves
    """
    position = Position()
    for col in sequence:
        position.play_col(col)
    return position

def test_position(sequence_str, expected_score):
    """
    Test a position and check if the solver gives the expected score
    """
    sequence = parse_sequence(sequence_str)
    position = setup_position_from_sequence(sequence)
    
    solver = Solver()
    
    start_time = time.time()
    score = solver.solve(position)
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Testing position: {sequence_str}")
    print(f"Board state:")
    print(position)
    print(f"Expected score: {expected_score}")
    print(f"Actual score: {score}")
    print(f"Result: {'PASS' if score == expected_score else 'FAIL'}")
    print(f"Nodes evaluated: {solver.get_node_count()}")
    print(f"Time elapsed: {elapsed:.4f} seconds")
    print("-" * 50)
    
    return score == expected_score

def profile_negamax():
    solver = Solver()  # or whatever your solver class is called
    sequence_str = "6517643167442672742"
    sequence = parse_sequence(sequence_str)
    position = setup_position_from_sequence(sequence)
    alpha = -1000
    beta = 1000
    solver.negamax(position, alpha, beta)

def main():
    test_cases = [
        ("44561362221", 2),
        ("1734475752", 0),
        ("351334437543377", 5)
    ]
    
    total = len(test_cases)
    passed = 0
    
    print("=== CONNECT FOUR SOLVER TEST ===\n")

    for sequence_str, expected_score in test_cases:
        seq = sequence_str
        exp = expected_score
        cProfile.runctx("test_position(seq, exp)", globals(), locals(), sort="cumulative")

if __name__ == "__main__":
    main()


#if __name__ == "__main__":
 #   test_positions = [
  #      274552224131661,
   #     5455174361263362,
    #   37313333717124171162542,
      #  6614446666373154,
     #   24617524315172127,
    #]

    #for pos in test_positions:
     #   compare(pos)
