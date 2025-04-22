class OpeningBook:
    def __init__(self):
        # Known winning sequences with their corresponding winner
        self.winning_sequences = {
            "4444422224536766626555573172311133356117": {"winner": 1},
            "443234463355": {"winner": 1},
            "44325634544655332334171777775666": {"winner": 2},
            "444441363322": {"winner": 1},
            "4435544223312234336664221776655557": {"winner": 2},
            "44355444645567667336": {"winner": 1},
            "4435543445333721251773755422": {"winner": 1},
            "4443443373233756655": {"winner": 1},
            "436553552332644": {"winner": 1},
            "4443454456323355766": {"winner": 1},
            "43655443554476363": {"winner": 1},
            "452334425355364222362": {"winner": 1},
            "436554453652466633221": {"winner": 1},
            "436553545546633": {"winner": 1},
            "4364455536524233275412111": {"winner": 1},
            "4444443256557265655633": {"winner": 1},
            "23133223532122561355676166757775671144": {"winner": 2},
            "362444434755555562271143333777666167211212": {"winner": 2},
            "654464476632254557745566222211111133": {"winner": 2},
            "23676266121311442555": {"winner": 2},
            "14463574242122133315": {"winner": 2},
            "65666555653144336111475441137774773322": {"winner": 2},
            "5355673261445433424222332554177766": {"winner": 2},
            "366324335525566322377225644177": {"winner": 2},
            "12223544524472214467666773333655": {"winner": 2}
        }
        try:
            self.load_from_file("battles.txt")
        except:
            pass
    
    def get_first_move(self):
        return 3
    
    def find_next_move(self, position, ai_player):
        # Get the current sequence of moves
        current_sequence = position.get_played_sequence()
        sequence_length = len(current_sequence)
        
        # Check if this is the first move and AI is player 1
        if sequence_length == 0 and ai_player == 1:
            return self.get_first_move()
        
        # Determine if AI plays odd or even positions based on whether it went first
        ai_plays_first = ai_player == 1
        next_move_is_odd = sequence_length % 2 == 0
        ai_should_play = (ai_plays_first and next_move_is_odd) or (not ai_plays_first and not next_move_is_odd)
        
        # If it's not AI's turn, don't use the opening book
        if not ai_should_play:
            return None
        
        # Look for a known sequence that matches the current sequence prefix
        for known_sequence, data in self.winning_sequences.items():
            if current_sequence == known_sequence[:sequence_length]:
                if sequence_length < len(known_sequence):
                    next_move = int(known_sequence[sequence_length]) - 1
                    if position.can_play(next_move):
                        return next_move
        return None
    
    def add_sequence(self, sequence, winner):
        self.winning_sequences[sequence] = {"winner": winner}
        self.save_to_file("battles.txt")
    
    def remove_sequence(self, sequence):
        """Remove a sequence from the opening book"""
        if sequence in self.winning_sequences:
            del self.winning_sequences[sequence]
            # Save to battles.txt automatically when removing a sequence
            self.save_to_file("battles.txt")
            return True
        return False
    
    def get_all_sequences(self):
        """Return all winning sequences in the opening book"""
        return self.winning_sequences
    
    def load_from_file(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                parts = line.split(' ')
                if len(parts) == 2:
                    sequence = parts[0].strip()
                    winner = int(parts[1].strip())
                    self.winning_sequences[sequence] = {"winner": winner}
            
            return True
        except Exception as e:
            print(f"Error loading opening book: {e}")
            return False
    
    def save_to_file(self, filename):
        try:
            with open(filename, 'w') as f:
                f.write("# Connect Four Opening Book\n")
                f.write("# Format: sequence winner\n")
                f.write("# sequence: string of column numbers (1-indexed)\n")
                f.write("# winner: 1 for Player 1, 2 for Player 2, 0 for draw\n\n")
                
                for sequence, data in self.winning_sequences.items():
                    f.write(f"{sequence} {data['winner']}\n")
            
            return True
        except Exception as e:
            print(f"Error saving opening book: {e}")
            return False