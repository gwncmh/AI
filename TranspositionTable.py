import numpy as np

class TranspositionTable:
    def __init__(self, log_size=24, key_type=np.uint64, value_type=np.int8):
        """
        Create a transposition table with 2^log_size entries.
        
        Parameters:
        -----------
        log_size : int
            Base-2 logarithm of the desired table size
        key_type : numpy dtype
            Data type for keys (e.g., np.uint32, np.uint64)
        value_type : numpy dtype
            Data type for values (e.g., np.int8, np.int16)
        """
        # Use a prime number size like the C++ version
        self.size = self._next_prime(1 << log_size)
        # Use numpy arrays for memory efficiency
        self.keys = np.zeros(self.size, dtype=key_type)
        self.values = np.zeros(self.size, dtype=value_type)
        self.key_type = key_type
        self.value_type = value_type
    
    def _next_prime(self, n):
        """Find the next prime number >= n"""
        def is_prime(num):
            if num <= 1:
                return False
            if num <= 3:
                return True
            if num % 2 == 0 or num % 3 == 0:
                return False
            i = 5
            while i * i <= num:
                if num % i == 0 or num % (i + 2) == 0:
                    return False
                i += 6
            return True
        
        while not is_prime(n):
            n += 1
        return n
    
    def index(self, key):
        """Calculate the index for a key using modulo the table size"""
        return key % self.size
    
    def put(self, key, value):
        """
        Store a value for a key.
        
        Parameters:
        -----------
        key : int
            The key to store
        value : int
            The value to associate with the key (0 is reserved for missing entries)
        """
        # Simple direct hash with overwrite on collision
        pos = self.index(key)
        self.keys[pos] = key
        self.values[pos] = value
    
    def get(self, key):
        """
        Get the value associated with a key, or 0 if not found.
        
        Parameters:
        -----------
        key : int
            The key to look up
            
        Returns:
        --------
        int
            The value associated with the key, or 0 if not found
        """
        pos = self.index(key)
        if self.keys[pos] == key:
            return self.values[pos]
        return 0
    
    def reset(self):
        """Clear the transposition table"""
        self.keys.fill(0)
        self.values.fill(0)
    
    def memory_usage(self):
        """
        Calculate the memory usage of the table in bytes.
        
        Returns:
        --------
        int
            Memory usage in bytes
        """
        key_bytes = self.keys.itemsize * self.size
        value_bytes = self.values.itemsize * self.size
        return key_bytes + value_bytes