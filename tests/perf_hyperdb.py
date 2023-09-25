import time
import unittest
import random
import string
import psutil
import os
import numpy as np
from hyperdb import HyperDB

class HyperDBPerformanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up testing environment."""
        cls.db = HyperDB()
        cls.random_docs = [''.join(random.choices(string.ascii_uppercase + string.digits, k=50)) for _ in range(10000)]
        cls.process = psutil.Process(os.getpid())

    def measure_memory(self):
        """Measure current memory usage."""
        return self.process.memory_info().rss / 1024.0 / 1024.0  # Memory in MB
    
    def test_add_documents_performance(self):
        """Test the performance of adding documents."""
        start_time = time.time()
        start_memory = self.measure_memory()
        
        self.db.add(self.random_docs)
        
        end_time = time.time()
        end_memory = self.measure_memory()
        
        time_taken = end_time - start_time
        memory_taken = end_memory - start_memory
        
        print(f"Time taken to add 10,000 documents: {time_taken:.2f} seconds")
        print(f"Memory consumed to add 10,000 documents: {memory_taken:.2f} MB")
    
    def test_query_performance(self):
        """Test the performance of querying documents."""
        # Assuming that the database is not empty at this point
        query = "Test Query"
        
        start_time = time.time()
        start_memory = self.measure_memory()
        
        self.db.query(query)
        
        end_time = time.time()
        end_memory = self.measure_memory()
        
        time_taken = end_time - start_time
        memory_taken = end_memory - start_memory
        
        print(f"Time taken to query: {time_taken:.2f} seconds")
        print(f"Memory consumed during query: {memory_taken:.2f} MB")
    
    def test_remove_documents_performance(self):
        """Test the performance of removing documents."""
        indices_to_remove = random.sample(range(len(self.random_docs)), 1000)
        
        start_time = time.time()
        start_memory = self.measure_memory()
        
        self.db.remove_document(indices_to_remove)
        
        end_time = time.time()
        end_memory = self.measure_memory()
        
        time_taken = end_time - start_time
        memory_taken = end_memory - start_memory
        
        print(f"Time taken to remove 1,000 documents: {time_taken:.2f} seconds")
        print(f"Memory consumed to remove 1,000 documents: {memory_taken:.2f} MB")
    
    def test_save_load_performance(self):
        """Test the performance of saving and loading the database."""
        start_time = time.time()
        start_memory = self.measure_memory()
        
        self.db.save("temp_db.pickle")
        self.db.load("temp_db.pickle")
        
        end_time = time.time()
        end_memory = self.measure_memory()
        
        time_taken = end_time - start_time
        memory_taken = end_memory - start_memory
        
        print(f"Time taken to save and load the database: {time_taken:.2f} seconds")
        print(f"Memory consumed during save and load: {memory_taken:.2f} MB")
        
if __name__ == '__main__':
    unittest.main()
