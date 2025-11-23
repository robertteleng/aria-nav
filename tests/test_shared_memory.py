import unittest
import numpy as np
import multiprocessing as mp
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from core.processing.shared_memory_manager import SharedMemoryRingBuffer

def child_reader(name, shape, dtype, idx, q):
    try:
        shm_reader = SharedMemoryRingBuffer(name, 3, shape, dtype, create=False)
        data = shm_reader.get(idx)
        # Verify data
        is_equal = np.array_equal(data, np.full(shape, 123, dtype=dtype))
        q.put(is_equal)
        shm_reader.cleanup()
    except Exception as e:
        q.put(e)

class TestSharedMemoryRingBuffer(unittest.TestCase):
    def setUp(self):
        self.name_prefix = "test_shm"
        self.count = 3
        self.shape = (100, 100, 3)
        self.dtype = np.uint8
        
        # Ensure clean state
        try:
            temp = SharedMemoryRingBuffer(self.name_prefix, self.count, self.shape, self.dtype, create=True)
            temp.cleanup()
        except:
            pass

    def test_basic_read_write(self):
        """Test writing and reading in the same process"""
        shm = SharedMemoryRingBuffer(self.name_prefix, self.count, self.shape, self.dtype, create=True)
        
        frame = np.random.randint(0, 255, self.shape, dtype=self.dtype)
        idx = shm.put(frame)
        
        read_frame = shm.get(idx)
        
        np.testing.assert_array_equal(frame, read_frame)
        
        shm.cleanup()

    def test_multiprocess_read(self):
        """Test writing in main process and reading in child process"""
        shm_writer = SharedMemoryRingBuffer(self.name_prefix, self.count, self.shape, self.dtype, create=True)
        
        frame = np.full(self.shape, 123, dtype=self.dtype)
        idx = shm_writer.put(frame)
        
        # Queue to get result from child
        q = mp.Queue()
        
        p = mp.Process(target=child_reader, args=(self.name_prefix, self.shape, self.dtype, idx, q))
        p.start()
        p.join()
        
        result = q.get()
        self.assertTrue(result, f"Child process failed to read correct data: {result}")
        
        shm_writer.cleanup()

if __name__ == '__main__':
    unittest.main()
