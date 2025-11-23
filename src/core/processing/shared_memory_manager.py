import logging
import multiprocessing.shared_memory as shm
import numpy as np
from typing import Tuple, Optional, List

log = logging.getLogger("SharedMemoryManager")

class SharedMemoryRingBuffer:
    """
    Manages a ring buffer of shared memory blocks for zero-copy frame passing.
    
    Structure:
    - Creates N blocks of shared memory, each sized for one frame.
    - Producer writes to blocks in a round-robin fashion.
    - Consumer reads from specific block indices provided via queue.
    """
    
    def __init__(
        self, 
        name_prefix: str, 
        count: int, 
        shape: Tuple[int, ...], 
        dtype: np.dtype = np.uint8,
        create: bool = True
    ):
        """
        Args:
            name_prefix: Unique prefix for shared memory blocks (e.g. 'aria_rgb')
            count: Number of buffers in the ring
            shape: Shape of the frame (height, width, channels)
            dtype: Data type of the frame
            create: True to create new memory, False to attach to existing
        """
        self.name_prefix = name_prefix
        self.count = count
        self.shape = shape
        self.dtype = np.dtype(dtype)
        
        # Calculate size in bytes
        self.frame_size = int(np.prod(shape) * self.dtype.itemsize)
        
        self.shm_blocks: List[shm.SharedMemory] = []
        self.arrays: List[np.ndarray] = []
        self.current_idx = 0
        
        self._initialize(create)
        
    def _initialize(self, create: bool) -> None:
        """Initialize shared memory blocks and numpy views."""
        for i in range(self.count):
            name = f"{self.name_prefix}_{i}"
            try:
                if create:
                    # Try to unlink if exists from previous crash
                    try:
                        temp = shm.SharedMemory(name=name)
                        temp.close()
                        temp.unlink()
                        log.warning(f"Cleaned up stale shared memory: {name}")
                    except FileNotFoundError:
                        pass
                        
                    block = shm.SharedMemory(name=name, create=True, size=self.frame_size)
                    log.info(f"Created shared memory block: {name} ({self.frame_size} bytes)")
                else:
                    block = shm.SharedMemory(name=name)
                    log.info(f"Attached to shared memory block: {name}")
                
                self.shm_blocks.append(block)
                
                # Create numpy view
                array = np.ndarray(self.shape, dtype=self.dtype, buffer=block.buf)
                self.arrays.append(array)
                
            except Exception as e:
                log.error(f"Failed to initialize shared memory {name}: {e}")
                self.cleanup()
                raise

    def put(self, frame: np.ndarray) -> int:
        """
        Write frame to the next available buffer.
        
        Returns:
            int: Index of the buffer written to.
        """
        if frame.shape != self.shape:
            raise ValueError(f"Frame shape mismatch. Expected {self.shape}, got {frame.shape}")
            
        idx = self.current_idx
        
        # Copy data to shared memory
        # np.copyto is faster than array[:] = frame
        np.copyto(self.arrays[idx], frame)
        
        # Advance index
        self.current_idx = (self.current_idx + 1) % self.count
        
        return idx

    def get(self, idx: int) -> np.ndarray:
        """
        Get a view of the frame at the specified index.
        
        Returns:
            np.ndarray: Zero-copy view of the shared memory.
        """
        if not (0 <= idx < self.count):
            raise IndexError(f"Buffer index {idx} out of range (0-{self.count-1})")
            
        return self.arrays[idx]

    def cleanup(self) -> None:
        """Close and unlink all shared memory blocks."""
        for block in self.shm_blocks:
            try:
                block.close()
                # Only unlink if we created it? 
                # Ideally the creator unlinks, but for simplicity we'll let the owner call cleanup.
                # If we are just a reader, we should probably just close.
                # But this class doesn't strictly know if it's owner or reader after init.
                # We'll assume the caller knows when to call cleanup (usually main process).
                try:
                    block.unlink()
                except FileNotFoundError:
                    pass
            except Exception as e:
                log.error(f"Error cleaning up shared memory: {e}")
        
        self.shm_blocks.clear()
        self.arrays.clear()
        log.info(f"Cleaned up shared memory: {self.name_prefix}")
