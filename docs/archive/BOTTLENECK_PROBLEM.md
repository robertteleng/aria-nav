# Performance Bottleneck Analysis - aria-nav

## System Overview

Real-time navigation system for visually impaired using Meta Aria glasses:
- **Hardware**: Meta Aria glasses (1408x1408 RGB camera), NVIDIA RTX 2060 GPU
- **Processing**: TensorRT YOLO + Depth Anything V2 in separate GPU worker processes
- **Architecture**: Multiprocessing with separate workers for depth estimation and object detection

## Current Performance (After Optimization)

- **Actual FPS**: 16.7 FPS (was 16.3)
- **Theoretical Max**: 16.9 FPS (based on 59.1ms average iteration)
- **Efficiency**: 98.7% (excellent!)
- **GPU Usage**: Only 50% utilized (underutilized - bottleneck is architectural)
- **VRAM Usage**: 1.2GB (RTX 2060 has 6GB total - room for double buffering)

## The Problem

### Timing Breakdown (from instrumentation - Latest):

**Baseline (before optimization):**
```
Component                           Time (ms)    % of Total
=============================================================
Observer (get frames from Aria)     0.32ms       0.5%
Build frames dict                   0.25ms       0.4%
Pipeline.process (GPU workers)      56.08ms      92.5%  ← BOTTLENECK
  └─ Waiting on result_queue.get()
SLAM handling                       0.03ms       0.0%
Get depth/events                    0.05ms       0.1%
Render SLAM overlays                0.81ms       1.3%
PresentationManager.update_display  3.12ms       5.1%
=============================================================
TOTAL per frame                     60.65ms      100%
Efficiency                          98.6%
```

**After Solution #3 (non-blocking queues):**
```
Component                           Time (ms)    % of Total
=============================================================
Observer (get frames from Aria)     0.33ms       0.6%
Build frames dict                   0.23ms       0.4%
Pipeline.process (GPU workers)      54.64ms      92.4%  ← IMPROVED -2.6%
  └─ Non-blocking collection
SLAM handling                       0.03ms       0.0%
Get depth/events                    0.05ms       0.1%
Render SLAM overlays                0.87ms       1.5%
PresentationManager.update_display  2.98ms       5.0%
=============================================================
TOTAL per frame                     59.12ms      100%
Efficiency                          98.7%
Improvement                         -1.53ms (-2.5%)
```

### Key Observations:

1. **92.4% of time spent waiting** for GPU workers to return results
2. **GPU only at 50%** despite having work to do (architectural limitation)
3. **IPC overhead**: ~20-25ms gap between actual GPU work (30-35ms) and main loop wait (54ms)
4. **Worker processing time** (measured): 
   - Depth inference: ~14-20ms
   - YOLO RGB: ~7ms  
   - YOLO SLAM: ~5ms
   - Total GPU work: ~30-35ms
   - Total with IPC: ~54ms (36% overhead)
5. **Occasional spikes**: 36 frames exceeded 118ms (2x average)
   - Worst spike: 576ms (frame 1, model loading)
   - Regular spikes: 280-290ms (~5x average)

## Architecture Details

### Current Flow:

```python
# Main loop (main.py)
while not stop:
    # 1. Get frames from Aria SDK (0.3ms)
    frame = observer.get_latest_frame('rgb')
    slam1 = observer.get_latest_frame('slam1')
    slam2 = observer.get_latest_frame('slam2')
    
    # 2. Process through pipeline (56ms) ← BLOCKS HERE
    result = coordinator.process_frame(frame, motion, frames_dict)
    
    # 3. Update display (3ms)
    presentation.update_display(result, ...)
```

### Pipeline Process:

```python
# navigation_pipeline.py
def process(self, frame, frames_dict):
    # Enqueue frames to workers (instant)
    self._enqueue_frames(frame_id, frames_dict)
    
    # Collect results - BLOCKS HERE (56ms avg, 280ms spikes)
    results = self._collect_results_with_overlap(frame_id)
    
    # Merge results (instant)
    return self._merge_results(results, frames_dict)
```

### Worker Architecture:

```python
# Separate GPU processes
CentralWorker (depth + YOLO RGB):  ~30ms processing
SlamWorker (YOLO SLAM1 + SLAM2):   ~5ms processing

# Queue flow:
main → central_queue → CentralWorker → result_queue → main (BLOCKED)
main → slam_queue → SlamWorker → result_queue → main
```

### Blocking Code:

```python
def _collect_results_with_overlap(self, frame_id):
    # First: drain available results
    while True:
        try:
            result = self.result_queue.get_nowait()
            available_results.append(result)
        except queue.Empty:
            break
    
    # Then: BLOCK waiting for central worker
    if "central" not in self.pending_results:
        central_result = self.result_queue.get(timeout=15.0)  ← BLOCKS 56ms!
        self.pending_results[central_result.camera] = central_result
    
    return self.pending_results
```

## The Paradox

- **GPU has capacity**: Only 50% utilized, 1.2GB/6GB VRAM
- **Workers process fast**: ~30ms for depth+YOLO
- **Main loop wastes time**: 56ms waiting despite 30ms actual work
- **Queue is synchronous**: Main thread IDLE while workers process

## Why This Happens

1. **Synchronous pipeline**: Main loop can't proceed without central worker result
2. **Sequential processing**: One frame at a time, no overlap
3. **Queue overhead**: Serialization/deserialization adds latency
4. **Worker idle time**: After finishing, workers wait for next frame
5. **No frame pipelining**: Can't start frame N+1 while N is processing

## Potential Solutions

### ✅ Option A: Non-blocking Pipeline with Small Queues (IMPLEMENTED)
```python
# Reduce queue sizes and use non-blocking operations
self.central_queue = mp.Queue(maxsize=2)  # was 4
self.result_queue = mp.Queue(maxsize=4)

def _enqueue_frames():
    self.central_queue.put_nowait(frame)  # No blocking

def _collect_results():
    # Drain available results
    while True:
        result = self.result_queue.get_nowait()
        self.cached_result = result
    
    # Use cached result if no new one
    return self.cached_result
```

**Results (Implemented - Commit 4d4baf9):**
- ✅ FPS: 16.3 → 16.7 (+2.5%)
- ✅ Latency: 60.65ms → 59.12ms (-2.5%)
- ✅ Pipeline.process: 56.08ms → 54.64ms (-2.6%)
- ✅ Better spike handling
- ❌ Limited improvement (architectural bottleneck remains)

### Option B: Double Buffering Workers (RECOMMENDED NEXT STEP)
```python
# Two worker instances per camera, alternate
workers_a = [CentralWorker(), SlamWorker()]
workers_b = [CentralWorker(), SlamWorker()]
current_worker_set = 0

def process(frame):
    # Submit to one set
    if current_worker_set == 0:
        submit_to_workers_a(frame_n+1)
        result = get_from_workers_b(frame_n)  # Ready
    else:
        submit_to_workers_b(frame_n+1)
        result = get_from_workers_a(frame_n)  # Ready
    
    current_worker_set = 1 - current_worker_set
    return result
```

**Expected Results:**
- 🎯 FPS: 16.7 → 24-27 (+40-60%)
- 🎯 Latency: 59ms → 40-45ms (-24%)
- 🎯 GPU utilization: 50% → 80-90%
- ⚠️ VRAM: 1.2GB → 2.4GB (fits in 6GB)
- ⚠️ Complexity: Worker synchronization, crash handling

### ❌ Option C: SharedMemory Zero-Copy (FAILED)
```python
# Attempt: Avoid serialization overhead with shared memory
SharedMemoryRingBuffer for zero-copy frame passing
```

**Results (Tested - REVERTED):**
- ❌ FPS: 16.3 → 10.4 (-36%)
- ❌ Latency: 60ms → 96ms (+60%)
- ❌ Spike: 19.4 seconds (catastrophic)
- 🐛 Root cause: Race conditions (no locks), buffer count mismatch
- 🐛 Thread safety issues between put/get operations

### Option D: Async Pipeline with Future Pattern
```python
# Don't wait, use last available result
def _collect_results_nonblocking(self):
    try:
        result = self.result_queue.get_nowait()
        self.cached_result = result
    except queue.Empty:
        pass  # Use cached result
    
    return self.cached_result  # May be from previous frame
```

### Option D: Async Pipeline with Future Pattern
```python
# Submit frame, get Future, don't wait immediately
def process_async(self, frame):
    future = self._submit_to_workers(frame)
    self.pending_futures.append(future)
    
    # Check if older futures are ready
    for fut in self.pending_futures:
        if fut.ready():
            return fut.get()
    
    # Fallback: wait for oldest
    return self.pending_futures[0].get(timeout=0.1)
```

**Status:** Not tested
**Pros**: Allows frame overlap, better GPU utilization
**Cons**: More complex, potential frame reordering, similar to double buffering

### Option E: Optimize IPC Overhead
```python
# Ideas not tested:
1. Use msgpack instead of pickle (faster serialization)
2. Compress depth maps before sending
3. Pass only detections, not full frames
4. Use faster IPC (pipes, sockets)
```

**Status:** Not tested
**Expected:** ~5-10ms reduction (8-15% FPS gain)
**Effort:** Medium, without addressing core architectural issue

## Questions for Evaluation

1. **Which approach best balances responsiveness vs. accuracy?**
   - Navigation system needs real-time feedback
   - But also needs accurate obstacle detection

2. **Is frame overlap acceptable?**
   - Can we process frame N+1 while N is still rendering?
   - Does Aria SDK provide frames faster than we can process?

3. **How to handle spike scenarios?**
   - When worker takes 280ms (5x normal), should we skip that frame?
   - Or wait and accept temporary FPS drop?

4. **Memory vs. throughput tradeoff?**
   - Double buffering uses 2x GPU memory
   - But could achieve true 30 FPS (Aria's native rate)

5. **Is 16 FPS sufficient for the use case?**
   - Maybe the bottleneck doesn't matter if user experience is good
   - Audio feedback latency is ~50ms (acceptable)

## Constraints

- **Must maintain safety**: Can't miss critical obstacles
- **Real-time audio feedback**: <100ms latency preferred
- **GPU memory**: Limited to 6GB VRAM total
- **Meta Aria SDK**: Provides frames at 30 FPS natively
- **User experience**: Smooth, not janky

## Current Working Theory

The bottleneck is **architectural**, not computational:
- GPU has spare capacity (50% usage)
- Workers finish in 30ms but main waits 56ms
- The gap (26ms) is queue synchronization overhead
- Solution requires **redesigning the pipeline flow**, not optimizing GPU code

## What's the Best Architecture for This?

Given:
- Fast GPU workers (30ms)
- 30 FPS input stream (Aria)
- Need for real-time audio feedback
- Safety-critical application

**What pipeline pattern would you recommend?**
