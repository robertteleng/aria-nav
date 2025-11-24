# Performance Bottleneck Analysis - aria-nav

## System Overview

Real-time navigation system for visually impaired using Meta Aria glasses:
- **Hardware**: Meta Aria glasses (1408x1408 RGB camera), NVIDIA RTX 2060 GPU
- **Processing**: TensorRT YOLO + Depth Anything V2 in separate GPU worker processes
- **Architecture**: Multiprocessing with separate workers for depth estimation and object detection

## Current Performance

- **Actual FPS**: 16.3 FPS
- **Theoretical Max**: 16.5 FPS (based on 60.6ms average iteration)
- **Efficiency**: 98.6% (excellent!)
- **GPU Usage**: Only 50% utilized
- **VRAM Usage**: 1.2GB (RTX 2060 has 6GB total)

## The Problem

### Timing Breakdown (from instrumentation):

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
```

### Key Observations:

1. **92.5% of time spent waiting** for GPU workers to return results via `result_queue.get(timeout=15.0)`
2. **GPU only at 50%** despite having work to do
3. **Occasional spikes**: 17 frames exceeded 121ms (2x average)
   - Worst spike: 670ms (frame 1, model loading)
   - Regular spikes: 280-290ms (~5x average worker time)
4. **Worker processing time**: 
   - Depth inference: ~14ms
   - YOLO RGB: ~7ms  
   - YOLO SLAM: ~2-3ms
   - Total: ~30ms actual GPU work

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

### Option A: Non-blocking Pipeline with Frame Skip
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

**Pros**: Simple, guaranteed responsive
**Cons**: May repeat detections from old frames

### Option B: Async Pipeline with Future Pattern
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

**Pros**: Allows frame overlap, better GPU utilization
**Cons**: More complex, potential frame reordering

### Option C: Reduce Timeout with Cached Fallback
```python
# Short timeout, fallback to previous
try:
    result = self.result_queue.get(timeout=0.05)  # 50ms max
except queue.Empty:
    log.warning("Worker timeout, reusing previous result")
    result = self.last_result
```

**Pros**: Minimal code change, graceful degradation
**Cons**: Still blocks (just less), may drop frames under load

### Option D: Double Buffering Workers
```python
# Two worker instances per camera, alternate
workers = [CentralWorker(), CentralWorker()]
current_worker = 0

def process(frame):
    worker = workers[current_worker]
    current_worker = 1 - current_worker
    
    # Submit to one worker while other finishes
    worker.submit(frame)
    return other_worker.get_result()
```

**Pros**: True parallel processing, maximizes GPU
**Cons**: 2x memory, complex synchronization

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
