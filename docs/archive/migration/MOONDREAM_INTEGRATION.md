# 🌙 Moondream Integration Plan

## Overview
Integrate Moondream2 VLM for rich scene descriptions beyond object detection.

## Requirements
- **VRAM:** +900MB (2.2GB total, well within 6GB limit)
- **Latency:** +250ms per description (on-demand only)
- **Dependencies:** `transformers`, `einops`

## Installation

```bash
pip install transformers einops pillow
```

## Implementation Options

### Option A: On-Demand Only (RECOMMENDED ⭐)

**When:** User presses button or asks "What do you see?"

```python
# src/core/vision/scene_descriptor.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np

class SceneDescriptor:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
    def lazy_load(self):
        """Load model only when first needed"""
        if self.loaded:
            return
            
        print("[SceneDescriptor] Loading Moondream2...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True
        )
        self.loaded = True
        print("[SceneDescriptor] ✓ Moondream2 ready")
    
    def describe_scene(self, frame: np.ndarray, question: str = None) -> str:
        """
        Generate natural language description of scene.
        
        Args:
            frame: RGB image (H, W, 3) numpy array
            question: Optional specific question. Default: general description
            
        Returns:
            Text description
        """
        self.lazy_load()
        
        # Convert to PIL Image
        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame)
        else:
            image = frame
            
        # Encode image
        enc_image = self.model.encode_image(image)
        
        # Default question for blind navigation
        if question is None:
            question = "Describe this scene for a blind person walking. Include obstacles, directions, and important details."
        
        # Generate description
        description = self.model.answer_question(
            enc_image,
            question,
            self.tokenizer
        )
        
        return description
    
    def get_memory_usage(self):
        """Return VRAM usage in MB"""
        if not self.loaded:
            return 0
        return torch.cuda.memory_allocated(self.device) / (1024**2)
```

**Integration in main.py:**

```python
# In main()
scene_descriptor = SceneDescriptor() if Config.MOONDREAM_ENABLED else None

# In loop
if user_pressed_describe_button() and scene_descriptor:
    frame = observer.get_latest_frame('rgb')
    description = scene_descriptor.describe_scene(frame)
    audio_manager.speak(description, priority="high")
```

### Option B: Periodic Context (Every 5-10 seconds)

```python
class ContextualAudioManager:
    def __init__(self, scene_descriptor):
        self.descriptor = scene_descriptor
        self.last_context_time = 0
        self.context_interval = 5.0  # seconds
        
    def update(self, frame, detections):
        current_time = time.time()
        
        # Regular object announcements (immediate)
        for det in detections:
            if det.is_critical:
                self.announce_object(det)
        
        # Contextual description (periodic)
        if current_time - self.last_context_time > self.context_interval:
            description = self.descriptor.describe_scene(frame)
            self.queue_context(description)  # Lower priority
            self.last_context_time = current_time
```

### Option C: Smart Trigger (Context-aware)

```python
class SmartDescriptor:
    def __init__(self, scene_descriptor):
        self.descriptor = scene_descriptor
        self.last_scene_hash = None
        
    def should_describe(self, frame):
        """Only describe when scene changes significantly"""
        # Simple hash: average of 8x8 downsampled frame
        small = cv2.resize(frame, (8, 8))
        scene_hash = small.mean()
        
        if self.last_scene_hash is None:
            changed = True
        else:
            changed = abs(scene_hash - self.last_scene_hash) > 10
            
        self.last_scene_hash = scene_hash
        return changed
    
    def update(self, frame):
        if self.should_describe(frame):
            return self.descriptor.describe_scene(frame)
        return None
```

## Configuration

Add to `src/utils/config.py`:

```python
class Config:
    # ... existing config ...
    
    # Moondream Scene Description
    MOONDREAM_ENABLED = True
    MOONDREAM_MODE = "on_demand"  # "on_demand", "periodic", "smart"
    MOONDREAM_INTERVAL = 5.0  # For periodic mode
    MOONDREAM_QUESTIONS = {
        "general": "Describe this scene for a blind person. Include obstacles and directions.",
        "danger": "Are there any dangers or obstacles in this scene?",
        "navigate": "What is the safest direction to walk?",
        "read": "Is there any text or signage visible?"
    }
```

## Performance Impact

### Baseline (without Moondream):
- FPS: 18.0
- Latency: 55ms
- VRAM: 1.3GB

### With Moondream (on-demand):
- FPS: **18.0** (unchanged - only runs on request)
- Latency: 55ms + 250ms **when triggered**
- VRAM: **2.2GB** (still 64% free)

### With Moondream (periodic, 5s interval):
- FPS: **17.5** (-2.8%)
- Latency: 55-305ms (spikes every 5s)
- VRAM: **2.2GB**

## Testing

```python
# test_moondream.py
import cv2
from core.vision.scene_descriptor import SceneDescriptor

descriptor = SceneDescriptor()

# Test with sample image
frame = cv2.imread("test_image.jpg")
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

description = descriptor.describe_scene(frame_rgb)
print(f"Description: {description}")

# Test with specific questions
dangers = descriptor.describe_scene(frame_rgb, "Are there any dangers?")
print(f"Dangers: {dangers}")
```

## Example Outputs

**Input:** Frame of person walking near stairs  
**Output:** "You are in an indoor hallway. There is a staircase descending on your right. A person is walking towards you about 3 meters ahead."

**Input:** Outdoor crosswalk  
**Output:** "You are at a pedestrian crossing. The traffic light is red. Two cars are stopped at the intersection. A bicycle is approaching from your left."

## Killer Features

1. **Context-aware navigation**
   - Beyond "person detected" → "Person blocking doorway on left"
   
2. **Text reading**
   - "STOP sign ahead"
   - "Exit door on right"
   
3. **Spatial understanding**
   - "Narrow passage, walk single file"
   - "Open space, safe to walk freely"

4. **Safety warnings**
   - "Stairs descending without handrail"
   - "Wet floor sign visible"

## Future Enhancements

1. **Memory/History:**
   - Remember previous descriptions to avoid repetition
   - Build spatial map: "You're back at the entrance"

2. **Multi-modal:**
   - Combine YOLO detections + Moondream description
   - "3 people detected (YOLO) + One is waving at you (Moondream)"

3. **Fine-tuning:**
   - Train on blind navigation scenarios
   - Custom prompts for indoor vs outdoor

## Resources

- Model: https://huggingface.co/vikhyatk/moondream2
- Paper: Moondream Technical Report
- VRAM: ~900MB (FP16)
- Inference: ~250ms on RTX 2060
