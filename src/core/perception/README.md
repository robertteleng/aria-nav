# Scene Understanding Module

Vision-language scene description for Scene Aria System using FastVLM-0.5B.

## Quick Start

```python
from scene_understanding import FastVLMWrapper

# Initialize wrapper
wrapper = FastVLMWrapper()

# Warmup (important for performance)
wrapper.warmup()

# Describe scene
description = wrapper.describe_scene("frame.jpg")
print(description)
# Output: "Person ahead wearing backpack, white cane visible"
```

## Performance

**RTX 2060 (Turing):**
- Model latency: 372ms (mean), 374ms (p95)
- End-to-end: 752ms ✅ Industry standard compliant
- VRAM: 1.19 GB allocated, 1.39 GB reserved

**Industry comparison:**
- Envision Glasses: 700-1200ms
- OrCam MyEye: 900ms+
- WeWalk, Biped NOA: 600-900ms

## Configuration

```python
wrapper = FastVLMWrapper(
    model_id="apple/FastVLM-0.5B",
    device="cuda",
    torch_dtype=torch.float16,
    max_new_tokens=16,  # Optimal: complete descriptions
    use_torch_compile=True,  # Overhead on Turing but future-compatible
    verbose=True
)
```

## Usage Examples

### Basic Usage

```python
from scene_understanding import describe_image

# One-liner (loads model each time)
description = describe_image("scene.jpg")
```

### Advanced Usage

```python
from scene_understanding import FastVLMWrapper
from pathlib import Path

# Initialize once, reuse
wrapper = FastVLMWrapper(verbose=True)
wrapper.warmup(num_iterations=3)

# Process multiple images
for image_path in Path("frames/").glob("*.jpg"):
    desc, latency = wrapper.describe_scene(
        image_path,
        prompt="Describe briefly.",
        return_latency=True
    )
    print(f"{image_path.name}: {desc} ({latency:.0f}ms)")

# Get performance stats
stats = wrapper.get_stats()
print(f"Average latency: {stats['average_latency_ms']:.1f}ms")
print(f"VRAM usage: {stats['vram_allocated_gb']:.2f}GB")
```

### Custom Prompts

```python
# Navigation-focused
desc = wrapper.describe_scene(
    "street.jpg",
    prompt="Describe obstacles and navigation hazards."
)

# Detail-focused
desc = wrapper.describe_scene(
    "room.jpg",
    prompt="What objects and people are present?"
)
```

## API Reference

### FastVLMWrapper

**Methods:**

- `__init__(...)`: Initialize wrapper with configuration
- `describe_scene(image, prompt, return_latency)`: Generate description
- `warmup(num_iterations)`: Warmup model (important!)
- `get_stats()`: Get performance statistics
- `reset_stats()`: Reset performance counters

**Returns:**
- `describe_scene()`: String description, or (description, latency_ms) tuple

### describe_image()

Convenience function for one-off usage.

```python
describe_image(
    image_path: str,
    model_id: str = "apple/FastVLM-0.5B",
    prompt: str = "Describe this image briefly.",
    verbose: bool = False
) -> str
```

## Technical Details

**Optimizations Applied:**
- ✅ PyTorch FP16 precision
- ✅ CUDA backend
- ✅ torch.backends.cudnn.benchmark = True
- ✅ torch.backends.cuda.matmul.allow_tf32 = True
- ✅ torch.compile (overhead on Turing, kept for compatibility)
- ✅ use_cache=False (neutral for 16 tokens)

**Why 16 tokens?**
- 8 tokens: Cuts mid-sentence ("The image you've provided is a solid")
- 16 tokens: Complete descriptions ("Person ahead wearing backpack, white cane visible")
- 32 tokens: Unnecessary details, adds latency

**Decision validated by:**
- Internal benchmarks (372ms model, 752ms E2E)
- External expert review (ChatGPT)
- Industry standard compliance

## Integration with Scene Aria System

```python
# In your main pipeline
from scene_understanding import FastVLMWrapper

class SceneAriaSystem:
    def __init__(self):
        self.wake_detector = WakeWordDetector()
        self.camera = AriaCameraCapture()
        self.scene_describer = FastVLMWrapper()
        self.scene_describer.warmup()
        self.tts = TTSEngine()
    
    def on_wake_word(self):
        # Capture frame
        frame = self.camera.get_snapshot()
        
        # Describe scene
        description = self.scene_describer.describe_scene(frame)
        
        # Speak description
        self.tts.speak(description)
```

## Future: Hybrid Strategy

Intelligent switching between FastViT (420ms) and FastVLM (752ms):

```python
class HybridSceneUnderstanding:
    def __init__(self):
        self.fastvit = FastViTClassifier()  # 110ms
        self.fastvlm = FastVLMWrapper()     # 372ms
    
    def describe(self, image, mode="auto"):
        # Always classify first
        scene_class = self.fastvit.classify(image)
        
        if mode == "quick":
            return self.template(scene_class)  # 420ms total
        
        elif self.is_complex(scene_class):
            return self.fastvlm.describe_scene(image)  # 752ms total
        
        else:
            return self.template(scene_class)  # 420ms total
    
    def is_complex(self, scene_class):
        """Scenes requiring detailed VLM description"""
        return scene_class in [
            "street_crossing",
            "public_transport",
            "crowded_space",
            "stairs_escalator"
        ]
```

**Benefits:**
- 60-70% cases: FastViT 420ms
- 30-40% cases: FastVLM 752ms
- Average: ~550ms (27% improvement)
- Maintains VLM quality where critical

## Troubleshooting

**CUDA out of memory:**
- Reduce max_new_tokens to 8 or 12
- Close other GPU processes

**Slow first inference:**
- Always call `warmup()` after initialization
- torch.compile needs warmup to optimize graph

**Import errors:**
```bash
pip install torch transformers pillow loguru
```

## License

Part of Scene Aria System - MIT License
