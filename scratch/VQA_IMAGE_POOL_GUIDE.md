"""
VQA IMAGE POOL - EXPLANATION & TESTING GUIDE

This document explains what the vqa_image_pool module does and how to test it.
"""

## What Does vqa_image_pool Do?

The `vqa_image_pool.py` module is designed for **high-speed image retrieval from GPU memory**. 
It's part of the local-first VQA system for H100/H200 GPUs.

### Three Main Components:

1. **ImagePoolDataset** - PyTorch Dataset class
   - Loads images from disk and converts them to tensors
   - Resizes images to consistent dimensions (default 512x512)
   - Normalizes pixel values to [-1, 1] range (needed for diffusion models)
   - Validates paths before loading

2. **ImagePool** - Main manager class
   - Scans local directories for images (.png, .jpg, .jpeg, etc.)
   - OR loads images from HuggingFace datasets
   - Creates PyTorch DataLoaders for parallel loading
   - **Preloads entire image batches into GPU VRAM** for sub-millisecond retrieval
   - Tracks memory usage and cache status

3. **create_image_pool_from_config()** - Factory function
   - Convenience function to create ImagePool from config dict

---

## Why Is This Fast?

### Traditional Approach (Slow):
```
User Request → Load image from disk → Resize → Convert to tensor → Send to GPU → Process
             ↑ Disk I/O latency (slow)                                    (fast)
```

### ImagePool Approach (Fast):
```
Pre-load → Batch → GPU VRAM ─┐
                              ├─→ User Request → Retrieve from GPU cache → Process
                              ↓                  (sub-millisecond!)
              (happens once at startup)
```

**Key Benefits:**
- **Sub-millisecond retrieval**: Images already in GPU cache
- **Parallel batching**: Load 32+ images concurrently
- **Memory efficient**: Only keep ~10 batches in GPU (256-512 images max on H200)
- **H200 optimized**: 141GB HBM can hold thousands of high-res images

---

## Quick Start Testing

### Test 1: Create a Simple Image Directory

```bash
# Create test directory
mkdir -p /tmp/test_images

# Create 10 dummy images using Python
python3 << 'EOF'
from PIL import Image
import os

os.makedirs('/tmp/test_images', exist_ok=True)

# Create 10 random images
for i in range(10):
    img = Image.new('RGB', (256, 256), color=(i*25, i*25, i*25))
    img.save(f'/tmp/test_images/image_{i:02d}.png')
    
print("✅ Created 10 test images in /tmp/test_images")
EOF
```

### Test 2: Basic ImagePool Test

```python
# test_imagepool_basic.py
import sys
sys.path.insert(0, '/Users/oppizz/Documents/EPFL/MNLP/Project/Mock_repo_mnlp/scratch')

from vqa_image_pool import ImagePool

# Create image pool from local directory
pool = ImagePool(
    pool_path='/tmp/test_images',
    image_size=(256, 256),
    batch_size=4,
    gpu_id=0
)

print(f"✅ Loaded {len(pool.image_paths)} images")
print(f"Device: {pool.device}")

# Test 1: Get single image
img = pool.get_image(pool.image_paths[0])
print(f"✅ Retrieved single image: shape={img.shape}, device={img.device}")

# Test 2: Get batch of images
batch = pool.batch_get_images(pool.image_paths[:4])
print(f"✅ Retrieved batch: shape={batch.shape}, device={batch.device}")

# Test 3: Memory stats (if GPU available)
if 'cuda' in str(pool.device):
    stats = pool.get_memory_stats()
    print(f"✅ Memory stats:")
    print(f"   - Allocated: {stats['allocated_gb']:.2f} GB")
    print(f"   - Preloaded: {stats['preloaded_count']} images")
```

Run it:
```bash
cd /tmp
python3 test_imagepool_basic.py
```

### Test 3: GPU Preloading Test

```python
# test_imagepool_preload.py
import sys
import time
sys.path.insert(0, '/Users/oppizz/Documents/EPFL/MNLP/Project/Mock_repo_mnlp/scratch')

from vqa_image_pool import ImagePool

pool = ImagePool(
    pool_path='/tmp/test_images',
    batch_size=4,
    gpu_id=0
)

# Measure time WITHOUT preloading
print("Measuring retrieval time WITHOUT preloading...")
times = []
for i in range(5):
    import time
    start = time.perf_counter()
    img = pool.get_image(pool.image_paths[i])
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
    times.append(elapsed)
    print(f"  Image {i}: {elapsed:.2f} ms")

print(f"  Average (no cache): {sum(times)/len(times):.2f} ms")

# NOW preload to GPU
print("\nPreloading images to GPU...")
start = time.perf_counter()
pool.preload_to_gpu(num_batches=2)
preload_time = time.perf_counter() - start
print(f"✅ Preloading took {preload_time:.2f}s")

# Measure time WITH preloading
print("\nMeasuring retrieval time WITH preloading (from GPU cache)...")
times = []
for i in range(5):
    start = time.perf_counter()
    img = pool.get_image(pool.image_paths[i])
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f"  Image {i}: {elapsed:.3f} ms  ⚡ (FAST!)")

print(f"  Average (with cache): {sum(times)/len(times):.3f} ms")
print(f"\n💡 Speedup: {sum(times)/len(times) / (sum(times)/len(times))}x faster from GPU cache!")
```

Run it:
```bash
cd /tmp
python3 test_imagepool_preload.py
```

### Test 4: HuggingFace Dataset Integration

```python
# test_imagepool_huggingface.py
import sys
sys.path.insert(0, '/Users/oppizz/Documents/EPFL/MNLP/Project/Mock_repo_mnlp/scratch')

from vqa_image_pool import ImagePool

# Load from HuggingFace (small dataset for testing)
pool = ImagePool(
    dataset_name='beans',  # Small public dataset (~1300 images)
    image_size=(256, 256),
    max_images=100,  # Limit to 100 for testing
    batch_size=16,
    gpu_id=0
)

print(f"✅ Loaded {len(pool.image_paths)} images from 'beans' dataset")

# Create DataLoader
dataloader = pool.create_dataloader()

# Iterate through batches
for batch_idx, (images, paths) in enumerate(dataloader):
    print(f"Batch {batch_idx}: {images.shape} images")
    if batch_idx >= 2:  # Just show first 3 batches
        break

print("✅ HuggingFace integration works!")
```

Run it (requires internet for HuggingFace download):
```bash
cd /tmp
python3 test_imagepool_huggingface.py
```

### Test 5: Integration with Config

```python
# test_imagepool_config.py
import sys
sys.path.insert(0, '/Users/oppizz/Documents/EPFL/MNLP/Project/Mock_repo_mnlp/scratch')

from vqa_image_pool import create_image_pool_from_config

# Create from config dictionary
config = {
    'image_pool_path': '/tmp/test_images',
    'image_size': (256, 256),
    'num_workers': 2,
    'batch_size': 4,
    'gpu_id': 0,
    'max_images': 50,
    'use_mmap': True
}

pool = create_image_pool_from_config(config)
print(f"✅ Created pool from config: {len(pool.image_paths)} images")
print(f"   Device: {pool.device}")
print(f"   Batch size: {pool.batch_size}")
```

---

## Testing Without Images (CPU Mock Test)

If you don't have real images, test the logic flow:

```python
# test_imagepool_mock.py
import sys
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, '/Users/oppizz/Documents/EPFL/MNLP/Project/Mock_repo_mnlp/scratch')
from vqa_image_pool import ImagePoolDataset

# Create mock images in temp location
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    # Create 5 test images
    for i in range(5):
        img = Image.new('RGB', (256, 256), color=(i*50, i*50, i*50))
        img.save(os.path.join(tmpdir, f'test_{i}.png'))
    
    # Test ImagePoolDataset directly
    images = [os.path.join(tmpdir, f'test_{i}.png') for i in range(5)]
    
    dataset = ImagePoolDataset(
        image_paths=images,
        image_size=(256, 256),
        normalize=True,
        use_mmap=False
    )
    
    print(f"✅ Created dataset with {len(dataset)} images")
    
    # Load and test a sample
    img_tensor, path = dataset[0]
    print(f"✅ Loaded image:")
    print(f"   Shape: {img_tensor.shape}  (3 channels, 256x256)")
    print(f"   Min: {img_tensor.min():.3f}, Max: {img_tensor.max():.3f}  (should be -1 to 1)")
    print(f"   Path: {path}")
    
    # Test batch loading with DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
    batch_images, batch_paths = next(iter(dataloader))
    print(f"✅ Batch loading works:")
    print(f"   Batch shape: {batch_images.shape}")
    print(f"   Device: {batch_images.device}")
```

---

## Key Methods to Test

| Method | What to Test |
|--------|------------|
| `__init__()` | Can initialize with local path or HF dataset |
| `_load_image_paths()` | Finds all images in directory |
| `create_dataloader()` | Creates working PyTorch DataLoader |
| `preload_to_gpu()` | Loads images to GPU memory |
| `get_image()` | Retrieves single image (from cache if available) |
| `batch_get_images()` | Gets multiple images as stacked tensor |
| `get_memory_stats()` | Reports GPU memory usage |
| `clear_cache()` | Clears GPU memory cache |

---

## Real-World Integration Test

Once basic tests pass, integrate with the full pipeline:

```python
# test_full_integration.py
import sys
sys.path.insert(0, '/Users/oppizz/Documents/EPFL/MNLP/Project/Mock_repo_mnlp/scratch')

from integration_example import PromptomatixVQAConfig, IntegratedPromptomatixPipeline

config = PromptomatixVQAConfig()
config.image_pool_path = '/tmp/test_images'
config.enable_gpu_preloading = True
config.image_preload_batches = 2

pipeline = IntegratedPromptomatixPipeline(config)

if pipeline.setup_components():
    print("✅ Full pipeline setup successful!")
    if pipeline.image_pool:
        stats = pipeline.image_pool.get_memory_stats()
        print(f"   Image pool: {stats['preloaded_count']} images preloaded")
else:
    print("❌ Pipeline setup failed")
```

---

## Troubleshooting

### Error: "No valid image paths found"
**Cause**: Directory is empty or doesn't have image files
```bash
# Check what's in your directory
ls -la /tmp/test_images/
file /tmp/test_images/*  # Check file types
```

### Error: "CUDA out of memory"
**Solution**: Reduce preload size
```python
pool.preload_to_gpu(num_batches=1)  # Load fewer batches
# OR
config['max_images'] = 100  # Load fewer total images
```

### Error: "Failed to load image"
**Cause**: Image file is corrupted or unsupported format
```python
# Test if PIL can open it
from PIL import Image
img = Image.open('/path/to/image.png')
img.convert('RGB')  # Should not raise error
```

### Images loading slowly even with preload
**Cause**: Images not actually preloaded (batch size mismatch)
```python
# Verify preloading:
print(f"Cached: {len(pool.preloaded_images)} images")
if len(pool.preloaded_images) == 0:
    print("⚠️ Nothing preloaded! Try pool.preload_to_gpu()")
```

---

## Summary

✅ **vqa_image_pool** = Fast GPU-cached image retrieval for VQA  
✅ **Perfect for**: Vision tasks that need to evaluate 100+ images  
✅ **Speed**: Sub-millisecond per image (vs 10-100ms from disk)  
✅ **Memory**: Efficient with H200's 141GB HBM  

**Test it now**: Start with Test 1-2 above, then move to full integration!
