# Multimodal Synthetic Data Generation

## Overview

The **Multimodal Synthetic Data Generation** module (`multimodal_synthetic_data.py`) generates high-quality Visual Question Answering (VQA) datasets using local Vision Language Models (VLMs). It leverages vLLM's continuous batching and paged attention mechanisms for parallel generation with optional self-consistency checking across multiple VLM models.

## Key Features

### 1. **Local VLM Integration**
- Supports multiple Vision Language Models (Qwen2-VL-7B, LLaVA-Next-7B)
- Runs on H100/H200 GPUs with vLLM inference servers
- Configurable endpoints and model parameters

### 2. **Batch Parallel Generation**
- ThreadPoolExecutor-based parallel VQA pair generation
- Configurable batch size for throughput optimization
- Timeout handling and graceful error recovery

### 3. **Self-Consistency Checking**
- Cross-VLM verification of generated answers
- Semantic similarity scoring using word overlap
- Confidence-based filtering with adjustable thresholds

### 4. **Domain-Specific Templates**
- Difficulty levels: easy, medium, hard
- Visual categories: general, scene, object, people, action, text
- Customizable prompts for targeted generation

### 5. **Quality Assurance**
- Deduplication based on question similarity
- Confidence scoring for each VQA pair
- Metadata tracking (generation time, tokens, quality metrics)

### 6. **Flexible Export**
- JSON format with metadata
- JSONL format for streaming/distributed training
- Compatible with standard ML frameworks

## Core Classes

### `VQAPair`
Represents a single Visual Question Answering pair.

```python
@dataclass
class VQAPair:
    image_id: str           # Unique image identifier
    question: str           # Generated question
    answer: str             # VLM-generated answer
    confidence: float       # Consistency check score (0-1)
    visual_category: str    # Category of visual content
    difficulty: str         # Question difficulty: easy/medium/hard
```

### `SyntheticVQADataset`
Container for the generated VQA dataset with metadata.

```python
@dataclass
class SyntheticVQADataset:
    pairs: List[VQAPair]              # List of VQA pairs
    generation_time_sec: float        # Total generation time
    total_tokens_generated: int       # Approximate token count
    quality_filter_applied: bool      # Whether consistency filtering was used
    average_confidence: float         # Mean confidence score
```

### `LocalVLMSyntheticGenerator`
Main generator class for creating VQA pairs.

## Usage Examples

### Basic Setup
```python
from multimodal_synthetic_data import LocalVLMSyntheticGenerator

generator = LocalVLMSyntheticGenerator(
    vllm_base_url="http://localhost",
    primary_vlm="qwen2-vl-7b",
    use_self_consistency=True,
    consistency_threshold=0.8,
    batch_size=32,
    timeout=60
)
```

### Generate Single VQA Pair
```python
vqa_pair = generator.generate_vqa_pair(
    image_id="img_001",
    image_data="https://example.com/image.jpg",  # URL or base64
    visual_category="object",
    difficulty="medium"
)

print(f"Q: {vqa_pair.question}")
print(f"A: {vqa_pair.answer}")
print(f"Confidence: {vqa_pair.confidence}")
```

### Batch Generation
```python
dataset = generator.generate_batch(
    image_ids=["img_001", "img_002", "img_003"],
    image_data_list=[
        "https://example.com/img1.jpg",
        "https://example.com/img2.jpg",
        "https://example.com/img3.jpg"
    ],
    categories=["scene", "object", "people"],
    difficulties=["easy", "medium", "hard"]
)

print(f"Generated {len(dataset.pairs)} VQA pairs")
print(f"Generation time: {dataset.generation_time_sec:.2f}s")
print(f"Average confidence: {dataset.average_confidence:.2f}")
```

### Deduplication
```python
unique_pairs = generator.deduplicate_pairs(
    dataset.pairs,
    similarity_threshold=0.85
)
```

### Export Dataset
```python
# Export to JSON with metadata
generator.export_dataset(
    dataset,
    output_path="vqa_dataset.json",
    format="json"
)

# Or export to JSONL for streaming
generator.export_dataset(
    dataset,
    output_path="vqa_dataset.jsonl",
    format="jsonl"
)
```

## Configuration

### VLM Endpoints
```python
VLM_CONFIGS = {
    'qwen2-vl-7b': {
        'port': 8000,
        'model_name': 'Qwen/Qwen2-VL-7B-Instruct',
        'max_tokens': 256
    },
    'llava-next-7b': {
        'port': 8001,
        'model_name': 'llava-hf/llava-1.5-7b-hf',
        'max_tokens': 256
    }
}
```

### Generation Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `vllm_base_url` | `http://localhost` | Base URL for vLLM servers |
| `primary_vlm` | `qwen2-vl-7b` | Primary VLM model to use |
| `use_self_consistency` | `True` | Enable cross-VLM consistency checking |
| `consistency_threshold` | `0.8` | Minimum consistency score (0-1) |
| `batch_size` | `32` | Number of parallel workers |
| `timeout` | `60` | Request timeout in seconds |

## Visual Categories

- **general**: General visual content
- **scene**: Outdoor scenes, landscapes, environments
- **object**: Specific objects, properties, composition
- **people**: Human subjects, expressions, actions, interactions
- **action**: Dynamic actions, events, motion
- **text**: Visible text content in images

## Difficulty Levels

### Easy
- Simple and direct questions
- Asking about obvious visual elements
- Single-step reasoning

### Medium
- Requiring some reasoning about relationships
- Questions about attributes and composition
- Multi-element scene analysis

### Hard
- Deep reasoning and multi-step analysis
- Common sense or background knowledge required
- Complex spatial relationships or abstract concepts

## Quality Assurance

### Consistency Checking
The generator can verify VQA pair quality by asking secondary VLMs the same question:
1. Primary VLM generates Q&A pair
2. Secondary VLMs answer the same question
3. Word overlap computed for each secondary answer
4. Average consistency score determines acceptance

Pairs with consistency ≥ `consistency_threshold` are included in the dataset.

### Deduplication
Removes near-duplicate VQA pairs based on question similarity:
- Uses Jaccard similarity on tokenized questions
- Threshold of 0.85 by default
- Preserves diverse question coverage

## Performance Considerations

### Throughput
- Batch size 32 typical for H100 GPU
- Parallel generation with ThreadPoolExecutor
- Timeout protection against slow requests

### Token Efficiency
- Max tokens: 256 per VQA pair
- Typical generation: 100-150 tokens per pair
- Total tracked in `SyntheticVQADataset.total_tokens_generated`

### Memory
- Base64 image encoding for HTTP payload
- No image caching (stream-based processing)
- Minimal metadata overhead

## Error Handling

The generator includes robust error handling:
- Request timeouts (returns `None` for failed pairs)
- JSON parsing failures (logs and continues)
- VLM connection errors (fallback to primary model)
- Batch continuation on individual failures

Failed generations are skipped; the batch completes with available pairs.

## Integration with Image Pool

Combine with `vqa_image_pool.py` for end-to-end synthetic dataset generation:

```python
from vqa_image_pool import ImagePool
from multimodal_synthetic_data import LocalVLMSyntheticGenerator

# Load images
pool = ImagePool(dataset_name='beans', max_images=100)
dataloader = pool.create_dataloader()

# Generate VQA pairs
generator = LocalVLMSyntheticGenerator()

for batch_idx, (images, paths) in enumerate(dataloader):
    # Convert images to base64 or upload to accessible server
    # Then generate VQA pairs for this batch
    dataset = generator.generate_batch(
        image_ids=[f"batch_{batch_idx}_img_{i}" for i in range(len(images))],
        image_data_list=[...],  # Base64 or URLs
        categories=["scene"] * len(images),
        difficulties=["medium"] * len(images)
    )
```

## Troubleshooting

### VLM Connection Errors
- Verify vLLM servers are running on configured ports
- Check network connectivity: `curl http://localhost:8000/v1/models`
- Ensure models are loaded: check vLLM logs

### JSON Parsing Failures
- Some VLMs may return malformed JSON
- Generator attempts to extract JSON from markdown code blocks
- Check VLM output format in logs

### Low Consistency Scores
- Increase `consistency_threshold` tolerance
- Verify secondary VLMs are running correctly
- Check image data format (URL vs base64)

### Timeout Issues
- Increase `timeout` parameter for slower networks
- Reduce `batch_size` to avoid queue bottlenecks
- Check vLLM throughput: `nvidia-smi dmon`

## Testing

See test files for comprehensive examples:
- `test_synthetic_vqa_basic.py` - Basic functionality without VLM servers
- `test_synthetic_vqa_generation.py` - Single pair generation
- `test_synthetic_vqa_batch.py` - Batch generation with mocking
- `test_synthetic_vqa_dedup.py` - Deduplication testing
- `test_synthetic_vqa_export.py` - Export format validation
