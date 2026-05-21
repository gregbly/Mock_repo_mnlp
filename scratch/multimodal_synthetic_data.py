"""
Part 1.2: Multimodal Synthetic Data Generation

This module generates high-fidelity Q&A pairs for VQA tasks using local VLMs.
Leverages vLLM's continuous batching and paged attention for parallel generation.
Includes self-consistency checking across multiple VLM models for quality assurance.
"""

import os
import logging
import json
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import time

import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class VQAPair:
    """Represents a single Visual Question Answering pair."""
    image_id: str
    question: str
    answer: str
    confidence: float
    visual_category: str
    difficulty: str  # "easy", "medium", "hard"


@dataclass
class SyntheticVQADataset:
    """Container for generated VQA dataset."""
    pairs: List[VQAPair]
    generation_time_sec: float
    total_tokens_generated: int
    quality_filter_applied: bool
    average_confidence: float


class LocalVLMSyntheticGenerator:
    """
    Generates multimodal synthetic data using local Vision Language Models.

    Supports:
    - Batch parallel generation via vLLM continuous batching
    - Multiple VLM models for self-consistency checking
    - Domain-specific prompt templates
    - Quality filtering and deduplication
    """

    # Default VLM endpoints on H100 cluster
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

    def __init__(
        self,
        vllm_base_url: str = "http://localhost",
        primary_vlm: str = "qwen2-vl-7b",
        use_self_consistency: bool = True,
        consistency_threshold: float = 0.8,
        batch_size: int = 32,
        timeout: int = 60
    ):
        """
        Initialize synthetic data generator.

        Args:
            vllm_base_url: Base URL for vLLM servers (e.g., http://localhost)
            primary_vlm: Primary VLM model to use
            use_self_consistency: Enable cross-VLM consistency checking
            consistency_threshold: Minimum agreement score (0-1) for inclusion
            batch_size: Number of images to process in parallel
            timeout: Request timeout in seconds
        """
        self.vllm_base_url = vllm_base_url
        self.primary_vlm = primary_vlm
        self.use_self_consistency = use_self_consistency
        self.consistency_threshold = consistency_threshold
        self.batch_size = batch_size
        self.timeout = timeout

        # Validate primary VLM
        if primary_vlm not in self.VLM_CONFIGS:
            raise ValueError(f"Unsupported VLM: {primary_vlm}")

        self.primary_config = self.VLM_CONFIGS[primary_vlm]

    def _get_vlm_endpoint(self, vlm_name: str) -> str:
        """Get vLLM API endpoint for a given model."""
        if vlm_name not in self.VLM_CONFIGS:
            raise ValueError(f"Unknown VLM: {vlm_name}")

        config = self.VLM_CONFIGS[vlm_name]
        return f"{self.vllm_base_url}:{config['port']}/v1/chat/completions"

    def _create_vqa_prompt(
        self,
        visual_category: str = "general",
        difficulty: str = "medium"
    ) -> str:
        """
        Create domain-specific prompt for VQA pair generation.

        Args:
            visual_category: Category like 'scene', 'object', 'people', 'action'
            difficulty: 'easy', 'medium', or 'hard'

        Returns:
            Formatted prompt for VLM
        """
        difficulty_specs = {
            'easy': 'simple and direct, asking about obvious visual elements',
            'medium': 'requiring some reasoning about relationships and attributes',
            'hard': 'requiring deep reasoning, multi-step analysis, or common sense'
        }

        category_specs = {
            'general': 'general visual content',
            'scene': 'outdoor scenes, landscapes, or environments',
            'object': 'specific objects, their properties, and composition',
            'people': 'human subjects, expressions, actions, and interactions',
            'action': 'dynamic actions, events, or motion',
            'text': 'visible text content in images'
        }

        spec = difficulty_specs.get(difficulty, difficulty_specs['medium'])
        cat = category_specs.get(visual_category, category_specs['general'])

        prompt = f"""You are an expert visual analyst generating diverse Question-Answer pairs for Vision Language Models training.

Analyze the image and generate a high-quality visual question-answer pair that is {spec}.

The question should focus on: {cat}

Requirements:
1. Question should be specific and answerable from the image
2. Answer should be concise but complete (1-3 sentences max)
3. Avoid yes/no questions unless appropriate
4. Question should NOT reveal the answer

Output ONLY valid JSON in this format:
{{
    "question": "Your question here?",
    "answer": "Your answer here",
    "category": "{visual_category}",
    "difficulty": "{difficulty}"
}}"""

        return prompt

    def generate_vqa_pair(
        self,
        image_id: str,
        image_data: str,  # Base64 or URL
        visual_category: str = "general",
        difficulty: str = "medium",
        vlm_name: Optional[str] = None
    ) -> Optional[VQAPair]:
        """
        Generate a single VQA pair using local VLM.

        Args:
            image_id: Unique image identifier
            image_data: Base64-encoded or URL to image
            visual_category: Category of visual content
            difficulty: Question difficulty level
            vlm_name: VLM to use (None = use primary)

        Returns:
            Generated VQAPair or None if generation failed
        """
        vlm_name = vlm_name or self.primary_vlm
        endpoint = self._get_vlm_endpoint(vlm_name)
        config = self.VLM_CONFIGS[vlm_name]

        prompt = self._create_vqa_prompt(visual_category, difficulty)

        # Build request payload
        if image_data.startswith('http'):
            image_source = {"url": image_data}
        else:
            image_source = {"url": f"data:image/png;base64,{image_data}"}

        payload = {
            "model": config['model_name'],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": image_source
                        }
                    ]
                }
            ],
            "temperature": 0.7,
            "max_tokens": config['max_tokens']
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code != 200:
                logger.warning(f"VLM returned {response.status_code}: {response.text}")
                return None

            # Parse response
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()

            # Extract JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            vqa_data = json.loads(content)

            return VQAPair(
                image_id=image_id,
                question=vqa_data['question'],
                answer=vqa_data['answer'],
                confidence=1.0,  # Will be updated by consistency check
                visual_category=vqa_data.get('category', visual_category),
                difficulty=vqa_data.get('difficulty', difficulty)
            )

        except requests.exceptions.Timeout:
            logger.error(f"VLM request timeout for image {image_id}")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse VLM response for image {image_id}")
        except Exception as e:
            logger.error(f"Error generating VQA pair for {image_id}: {str(e)}")

        return None

    def check_consistency(
        self,
        vqa_pair: VQAPair,
        image_data: str
    ) -> Tuple[bool, float]:
        """
        Check if answer is consistent across multiple VLMs.

        Args:
            vqa_pair: Initial VQA pair from primary VLM
            image_data: Image data for verification

        Returns:
            (passes_consistency_check, consistency_score)
        """
        if not self.use_self_consistency:
            return True, 1.0

        consistency_scores = []

        # Check against other VLMs
        for vlm_name in self.VLM_CONFIGS:
            if vlm_name == self.primary_vlm:
                consistency_scores.append(1.0)
                continue

            try:
                # Ask the secondary VLM the same question
                endpoint = self._get_vlm_endpoint(vlm_name)
                config = self.VLM_CONFIGS[vlm_name]

                # Simple prompt asking the question
                prompt = f"""Answer this question about the image:

Question: {vqa_pair.question}

Provide only the answer in 1-2 sentences."""

                if image_data.startswith('http'):
                    image_source = {"url": image_data}
                else:
                    image_source = {"url": f"data:image/png;base64,{image_data}"}

                payload = {
                    "model": config['model_name'],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": image_source
                                }
                            ]
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 128
                }

                response = requests.post(
                    endpoint,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    secondary_answer = response.json()['choices'][0]['message']['content'].strip()

                    # Calculate semantic similarity (simple word overlap)
                    primary_words = set(vqa_pair.answer.lower().split())
                    secondary_words = set(secondary_answer.lower().split())

                    if primary_words and secondary_words:
                        overlap = len(primary_words & secondary_words) / len(primary_words | secondary_words)
                        consistency_scores.append(overlap)

            except Exception as e:
                logger.debug(f"Consistency check failed for {vlm_name}: {str(e)}")

        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.5

        passes = avg_consistency >= self.consistency_threshold

        return passes, float(avg_consistency)

    def generate_batch(
        self,
        image_ids: List[str],
        image_data_list: List[str],
        categories: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None
    ) -> SyntheticVQADataset:
        """
        Generate VQA pairs for a batch of images in parallel.

        Args:
            image_ids: List of image identifiers
            image_data_list: List of base64/URL image data
            categories: Visual categories for each image (defaults to 'general')
            difficulties: Difficulty levels for each image (defaults to 'medium')

        Returns:
            SyntheticVQADataset with generated pairs
        """
        if len(image_ids) != len(image_data_list):
            raise ValueError("image_ids and image_data_list must have same length")

        categories = categories or ['general'] * len(image_ids)
        difficulties = difficulties or ['medium'] * len(image_ids)

        generated_pairs = []
        start_time = time.time()
        total_tokens = 0

        # Generate in parallel batches
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = []

            for img_id, img_data, cat, diff in zip(image_ids, image_data_list, categories, difficulties):
                future = executor.submit(
                    self.generate_vqa_pair,
                    img_id, img_data, cat, diff
                )
                futures.append((future, img_id, img_data))

            # Collect results and apply consistency filtering
            for future, img_id, img_data in futures:
                try:
                    vqa_pair = future.result(timeout=self.timeout)

                    if vqa_pair:
                        # Check consistency if enabled
                        passes, consistency = self.check_consistency(vqa_pair, img_data)

                        if passes:
                            vqa_pair.confidence = consistency
                            generated_pairs.append(vqa_pair)
                            total_tokens += len(vqa_pair.answer.split()) + len(vqa_pair.question.split())

                except Exception as e:
                    logger.error(f"Error processing image {img_id}: {str(e)}")

        generation_time = time.time() - start_time

        avg_confidence = np.mean([p.confidence for p in generated_pairs]) if generated_pairs else 0

        return SyntheticVQADataset(
            pairs=generated_pairs,
            generation_time_sec=generation_time,
            total_tokens_generated=total_tokens,
            quality_filter_applied=self.use_self_consistency,
            average_confidence=avg_confidence
        )

    def deduplicate_pairs(
        self,
        pairs: List[VQAPair],
        similarity_threshold: float = 0.85
    ) -> List[VQAPair]:
        """
        Remove near-duplicate VQA pairs based on question similarity.

        Args:
            pairs: List of VQA pairs
            similarity_threshold: Minimum similarity to consider as duplicate

        Returns:
            Deduplicated list of VQA pairs
        """
        if not pairs:
            return []

        unique_pairs = []
        for pair in pairs:
            is_duplicate = False

            for unique_pair in unique_pairs:
                # Simple similarity: word overlap in questions
                q1_words = set(pair.question.lower().split())
                q2_words = set(unique_pair.question.lower().split())

                if q1_words and q2_words:
                    similarity = len(q1_words & q2_words) / len(q1_words | q2_words)

                    if similarity > similarity_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_pairs.append(pair)

        logger.info(f"Deduplicated {len(pairs)} pairs to {len(unique_pairs)}")
        return unique_pairs

    def export_dataset(
        self,
        dataset: SyntheticVQADataset,
        output_path: str,
        format: str = 'json'
    ) -> None:
        """
        Export generated VQA dataset to file.

        Args:
            dataset: SyntheticVQADataset to export
            output_path: Output file path
            format: 'json' or 'jsonl'
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if format == 'json':
            data = {
                'metadata': {
                    'generation_time_sec': dataset.generation_time_sec,
                    'total_tokens': dataset.total_tokens_generated,
                    'average_confidence': dataset.average_confidence,
                    'num_pairs': len(dataset.pairs),
                    'quality_filtered': dataset.quality_filter_applied
                },
                'pairs': [asdict(p) for p in dataset.pairs]
            }
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == 'jsonl':
            with open(output_path, 'w') as f:
                for pair in dataset.pairs:
                    f.write(json.dumps(asdict(pair)) + '\n')

        logger.info(f"Exported {len(dataset.pairs)} VQA pairs to {output_path}")
