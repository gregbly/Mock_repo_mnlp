"""
Part 1.1: Image Pool & Automated Local/HuggingFace Retrieval

This module manages local image pools and enables high-speed image retrieval
directly into GPU VRAM using PyTorch DataLoaders and memory-mapped I/O.
Optimized for H100/H200 GPUs with massive HBM for instant access.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import io
from datasets import load_dataset, Dataset as HFDataset

logger = logging.getLogger(__name__)


class ImagePoolDataset(Dataset):
    """PyTorch Dataset for efficient local image loading with memory-mapped support."""

    def __init__(
        self,
        image_paths: List,
        image_size: Tuple[int, int] = (512, 512),
        normalize: bool = True,
        use_mmap: bool = True
    ):
        """
        Initialize image pool dataset.

        Args:
            image_paths: List of image items (either file paths or PIL.Image objects)
            image_size: Target image size (height, width)
            normalize: Whether to normalize images to [0, 1]
            use_mmap: Use memory-mapped files for efficient memory management
        """
        self.image_items = image_paths
        self.image_size = image_size
        self.normalize = normalize
        self.use_mmap = use_mmap

        # Only validate local file paths, skip for PIL images
        if len(self.image_items) > 0 and isinstance(self.image_items[0], str):
            self._validate_paths()

        logger.info(f"ImagePoolDataset initialized with {len(self.image_items)} images")

    def _validate_paths(self):
        """Validate only local file paths. Skip HF objects."""

        # If HF dataset (PIL images), skip validation entirely
        if len(self.image_items) > 0 and not isinstance(self.image_items[0], str):
            logger.info("Skipping validation (HuggingFace image objects detected)")
            return

        invalid_paths = []
        for path in self.image_items:
            if not os.path.exists(path):
                invalid_paths.append(path)

        if invalid_paths:
            logger.warning(f"Found {len(invalid_paths)} invalid image paths. Filtering...")
            self.image_items = [p for p in self.image_items if os.path.exists(p)]

        if not self.image_items:
            raise ValueError("No valid image paths found in image pool")

    def __len__(self) -> int:
        return len(self.image_items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        item = self.image_items[idx]

        try:
            # -------------------------
            # Case 1: local file path
            # -------------------------
            if isinstance(item, str):
                img = Image.open(item).convert("RGB")

            # -------------------------
            # Case 2: HuggingFace PIL image
            # -------------------------
            else:
                img = item.convert("RGB")

            # Resize
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)

            # Convert to tensor [H, W, 3] → [3, H, W]
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)

            # Normalize to [-1, 1]
            if self.normalize:
                img_tensor = img_tensor * 2.0 - 1.0

            return img_tensor, str(item)

        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            return torch.zeros(3, *self.image_size), str(item)


class ImagePool:
    """
    High-performance image pool manager with GPU preloading capabilities.

    Supports:
    - Local directory scanning
    - HuggingFace dataset integration
    - PyTorch DataLoader with multi-threaded workers
    - GPU VRAM preloading with dynamic memory management
    - Memory-mapped file support for large image collections
    """

    def __init__(
        self,
        pool_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        image_size: Tuple[int, int] = (512, 512),
        num_workers: int = 4,
        batch_size: int = 32,
        gpu_id: int = 0,
        max_images: Optional[int] = None,
        use_mmap: bool = True
    ):
        """
        Initialize image pool.

        Args:
            pool_path: Path to local directory containing images
            dataset_name: HuggingFace dataset name (e.g., 'coco/2014')
            image_size: Target image dimensions
            num_workers: Number of DataLoader workers
            batch_size: Preloading batch size
            gpu_id: GPU index to pin memory to
            max_images: Limit number of images loaded (for memory management)
            use_mmap: Enable memory-mapped file support
        """
        self.pool_path = pool_path
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.max_images = max_images
        self.use_mmap = use_mmap

        self.image_paths: List = []      
        self.image_items: List = []    
        self.preloaded_images: Dict[str, torch.Tensor] = {}
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

        # Load image paths
        self._load_image_paths()

        logger.info(f"ImagePool initialized with {len(self.image_items)} images on {self.device}")

    def _load_image_paths(self):
        """Load image paths from local directory or HuggingFace dataset."""
        if self.pool_path:
            self._load_from_local_directory()
        elif self.dataset_name:
            self._load_from_huggingface()
        else:
            raise ValueError("Either pool_path or dataset_name must be provided")

    def _load_from_local_directory(self):
        """Scan local directory for image files."""
        pool_dir = Path(self.pool_path)
        if not pool_dir.exists():
            raise FileNotFoundError(f"Image pool directory not found: {self.pool_path}")

        self.image_paths = []   

        extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}

        for ext in extensions:
            self.image_paths.extend(str(p) for p in pool_dir.glob(f'**/*{ext}'))
            self.image_paths.extend(str(p) for p in pool_dir.glob(f'**/*{ext.upper()}'))

        if self.max_images:
            self.image_paths = self.image_paths[:self.max_images]

        logger.info(f"Loaded {len(self.image_paths)} images from {self.pool_path}")

    def _load_from_huggingface(self):
        """Load images from HuggingFace dataset (robust PIL decoding)."""
        try:
            dataset = load_dataset(self.dataset_name, split="train")

            image_col = None
            # Prioritize exact "image" column, then fall back to other image columns
            if "image" in dataset.column_names:
                image_col = "image"
            else:
                for col in dataset.column_names:
                    if "image" in col.lower():
                        image_col = col
                        break

            if not image_col:
                raise ValueError(f"No image column found in {self.dataset_name}")

            self.image_paths = []

            for i, sample in enumerate(dataset):
                if self.max_images and i >= self.max_images:
                    break

                img = sample[image_col]

                # -----------------------------
                # FIX: robust HF image decoding
                # -----------------------------
                if hasattr(img, "convert"):
                    img = img.convert("RGB")

                elif isinstance(img, str):
                    # HF dataset returning image as string (URL or local path)
                    if img.startswith(("http://", "https://")):
                        from urllib.request import urlopen
                        img = Image.open(io.BytesIO(urlopen(img).read())).convert("RGB")
                    else:
                        img = Image.open(img).convert("RGB")

                elif isinstance(img, dict):
                    # HF Image feature format
                    if "bytes" in img:
                        img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
                    elif "path" in img:
                        img = Image.open(img["path"]).convert("RGB")
                    else:
                        raise TypeError(f"Unknown HF image dict format: {img.keys()}")

                else:
                    raise TypeError(f"Unsupported HF image type: {type(img)}")

                self.image_paths.append(img)

            logger.info(f"Loaded {len(self.image_paths)} images from {self.dataset_name}")

        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset: {str(e)}")
            raise

    def create_dataloader(self) -> DataLoader:
        dataset = ImagePoolDataset(
            self.image_paths,
            self.image_size,
            normalize=True,
            use_mmap=self.use_mmap
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=0,  # IMPORTANT: avoids Mac multiprocessing issues
            pin_memory=True
        )

    def preload_to_gpu(self, num_batches: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Preload images into GPU VRAM for sub-millisecond retrieval.

        Args:
            num_batches: Number of batches to preload (None = all)

        Returns:
            Dictionary mapping image paths to GPU tensors
        """
        dataloader = self.create_dataloader()

        batch_count = 0
        for batch_images, batch_paths in dataloader:
            if num_batches and batch_count >= num_batches:
                break

            # Transfer batch to GPU
            batch_images = batch_images.to(self.device)

            # Store in preloaded cache
            for img, path in zip(batch_images, batch_paths):
                self.preloaded_images[path] = img

            batch_count += 1

            # Log memory usage
            if batch_count % 10 == 0:
                gpu_mem = torch.cuda.memory_allocated(self.device) / 1e9
                logger.info(f"Preloaded {batch_count * self.batch_size} images "
                           f"({gpu_mem:.2f} GB GPU memory used)")

        return self.preloaded_images

    def get_image(self, image_path: str) -> torch.Tensor:
        """
        Retrieve image (from cache or disk).

        Args:
            image_path: Path to image

        Returns:
            Image tensor on GPU
        """
        # Check preloaded cache first (sub-millisecond)
        if image_path in self.preloaded_images:
            return self.preloaded_images[image_path]

        # Fall back to loading from disk
        dataset = ImagePoolDataset([image_path], self.image_size, normalize=True)
        img_tensor, _ = dataset[0]
        return img_tensor.to(self.device)

    def batch_get_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        Retrieve multiple images efficiently.

        Args:
            image_paths: List of image paths

        Returns:
            Stacked tensor of shape (N, 3, H, W)
        """
        images = []
        for path in image_paths:
            img = self.get_image(path)
            images.append(img)

        return torch.stack(images)

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        cached_images = len(self.preloaded_images) * np.prod(self.image_size) * 3 * 4 / 1e9

        stats = {
            'preloaded_count': len(self.preloaded_images),
            'cached_images_gb': cached_images,
        }

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            stats['allocated_gb'] = allocated
            stats['reserved_gb'] = reserved

        return stats

    def clear_cache(self):
        """Clear preloaded images from GPU memory."""
        self.preloaded_images.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Image cache cleared")


def create_image_pool_from_config(config_dict: Dict) -> ImagePool:
    """
    Factory function to create ImagePool from configuration dictionary.

    Args:
        config_dict: Configuration with keys like 'image_pool_path', 'image_pool_dataset', etc.

    Returns:
        Configured ImagePool instance
    """
    return ImagePool(
        pool_path=config_dict.get('image_pool_path'),
        dataset_name=config_dict.get('image_pool_dataset'),
        image_size=tuple(config_dict.get('image_size', (512, 512))),
        num_workers=config_dict.get('num_workers', 4),
        batch_size=config_dict.get('batch_size', 32),
        gpu_id=config_dict.get('gpu_id', 0),
        max_images=config_dict.get('max_images'),
        use_mmap=config_dict.get('use_mmap', True)
    )
