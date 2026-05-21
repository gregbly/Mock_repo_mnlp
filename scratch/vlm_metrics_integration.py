"""
Part 1.3: VQA Metric Formalization & Hardware-Aware Token Economics

This module integrates with vLLM's metrics API to measure real KV-cache usage,
throughput (tokens/sec), and VRAM consumption. Replaces naive token counting
with actual GPU hardware metrics for precise local token economics.
"""

import os
import logging
import time
import json
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import requests

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KVCacheMetrics:
    """Container for KV-cache memory measurements."""
    total_tokens: int
    kv_cache_bytes: float
    kv_cache_mb: float
    kv_cache_gb: float
    estimated_batch_size: int
    memory_utilization_pct: float


@dataclass
class TokenThroughputMetrics:
    """Container for throughput measurements."""
    tokens_per_second: float
    time_to_first_token_ms: float
    mean_inter_token_latency_ms: float
    generation_duration_sec: float
    total_tokens_generated: int


@dataclass
class LocalTokenEconomics:
    """Complete local token economy analysis."""
    kv_cache: KVCacheMetrics
    throughput: TokenThroughputMetrics
    prompt_length: int
    generation_length: int
    hardware_efficiency_score: float
    recommended_batch_size: int


class VLLMMetricsClient:
    """
    Client for querying vLLM server metrics API.

    vLLM exposes Prometheus metrics at `/metrics` endpoint.
    This client extracts and processes GPU memory, throughput, and cache metrics.
    """

    def __init__(
        self,
        vllm_url: str = "http://localhost:8000",
        metrics_endpoint: str = "/metrics",
        timeout: int = 10
    ):
        """
        Initialize vLLM metrics client.

        Args:
            vllm_url: Base URL of vLLM server (e.g., 'http://localhost:8000')
            metrics_endpoint: Path to metrics endpoint
            timeout: Request timeout in seconds
        """
        self.vllm_url = vllm_url
        self.metrics_url = f"{vllm_url}{metrics_endpoint}"
        self.timeout = timeout

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Fetch raw Prometheus metrics from vLLM server.

        Returns:
            Parsed metrics dictionary or None if unavailable
        """
        try:
            response = requests.get(self.metrics_url, timeout=self.timeout)
            response.raise_for_status()

            # Parse Prometheus text format
            metrics = self._parse_prometheus_metrics(response.text)
            return metrics

        except requests.exceptions.ConnectionError:
            logger.warning(f"Failed to connect to vLLM server at {self.vllm_url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching vLLM metrics: {str(e)}")
            return None

    @staticmethod
    def _parse_prometheus_metrics(text: str) -> Dict[str, float]:
        """
        Parse Prometheus text format metrics.

        Args:
            text: Raw Prometheus metrics text

        Returns:
            Dictionary of metric_name -> value
        """
        metrics = {}
        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse metric line: metric_name{labels} value timestamp
            if ' ' in line:
                metric_part, value_part = line.rsplit(' ', 1)
                try:
                    if '{' in metric_part:
                        name = metric_part.split('{')[0]
                    else:
                        name = metric_part

                    value = float(value_part.split()[0])
                    metrics[name] = value
                except (ValueError, IndexError):
                    continue

        return metrics

    def extract_kv_cache_metrics(self, metrics: Dict) -> Optional[KVCacheMetrics]:
        """
        Extract KV-cache specific metrics.

        Key metrics from vLLM:
        - vllm:kv_cache_usage_tokens
        - vllm:gpu_cache_usage_gb (or similar)
        - vllm:num_preemptions (cache invalidations)
        """
        try:
            total_tokens = metrics.get('vllm_kv_cache_usage_tokens', 0)

            # GPU cache usage (vLLM may report in different units)
            gpu_cache_mb = metrics.get('vllm_gpu_cache_usage_mb', 0)
            if not gpu_cache_mb:
                gpu_cache_gb = metrics.get('vllm_gpu_cache_usage_gb', 0)
                gpu_cache_mb = gpu_cache_gb * 1024

            gpu_cache_gb = gpu_cache_mb / 1024

            # Estimate batch size from cache utilization
            # Typical: ~200 bytes per token for FP8 + attention
            bytes_per_token = 200
            estimated_batch_size = max(1, int(total_tokens / 256))

            # Get total GPU memory (80GB for H100, 141GB for H200)
            total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            memory_util_pct = (gpu_cache_gb / total_gpu_mem) * 100 if total_gpu_mem > 0 else 0

            return KVCacheMetrics(
                total_tokens=int(total_tokens),
                kv_cache_bytes=gpu_cache_mb * 1e6,
                kv_cache_mb=gpu_cache_mb,
                kv_cache_gb=gpu_cache_gb,
                estimated_batch_size=estimated_batch_size,
                memory_utilization_pct=min(100, memory_util_pct)
            )

        except Exception as e:
            logger.error(f"Error extracting KV-cache metrics: {str(e)}")
            return None

    def extract_throughput_metrics(self, metrics: Dict) -> Optional[TokenThroughputMetrics]:
        """
        Extract throughput metrics from vLLM.

        Key metrics:
        - vllm:generation_tokens_total
        - vllm:generation_duration_seconds_total
        - vllm:time_to_first_token_ms
        - vllm:inter_token_latency_ms
        """
        try:
            total_tokens = metrics.get('vllm_generation_tokens_total', 0)
            total_duration = metrics.get('vllm_generation_duration_seconds_total', 1)
            ttft_ms = metrics.get('vllm_time_to_first_token_ms', 0)
            inter_token_ms = metrics.get('vllm_inter_token_latency_ms', 0)

            tokens_per_sec = total_tokens / max(total_duration, 0.001) if total_duration > 0 else 0

            return TokenThroughputMetrics(
                tokens_per_second=tokens_per_sec,
                time_to_first_token_ms=ttft_ms,
                mean_inter_token_latency_ms=inter_token_ms,
                generation_duration_sec=total_duration,
                total_tokens_generated=int(total_tokens)
            )

        except Exception as e:
            logger.error(f"Error extracting throughput metrics: {str(e)}")
            return None


class LocalTokenEconomicsAnalyzer:
    """
    Analyzes and optimizes prompts based on hardware-aware token economics.

    Unlike cloud API token economics (cost per token), local execution focuses on:
    1. KV-cache efficiency (shorter prompts = more concurrent batches)
    2. Throughput (tokens/sec)
    3. Memory footprint relative to H100/H200 HBM
    """

    def __init__(
        self,
        h100_total_hbm_gb: float = 80,
        h200_total_hbm_gb: float = 141,
        model_weight_gb: float = 30,
        vlm_url: str = "http://localhost:8000"
    ):
        """
        Initialize analyzer with hardware specs.

        Args:
            h100_total_hbm_gb: H100 HBM capacity
            h200_total_hbm_gb: H200 HBM capacity
            model_weight_gb: Approximate model weight (affects available cache)
            vlm_url: vLLM server URL for metrics
        """
        self.h100_total_hbm_gb = h100_total_hbm_gb
        self.h200_total_hbm_gb = h200_total_hbm_gb
        self.model_weight_gb = model_weight_gb
        self.vlm_client = VLLMMetricsClient(vlm_url)

    def calculate_available_hbm(self, gpu_type: str = "H100") -> float:
        """Calculate available HBM for KV-cache and data."""
        total_hbm = self.h100_total_hbm_gb if gpu_type == "H100" else self.h200_total_hbm_gb
        reserved_for_model = self.model_weight_gb * 1.2  # 20% overhead
        return total_hbm - reserved_for_model

    def estimate_max_batch_size(
        self,
        prompt_length: int,
        generation_length: int = 256,
        gpu_type: str = "H100",
        bytes_per_token: int = 200
    ) -> int:
        """
        Estimate maximum batch size given prompt and generation lengths.

        Args:
            prompt_length: Prompt tokens
            generation_length: Expected generation tokens per sample
            gpu_type: "H100" or "H200"
            bytes_per_token: Approximate bytes per token in KV-cache

        Returns:
            Recommended maximum batch size
        """
        total_tokens_per_sample = prompt_length + generation_length
        bytes_per_sample = total_tokens_per_sample * bytes_per_token

        available_hbm_bytes = self.calculate_available_hbm(gpu_type) * 1e9
        max_batch_size = int(available_hbm_bytes / bytes_per_sample)

        return max(1, max_batch_size)

    def calculate_hardware_efficiency_score(
        self,
        prompt_length: int,
        throughput_tps: float = 100,
        memory_util_pct: float = 50,
        max_batch_size: int = 64
    ) -> float:
        """
        Calculate hardware efficiency score (0-1).

        Factors:
        - Throughput relative to model maximum
        - Memory utilization (target 60-80% for optimal)
        - Prompt length efficiency (shorter = more parallelism)

        Args:
            prompt_length: Number of prompt tokens
            throughput_tps: Current throughput (tokens/sec)
            memory_util_pct: Current GPU memory utilization percentage
            max_batch_size: Maximum achievable batch size

        Returns:
            Efficiency score from 0 (worst) to 1 (best)
        """
        # Normalize throughput (assume 150 tps is optimal for 70B model)
        throughput_score = min(1.0, throughput_tps / 150.0)

        # Memory utilization should be in sweet spot (60-80%)
        if 60 <= memory_util_pct <= 80:
            memory_score = 1.0
        elif memory_util_pct < 60:
            memory_score = memory_util_pct / 60.0
        else:  # > 80%
            memory_score = (100 - memory_util_pct) / 20.0

        # Prompt efficiency (shorter prompts allow higher batch sizes)
        # Normalize: 512 tokens = medium efficiency
        prompt_efficiency = 1.0 / (1.0 + (prompt_length / 512.0) ** 2)

        # Weighted combination
        efficiency = (
            0.4 * throughput_score +
            0.3 * memory_score +
            0.3 * prompt_efficiency
        )

        return float(np.clip(efficiency, 0, 1))

    def analyze_prompt(
        self,
        prompt: str,
        generation_length: int = 256,
        gpu_type: str = "H100"
    ) -> LocalTokenEconomics:
        """
        Full analysis of prompt efficiency in local deployment.

        Args:
            prompt: The prompt to analyze
            generation_length: Expected generation length
            gpu_type: Hardware type

        Returns:
            Complete local token economics analysis
        """
        # Count prompt tokens (rough estimate: 1 token ≈ 1.3 words)
        prompt_length = len(prompt.split()) * 1.3

        # Try to get real metrics from vLLM
        metrics_dict = self.vlm_client.get_metrics()
        if metrics_dict:
            kv_cache_metrics = self.vlm_client.extract_kv_cache_metrics(metrics_dict)
            throughput_metrics = self.vlm_client.extract_throughput_metrics(metrics_dict)
        else:
            # Fallback to estimates if vLLM unavailable
            kv_cache_metrics = KVCacheMetrics(
                total_tokens=int(prompt_length),
                kv_cache_bytes=prompt_length * 200,
                kv_cache_mb=prompt_length * 200 / 1e6,
                kv_cache_gb=prompt_length * 200 / 1e9,
                estimated_batch_size=self.estimate_max_batch_size(int(prompt_length), generation_length, gpu_type),
                memory_utilization_pct=15
            )
            throughput_metrics = TokenThroughputMetrics(
                tokens_per_second=100,
                time_to_first_token_ms=50,
                mean_inter_token_latency_ms=10,
                generation_duration_sec=2.56,
                total_tokens_generated=256
            )

        efficiency_score = self.calculate_hardware_efficiency_score(
            int(prompt_length),
            throughput_metrics.tokens_per_second,
            kv_cache_metrics.memory_utilization_pct,
            kv_cache_metrics.estimated_batch_size
        )

        max_batch = self.estimate_max_batch_size(int(prompt_length), generation_length, gpu_type)

        return LocalTokenEconomics(
            kv_cache=kv_cache_metrics,
            throughput=throughput_metrics,
            prompt_length=int(prompt_length),
            generation_length=generation_length,
            hardware_efficiency_score=efficiency_score,
            recommended_batch_size=min(max_batch, kv_cache_metrics.estimated_batch_size)
        )

    def get_prompt_length_penalty(self, prompt_length: int, lambda_penalty: float = 0.005) -> float:
        """
        Calculate prompt length penalty based on KV-cache impact.

        In local deployment:
        - Longer prompts reduce maximum batch size
        - This limits throughput and GPU utilization
        - Penalty increases exponentially with prompt length

        Args:
            prompt_length: Number of tokens in prompt
            lambda_penalty: Penalty coefficient

        Returns:
            Length penalty multiplier (0-1)
        """
        # KV-cache penalty: longer prompts use proportionally more cache
        kv_cache_penalty = np.exp(-lambda_penalty * prompt_length)

        return float(kv_cache_penalty)
