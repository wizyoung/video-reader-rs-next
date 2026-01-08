#!/usr/bin/env python3
"""
Comprehensive benchmark script comparing video_reader-rs, decord, and opencv.

Benchmark scenarios:
1. Full traversal - iterate through all frames
2. Dense get_batch - clustered/consecutive frame indices
3. Sparse get_batch - randomly scattered frame indices
4. Various batch sizes
5. Decode + Resize to 224x224

Metrics:
- Time (initialization, decode, total)
- Peak memory usage (measured via subprocess isolation for accuracy)
"""

import argparse
import gc
import json
import os
import sys
import time
import threading
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Any, Tuple, Dict

import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Memory measurements will be disabled.")
    print("Install with: pip install psutil")


# ==============================================================================
# Peak Memory Measurement via RSS Sampling (for subprocess)
# ==============================================================================
class PeakMemoryMonitor:
    """Monitor peak RSS memory usage during function execution."""

    def __init__(self, sample_interval: float = 0.001):
        """
        Args:
            sample_interval: Time between memory samples in seconds (default 1ms)
        """
        self.sample_interval = sample_interval
        self.peak_mem = 0.0
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process() if HAS_PSUTIL else None

    def _get_memory_mb(self) -> float:
        """Get current process RSS memory in MB."""
        if self._process:
            return self._process.memory_info().rss / 1024 / 1024
        return 0.0

    def _monitor_loop(self):
        """Background thread that samples memory."""
        while not self._stop_flag:
            current = self._get_memory_mb()
            if current > self.peak_mem:
                self.peak_mem = current
            time.sleep(self.sample_interval)

    def start(self):
        """Start memory monitoring."""
        gc.collect()
        self.peak_mem = self._get_memory_mb()
        self._stop_flag = False
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        """Stop monitoring and return peak memory in MB (absolute, not delta)."""
        self._stop_flag = True
        if self._thread:
            self._thread.join(timeout=0.1)
        return self.peak_mem


def measure_with_peak_memory(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Execute a function while measuring peak memory usage.

    Returns:
        Tuple of (function_result, peak_memory_mb) - absolute peak RSS
    """
    monitor = PeakMemoryMonitor()
    monitor.start()
    try:
        result = func(*args, **kwargs)
    finally:
        peak = monitor.stop()
    return result, peak


# ==============================================================================
# Video paths from scripts/decoding_accuracy_test.py
# ==============================================================================
DEFAULT_VIDEOS = [
    "test_videos/1_negpts_A.mp4",
    "test_videos/2_negpts_B.mp4",
    "test_videos/3_negpts_C.mp4",
    "test_videos/4_negdts_D.mp4",
    "test_videos/5_negdts_E.mp4",
    "test_videos/6_negdts_F.mp4",
    "test_videos/7_negpts_negdts_G.mp4",
    "test_videos/8_negpts_negdts_H.mp4",
    "test_videos/9_negpts_negdts_I.mp4",
    "test_videos/0a7ef2bd-4852-45c6-be96-645972ab2905.mp4",
    "test_videos/ea09afcd-425a-499e-86c3-a88d0b1d70c8.mov",
    "test_videos/v10033g50000d1mr95vog65ifnkmh8o0.mp4",
    "test_videos/v10033g50000d3f056vog65u26hllhtg.mp4",
    "test_videos/out_A_negpts.mp4",
    "test_videos/out_C_negpts.mp4",
    "test_videos/v10033g50000d4g0nmfog65po93qnbj0 (1) 2.mp4",  # AV1
    "test_videos/v_3Hgwyprv8u4.mp4",
    "test_videos/22_std_clean.mp4",
    "test_videos/v_32-Bxdbf3mQ.mp4",
    "test_videos/v_72PUOTjZpQU.mp4",
    "test_videos/v_Ww2_b9f6Kh0.mp4",
    "test_videos/6159095415.mp4",
    "test_videos/v_KlmlCbJup5A.mp4",
]


@dataclass
class BenchResult:
    """Benchmark result for a single run."""

    library: str
    scenario: str
    init_time: float  # seconds
    decode_time: float  # seconds
    total_time: float  # seconds
    peak_memory_mb: float  # MB (absolute peak RSS in isolated subprocess)
    num_frames: int
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "BenchResult":
        return BenchResult(**d)


# ==============================================================================
# Subprocess benchmark runners
# ==============================================================================
def _run_benchmark_in_subprocess(
    bench_type: str,
    video_path: str,
    indices: Optional[list] = None,
    scenario: str = "",
    with_fallback: Optional[bool] = None,
) -> Dict:
    """
    Run a single benchmark in an isolated subprocess.
    This function is the target for multiprocessing.
    Returns a dict that can be converted to BenchResult.
    """
    try:
        if bench_type == "vr_full_traverse":
            return _bench_vr_full_traverse(video_path)
        elif bench_type == "vr_get_batch":
            return _bench_vr_get_batch(video_path, indices, scenario, with_fallback)
        elif bench_type == "vr_get_batch_resize":
            return _bench_vr_get_batch_resize(video_path, indices, scenario)
        elif bench_type == "vr_get_batch_resize_cv2":
            return _bench_vr_get_batch_resize_cv2(video_path, indices, scenario)
        elif bench_type == "vr_get_batch_resize_direct":
            return _bench_vr_get_batch_resize_direct(video_path, indices, scenario)
        elif bench_type == "decord_full_traverse":
            return _bench_decord_full_traverse(video_path)
        elif bench_type == "decord_get_batch":
            return _bench_decord_get_batch(video_path, indices, scenario)
        elif bench_type == "decord_get_batch_resize":
            return _bench_decord_get_batch_resize(video_path, indices, scenario)
        elif bench_type == "decord_get_batch_resize_native":
            return _bench_decord_get_batch_resize_native(video_path, indices, scenario)
        elif bench_type == "opencv_full_traverse":
            return _bench_opencv_full_traverse(video_path)
        elif bench_type == "opencv_get_batch":
            return _bench_opencv_get_batch(video_path, indices, scenario)
        elif bench_type == "opencv_get_batch_resize":
            return _bench_opencv_get_batch_resize(video_path, indices, scenario)
        else:
            return {
                "library": "unknown",
                "scenario": scenario,
                "init_time": 0,
                "decode_time": 0,
                "total_time": 0,
                "peak_memory_mb": 0,
                "num_frames": 0,
                "success": False,
                "error": f"Unknown benchmark type: {bench_type}",
            }
    except Exception as e:
        return {
            "library": bench_type,
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": str(e)[:200],
        }


def _subprocess_worker(queue, bench_type, video_path, indices, scenario, with_fallback):
    """Worker function for subprocess - must be at module level for pickle."""
    result = _run_benchmark_in_subprocess(bench_type, video_path, indices, scenario, with_fallback)
    queue.put(result)


def run_isolated_benchmark(
    bench_type: str,
    video_path: str,
    indices: Optional[list] = None,
    scenario: str = "",
    with_fallback: Optional[bool] = None,
    timeout: float = 300.0,
) -> BenchResult:
    """
    Run a benchmark in a completely isolated subprocess.
    This ensures accurate memory measurement without interference from previous benchmarks.
    """
    # Use spawn to get a clean process (especially important on macOS)
    ctx = mp.get_context("spawn")

    # Create a queue to receive results
    result_queue = ctx.Queue()

    proc = ctx.Process(
        target=_subprocess_worker, args=(result_queue, bench_type, video_path, indices, scenario, with_fallback)
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        return BenchResult(
            library=bench_type,
            scenario=scenario,
            init_time=0,
            decode_time=0,
            total_time=0,
            peak_memory_mb=0,
            num_frames=0,
            success=False,
            error="Timeout",
        )

    if result_queue.empty():
        return BenchResult(
            library=bench_type,
            scenario=scenario,
            init_time=0,
            decode_time=0,
            total_time=0,
            peak_memory_mb=0,
            num_frames=0,
            success=False,
            error="No result from subprocess",
        )

    result_dict = result_queue.get()
    return BenchResult.from_dict(result_dict)


# ==============================================================================
# video_reader-rs benchmark implementations (run inside subprocess)
# ==============================================================================
def _bench_vr_full_traverse(video_path: str) -> Dict:
    """Benchmark video_reader-rs full traversal."""
    from video_reader import PyVideoReader

    def run():
        nonlocal init_time
        t_start = time.perf_counter()
        vr = PyVideoReader(video_path)
        init_time = time.perf_counter() - t_start

        frame_count = 0
        for f in vr:
            frame_count += 1
            _ = f
        return frame_count, time.perf_counter()

    init_time = 0.0
    try:
        t_start = time.perf_counter()
        (frame_count, t_end), peak_mem = measure_with_peak_memory(run)

        return {
            "library": "video_reader",
            "scenario": "full_traverse",
            "init_time": init_time,
            "decode_time": t_end - t_start - init_time,
            "total_time": t_end - t_start,
            "peak_memory_mb": peak_mem,
            "num_frames": frame_count,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "library": "video_reader",
            "scenario": "full_traverse",
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": str(e)[:100],
        }


def _bench_vr_get_batch(video_path: str, indices: list, scenario: str, with_fallback: Optional[bool] = None) -> Dict:
    """Benchmark video_reader-rs get_batch."""
    from video_reader import PyVideoReader

    lib_name = f"video_reader(fb={with_fallback})"

    def run():
        nonlocal init_time
        t_start = time.perf_counter()
        vr = PyVideoReader(video_path)
        init_time = time.perf_counter() - t_start

        batch = vr.get_batch(indices, with_fallback=with_fallback)
        return batch, time.perf_counter()

    init_time = 0.0
    try:
        t_start = time.perf_counter()
        (batch, t_end), peak_mem = measure_with_peak_memory(run)

        return {
            "library": lib_name,
            "scenario": scenario,
            "init_time": init_time,
            "decode_time": t_end - t_start - init_time,
            "total_time": t_end - t_start,
            "peak_memory_mb": peak_mem,
            "num_frames": batch.shape[0],
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "library": lib_name,
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": str(e)[:100],
        }


def _bench_vr_get_batch_resize(video_path: str, indices: list, scenario: str) -> Dict:
    """Benchmark video_reader-rs get_batch with FFmpeg filter resize to 224x224."""
    from video_reader import PyVideoReader

    lib_name = "video_reader+resize"
    resize_filter = "format=yuv420p,scale=w=224:h=224:flags=fast_bilinear"

    def run():
        nonlocal init_time
        t_start = time.perf_counter()
        vr = PyVideoReader(video_path, filter=resize_filter)
        init_time = time.perf_counter() - t_start

        batch = vr.get_batch(indices, with_fallback=None)
        return batch, time.perf_counter()

    init_time = 0.0
    try:
        t_start = time.perf_counter()
        (batch, t_end), peak_mem = measure_with_peak_memory(run)

        return {
            "library": lib_name,
            "scenario": scenario,
            "init_time": init_time,
            "decode_time": t_end - t_start - init_time,
            "total_time": t_end - t_start,
            "peak_memory_mb": peak_mem,
            "num_frames": batch.shape[0],
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "library": lib_name,
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": str(e)[:100],
        }


def _bench_vr_get_batch_resize_cv2(video_path: str, indices: list, scenario: str) -> Dict:
    """Benchmark video_reader-rs get_batch + cv2.resize to 224x224 (no FFmpeg filter)."""
    from video_reader import PyVideoReader
    import cv2

    lib_name = "video_reader+cv2"

    def run():
        nonlocal init_time
        t_start = time.perf_counter()
        vr = PyVideoReader(video_path)
        init_time = time.perf_counter() - t_start

        batch = vr.get_batch(indices, with_fallback=None)
        # Resize using cv2 instead of FFmpeg filter
        resized = np.stack([cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR) for frame in batch])
        return resized, time.perf_counter()

    init_time = 0.0
    try:
        t_start = time.perf_counter()
        (batch, t_end), peak_mem = measure_with_peak_memory(run)

        return {
            "library": lib_name,
            "scenario": scenario,
            "init_time": init_time,
            "decode_time": t_end - t_start - init_time,
            "total_time": t_end - t_start,
            "peak_memory_mb": peak_mem,
            "num_frames": batch.shape[0],
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "library": lib_name,
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": str(e)[:100],
        }


def _bench_vr_get_batch_resize_direct(video_path: str, indices: list, scenario: str) -> Dict:
    """Benchmark video_reader-rs get_batch with target_width/height resize to 224x224."""
    from video_reader import PyVideoReader

    lib_name = "video_reader+direct_size"

    def run():
        nonlocal init_time
        t_start = time.perf_counter()
        # Use target_width/height for fixed output dimensions
        vr = PyVideoReader(video_path, target_width=224, target_height=224, resize_algo="fast_bilinear")
        init_time = time.perf_counter() - t_start

        batch = vr.get_batch(indices, with_fallback=None)
        return batch, time.perf_counter()

    init_time = 0.0
    try:
        t_start = time.perf_counter()
        (batch, t_end), peak_mem = measure_with_peak_memory(run)

        return {
            "library": lib_name,
            "scenario": scenario,
            "init_time": init_time,
            "decode_time": t_end - t_start - init_time,
            "total_time": t_end - t_start,
            "peak_memory_mb": peak_mem,
            "num_frames": batch.shape[0],
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "library": lib_name,
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": str(e)[:100],
        }


# ==============================================================================
# decord benchmark implementations (run inside subprocess)
# ==============================================================================
def _bench_decord_full_traverse(video_path: str) -> Dict:
    """Benchmark decord full traversal."""
    try:
        from decord import VideoReader
    except ImportError:
        return {
            "library": "decord",
            "scenario": "full_traverse",
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": "decord not installed",
        }

    def run():
        nonlocal init_time
        t_start = time.perf_counter()
        vr = VideoReader(video_path, num_threads=0)
        init_time = time.perf_counter() - t_start

        frame_count = len(vr)
        for i in range(frame_count):
            _ = vr[i].asnumpy()
        return frame_count, time.perf_counter()

    init_time = 0.0
    try:
        t_start = time.perf_counter()
        (frame_count, t_end), peak_mem = measure_with_peak_memory(run)

        return {
            "library": "decord",
            "scenario": "full_traverse",
            "init_time": init_time,
            "decode_time": t_end - t_start - init_time,
            "total_time": t_end - t_start,
            "peak_memory_mb": peak_mem,
            "num_frames": frame_count,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "library": "decord",
            "scenario": "full_traverse",
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": str(e)[:100],
        }


def _bench_decord_get_batch(video_path: str, indices: list, scenario: str) -> Dict:
    """Benchmark decord get_batch."""
    try:
        from decord import VideoReader
    except ImportError:
        return {
            "library": "decord",
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": "decord not installed",
        }

    def run():
        nonlocal init_time
        t_start = time.perf_counter()
        vr = VideoReader(video_path, num_threads=0)
        init_time = time.perf_counter() - t_start

        batch = vr.get_batch(indices).asnumpy()
        return batch, time.perf_counter()

    init_time = 0.0
    try:
        t_start = time.perf_counter()
        (batch, t_end), peak_mem = measure_with_peak_memory(run)

        return {
            "library": "decord",
            "scenario": scenario,
            "init_time": init_time,
            "decode_time": t_end - t_start - init_time,
            "total_time": t_end - t_start,
            "peak_memory_mb": peak_mem,
            "num_frames": batch.shape[0],
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "library": "decord",
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": str(e)[:100],
        }


def _bench_decord_get_batch_resize(video_path: str, indices: list, scenario: str) -> Dict:
    """Benchmark decord get_batch + cv2.resize to 224x224."""
    try:
        from decord import VideoReader
        import cv2
    except ImportError as e:
        return {
            "library": "decord+cv2",
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": f"import error: {e}",
        }

    def run():
        nonlocal init_time
        t_start = time.perf_counter()
        vr = VideoReader(video_path, num_threads=0)
        init_time = time.perf_counter() - t_start

        batch = vr.get_batch(indices).asnumpy()
        # Resize each frame to 224x224 using cv2
        resized = np.stack([cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR) for frame in batch])
        return resized, time.perf_counter()

    init_time = 0.0
    try:
        t_start = time.perf_counter()
        (batch, t_end), peak_mem = measure_with_peak_memory(run)

        return {
            "library": "decord+cv2",
            "scenario": scenario,
            "init_time": init_time,
            "decode_time": t_end - t_start - init_time,
            "total_time": t_end - t_start,
            "peak_memory_mb": peak_mem,
            "num_frames": batch.shape[0],
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "library": "decord+cv2",
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": str(e)[:100],
        }


def _bench_decord_get_batch_resize_native(video_path: str, indices: list, scenario: str) -> Dict:
    """Benchmark decord get_batch with native resize to 224x224."""
    try:
        from decord import VideoReader
    except ImportError as e:
        return {
            "library": "decord+native",
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": f"import error: {e}",
        }

    def run():
        nonlocal init_time
        t_start = time.perf_counter()
        # Use decord's native resize via width/height parameters
        vr = VideoReader(video_path, width=224, height=224, num_threads=0)
        init_time = time.perf_counter() - t_start

        batch = vr.get_batch(indices).asnumpy()
        return batch, time.perf_counter()

    init_time = 0.0
    try:
        t_start = time.perf_counter()
        (batch, t_end), peak_mem = measure_with_peak_memory(run)

        return {
            "library": "decord+native",
            "scenario": scenario,
            "init_time": init_time,
            "decode_time": t_end - t_start - init_time,
            "total_time": t_end - t_start,
            "peak_memory_mb": peak_mem,
            "num_frames": batch.shape[0],
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "library": "decord+native",
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": str(e)[:100],
        }


# ==============================================================================
# OpenCV benchmark implementations (run inside subprocess)
# ==============================================================================
def _bench_opencv_full_traverse(video_path: str) -> Dict:
    """Benchmark opencv full traversal."""
    try:
        import cv2
    except ImportError:
        return {
            "library": "opencv",
            "scenario": "full_traverse",
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": "opencv not installed",
        }

    def run():
        nonlocal init_time
        t_start = time.perf_counter()
        cap = cv2.VideoCapture(video_path)
        init_time = time.perf_counter() - t_start

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            _ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        return frame_count, time.perf_counter()

    init_time = 0.0
    try:
        t_start = time.perf_counter()
        (frame_count, t_end), peak_mem = measure_with_peak_memory(run)

        return {
            "library": "opencv",
            "scenario": "full_traverse",
            "init_time": init_time,
            "decode_time": t_end - t_start - init_time,
            "total_time": t_end - t_start,
            "peak_memory_mb": peak_mem,
            "num_frames": frame_count,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "library": "opencv",
            "scenario": "full_traverse",
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": str(e)[:100],
        }


def _bench_opencv_get_batch(video_path: str, indices: list, scenario: str) -> Dict:
    """Benchmark opencv get frames by seeking."""
    try:
        import cv2
    except ImportError:
        return {
            "library": "opencv",
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": "opencv not installed",
        }

    def run():
        nonlocal init_time
        t_start = time.perf_counter()
        cap = cv2.VideoCapture(video_path)
        init_time = time.perf_counter() - t_start

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                frames.append(None)
        cap.release()
        valid_frames = [f for f in frames if f is not None]
        return valid_frames, time.perf_counter()

    init_time = 0.0
    try:
        t_start = time.perf_counter()
        (valid_frames, t_end), peak_mem = measure_with_peak_memory(run)

        return {
            "library": "opencv",
            "scenario": scenario,
            "init_time": init_time,
            "decode_time": t_end - t_start - init_time,
            "total_time": t_end - t_start,
            "peak_memory_mb": peak_mem,
            "num_frames": len(valid_frames),
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "library": "opencv",
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": str(e)[:100],
        }


def _bench_opencv_get_batch_resize(video_path: str, indices: list, scenario: str) -> Dict:
    """Benchmark opencv get frames by seeking + cv2.resize to 224x224."""
    try:
        import cv2
    except ImportError:
        return {
            "library": "opencv+resize",
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": "opencv not installed",
        }

    def run():
        nonlocal init_time
        t_start = time.perf_counter()
        cap = cv2.VideoCapture(video_path)
        init_time = time.perf_counter() - t_start

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                frames.append(resized)
            else:
                frames.append(None)
        cap.release()
        valid_frames = [f for f in frames if f is not None]
        return valid_frames, time.perf_counter()

    init_time = 0.0
    try:
        t_start = time.perf_counter()
        (valid_frames, t_end), peak_mem = measure_with_peak_memory(run)

        return {
            "library": "opencv+resize",
            "scenario": scenario,
            "init_time": init_time,
            "decode_time": t_end - t_start - init_time,
            "total_time": t_end - t_start,
            "peak_memory_mb": peak_mem,
            "num_frames": len(valid_frames),
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "library": "opencv+resize",
            "scenario": scenario,
            "init_time": 0,
            "decode_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
            "num_frames": 0,
            "success": False,
            "error": str(e)[:100],
        }


# ==============================================================================
# Index generation utilities
# ==============================================================================
def generate_dense_indices(num_frames: int, batch_size: int = 30) -> list:
    """Generate dense/consecutive frame indices (e.g., 0-29)."""
    if num_frames <= batch_size:
        return list(range(num_frames))
    # Start from a random position
    start = num_frames // 4
    return list(range(start, min(start + batch_size, num_frames)))


def generate_sparse_indices(num_frames: int, batch_size: int = 30, seed: int = 42) -> list:
    """Generate sparse/scattered frame indices across the video."""
    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(num_frames, size=min(batch_size, num_frames), replace=False)
    return sorted(indices.tolist())


def generate_very_sparse_indices(num_frames: int, batch_size: int = 10, seed: int = 42) -> list:
    """Generate very sparse indices (e.g., every Nth frame)."""
    step = max(1, num_frames // batch_size)
    return list(range(0, num_frames, step))[:batch_size]


# ==============================================================================
# Main benchmark logic
# ==============================================================================
def get_video_frame_count(video_path: str) -> int:
    """Get frame count using video_reader."""
    from video_reader import PyVideoReader

    vr = PyVideoReader(video_path)
    return len(vr)


def benchmark_video(video_path: str, verbose: bool = False) -> list:
    """Run all benchmarks on a single video using subprocess isolation."""
    results = []
    name = os.path.basename(video_path)[:35]

    if not os.path.exists(video_path):
        print(f"{name}: SKIP (not found)")
        return results

    try:
        num_frames = get_video_frame_count(video_path)
    except Exception as e:
        print(f"{name}: ERROR getting frame count ({e})")
        return results

    if num_frames < 20:
        print(f"{name}: SKIP (only {num_frames} frames)")
        return results

    print(f"\n{'=' * 80}")
    print(f"Video: {name} ({num_frames} frames)")
    print(f"{'=' * 80}")

    # Generate test indices
    dense_indices = generate_dense_indices(num_frames, batch_size=30)
    sparse_indices = generate_sparse_indices(num_frames, batch_size=30)
    very_sparse_indices = generate_very_sparse_indices(num_frames, batch_size=10)

    # 1. Full traversal benchmarks
    print("\n--- Full Traversal ---")
    for bench_type, lib_name in [
        ("vr_full_traverse", "video_reader"),
        ("decord_full_traverse", "decord"),
        ("opencv_full_traverse", "opencv"),
    ]:
        result = run_isolated_benchmark(bench_type, video_path)
        results.append(result)
        if result.success:
            print(
                f"  {lib_name:28s}: {result.total_time:6.3f}s | peak:{result.peak_memory_mb:6.1f}MB | {result.num_frames} frames"
            )
        else:
            print(f"  {lib_name:28s}: ✗ ({result.error})")

    # 2. Dense get_batch benchmarks
    print(f"\n--- Dense get_batch ({len(dense_indices)} frames) ---")
    for bench_type, lib_name, indices in [
        ("vr_get_batch", "video_reader", dense_indices),
        ("decord_get_batch", "decord", dense_indices),
        ("opencv_get_batch", "opencv", dense_indices),
    ]:
        result = run_isolated_benchmark(bench_type, video_path, indices=indices, scenario="dense_batch")
        results.append(result)
        if result.success:
            print(f"  {lib_name:28s}: {result.total_time:6.3f}s | peak:{result.peak_memory_mb:6.1f}MB")
        else:
            print(f"  {lib_name:28s}: ✗ ({result.error})")

    # 3. Sparse get_batch benchmarks
    print(f"\n--- Sparse get_batch ({len(sparse_indices)} frames) ---")
    for bench_type, lib_name, indices in [
        ("vr_get_batch", "video_reader", sparse_indices),
        ("decord_get_batch", "decord", sparse_indices),
        ("opencv_get_batch", "opencv", sparse_indices),
    ]:
        result = run_isolated_benchmark(bench_type, video_path, indices=indices, scenario="sparse_batch")
        results.append(result)
        if result.success:
            print(f"  {lib_name:28s}: {result.total_time:6.3f}s | peak:{result.peak_memory_mb:6.1f}MB")
        else:
            print(f"  {lib_name:28s}: ✗ ({result.error})")

    # 4. Very sparse get_batch benchmarks
    print(f"\n--- Very Sparse get_batch ({len(very_sparse_indices)} frames) ---")
    for bench_type, lib_name, indices in [
        ("vr_get_batch", "video_reader", very_sparse_indices),
        ("decord_get_batch", "decord", very_sparse_indices),
        ("opencv_get_batch", "opencv", very_sparse_indices),
    ]:
        result = run_isolated_benchmark(bench_type, video_path, indices=indices, scenario="very_sparse_batch")
        results.append(result)
        if result.success:
            print(f"  {lib_name:28s}: {result.total_time:6.3f}s | peak:{result.peak_memory_mb:6.1f}MB")
        else:
            print(f"  {lib_name:28s}: ✗ ({result.error})")

    # 5. Decode + Resize benchmarks (sparse indices)
    print(f"\n--- Decode + Resize 224x224 ({len(sparse_indices)} frames, sparse) ---")

    result = run_isolated_benchmark(
        "decord_get_batch_resize", video_path, indices=sparse_indices, scenario="sparse_resize"
    )
    results.append(result)
    if result.success:
        print(f"  {'decord+cv2':28s}: {result.total_time:6.3f}s | peak:{result.peak_memory_mb:6.1f}MB")
    else:
        print(f"  {'decord+cv2':28s}: ✗ ({result.error})")

    result = run_isolated_benchmark(
        "decord_get_batch_resize_native", video_path, indices=sparse_indices, scenario="sparse_resize"
    )
    results.append(result)
    if result.success:
        print(f"  {'decord+native':28s}: {result.total_time:6.3f}s | peak:{result.peak_memory_mb:6.1f}MB")
    else:
        print(f"  {'decord+native':28s}: ✗ ({result.error})")

    result = run_isolated_benchmark(
        "opencv_get_batch_resize", video_path, indices=sparse_indices, scenario="sparse_resize"
    )
    results.append(result)
    if result.success:
        print(f"  {'opencv+resize':28s}: {result.total_time:6.3f}s | peak:{result.peak_memory_mb:6.1f}MB")
    else:
        print(f"  {'opencv+resize':28s}: ✗ ({result.error})")

    result = run_isolated_benchmark(
        "vr_get_batch_resize_cv2", video_path, indices=sparse_indices, scenario="sparse_resize"
    )
    results.append(result)
    if result.success:
        print(f"  {'video_reader+cv2':28s}: {result.total_time:6.3f}s | peak:{result.peak_memory_mb:6.1f}MB")
    else:
        print(f"  {'video_reader+cv2':28s}: ✗ ({result.error})")

    result = run_isolated_benchmark(
        "vr_get_batch_resize_direct", video_path, indices=sparse_indices, scenario="sparse_resize"
    )
    results.append(result)
    if result.success:
        print(f"  {'video_reader+direct_size':28s}: {result.total_time:6.3f}s | peak:{result.peak_memory_mb:6.1f}MB")
    else:
        print(f"  {'video_reader+direct_size':28s}: ✗ ({result.error})")

    return results


def print_summary(all_results: list):
    """Print aggregate summary across all videos."""
    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY")
    print("=" * 80)

    # Group by library and scenario
    from collections import defaultdict

    grouped = defaultdict(list)

    for r in all_results:
        if r.success:
            grouped[(r.library, r.scenario)].append(r)

    # Print summary table
    print(f"\n{'Library':<25s} {'Scenario':<20s} {'Avg Time':>10s} {'Avg Peak Mem':>12s} {'Count':>6s}")
    print("-" * 80)

    for (lib, scenario), results in sorted(grouped.items()):
        avg_time = np.mean([r.total_time for r in results])
        avg_mem = np.mean([r.peak_memory_mb for r in results])
        print(f"{lib:<25s} {scenario:<20s} {avg_time:>9.3f}s {avg_mem:>11.1f}MB {len(results):>6d}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark video_reader-rs vs decord vs opencv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all default videos from scripts/decoding_accuracy_test.py
  python benchmark.py

  # Run on a single video
  python benchmark.py --video /path/to/video.mp4

  # Run with verbose output
  python benchmark.py --verbose
        """,
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default=None,
        help="Path to a specific video file. If not provided, uses videos from scripts/decoding_accuracy_test.py",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-opencv", action="store_true", help="Skip opencv benchmarks")
    parser.add_argument("--no-decord", action="store_true", help="Skip decord benchmarks")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.video:
        videos = [args.video]
    else:
        videos = DEFAULT_VIDEOS

    print("=" * 80)
    print("Benchmark: video_reader-rs vs decord vs opencv")
    print("Peak memory measured via subprocess isolation (accurate RSS)")
    print("=" * 80)

    all_results = []
    for video_path in videos:
        results = benchmark_video(video_path, verbose=args.verbose)
        all_results.extend(results)

    if all_results:
        print_summary(all_results)
    else:
        print("\nNo benchmarks completed successfully.")


if __name__ == "__main__":
    main()
