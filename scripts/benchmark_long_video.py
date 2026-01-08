"""
Test decode cost estimation for a specific long video.
"""

import numpy as np
import time
import os
from video_reader import PyVideoReader

vid_path = "test_videos/vid_long1.mp4"
name = os.path.basename(vid_path)[:25]

try:
    vr = PyVideoReader(vid_path)
    gt_list = list(PyVideoReader(vid_path))
    frame_count = len(gt_list)
    print(f"Video: {name}")
    print(f"Frame count: {frame_count}")
    print()
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)

patterns = []
# Sparse patterns
for n in [10, 20, 30, 50, 100]:
    indices = np.linspace(0, frame_count - 1, n, dtype=int)
    indices = sorted(list(set(indices)))
    patterns.append((f"Sparse {n}", indices))

# Random patterns
rng = np.random.default_rng(seed=42)
for n in [10, 20, 30, 50, 100]:
    if frame_count >= n:
        indices = sorted(rng.choice(frame_count, size=n, replace=False).tolist())
        patterns.append((f"Random {n}", indices))

# Every K patterns
for k in [10, 30, 50, 100, 200]:
    indices = list(range(0, frame_count, k))
    if indices:
        patterns.append((f"Every {k}", indices))

print(f"{'Pattern':<15} {'N':>5} {'Seeks':>6} {'Rec':<5} {'Act':<5} {'OK':<3} {'SEQ':>8} {'SEEK':>8} {'Pen':>8}")
print("=" * 80)

for pattern_name, indices in patterns:
    if not indices:
        continue

    result = vr.estimate_decode_cost_detailed(indices)
    recommend = "SEQ" if result["recommendation"] == 1 else "SEEK"

    # Benchmark
    seq_times = []
    seek_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        vr_seq = PyVideoReader(vid_path)
        batch_seq = vr_seq.get_batch(indices, with_fallback=True)
        seq_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        vr_seek = PyVideoReader(vid_path)
        batch_seek = vr_seek.get_batch(indices, with_fallback=False)
        seek_times.append(time.perf_counter() - t0)

    seq_time = np.median(seq_times)
    seek_time = np.median(seek_times)

    actual_faster = "SEQ" if seq_time < seek_time else "SEEK"
    is_correct = recommend == actual_faster

    if not is_correct:
        penalty = abs(seek_time - seq_time)
    else:
        penalty = 0

    mark = "✓" if is_correct else "✗"
    print(
        f"{pattern_name:<15} {len(indices):>5} {result['seek_count']:>6} {recommend:<5} {actual_faster:<5} {mark:<3} {seq_time * 1000:>7.0f}ms {seek_time * 1000:>7.0f}ms {penalty * 1000:>7.0f}ms"
    )
