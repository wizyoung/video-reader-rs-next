#!/usr/bin/env python3
"""
Strict verification: use iterator [f for f in vr] as ground truth.
Multiple rounds of random frame sampling to test:
1. get_batch(with_fallback=None) - auto mode
2. get_batch(with_fallback=True)
3. get_batch(with_fallback=False)
4. getitem (reader[i] for each i)
5. slice vr[start:end]
"""

import os
import numpy as np
from video_reader import PyVideoReader


def run_tests():
    # replace with your video paths
    videos = [
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
        "test_videos/v_A0XGYLim9IU.mp4",
    ]

    print("=" * 80)
    print("Strict Verification: Iterator as Ground Truth (5 random rounds)")
    print("=" * 80)
    print()

    total_pass = 0
    total_fail = 0
    num_rounds = 5

    for vid_path in videos:
        name = os.path.basename(vid_path)[:40]

        if not os.path.exists(vid_path):
            print(f"{name}: SKIP (not found)")
            continue

        try:
            # Get ground truth using iterator (fresh reader)
            vr_gt = PyVideoReader(vid_path)
            gt_all = [f for f in vr_gt]
            num_frames = len(gt_all)

            if num_frames < 30:
                print(f"{name}: SKIP (only {num_frames} frames)")
                continue

            print(f"{name}: running {num_rounds} rounds...")
            all_runs_ok = True

            for run in range(num_rounds):
                # Random frame indices (sorted)
                rng = np.random.default_rng(seed=run)
                batch_size = min(30, num_frames)
                batch_indices = np.sort(rng.choice(np.arange(0, num_frames), size=batch_size, replace=False)).tolist()

                # Ground truth for this batch
                gt_batch = np.array([gt_all[i] for i in batch_indices])

                results = {}

                # Test 1: get_batch with fallback=None (auto mode)
                vr_auto = PyVideoReader(vid_path)
                try:
                    batch_auto = vr_auto.get_batch(batch_indices)
                    results["auto"] = np.array_equal(batch_auto, gt_batch)
                except Exception:
                    results["auto"] = False

                # Test 2: get_batch with fallback=True
                vr_fb = PyVideoReader(vid_path)
                try:
                    batch_fb = vr_fb.get_batch(batch_indices, with_fallback=True)
                    results["fallback"] = np.array_equal(batch_fb, gt_batch)
                except Exception:
                    results["fallback"] = False

                # Test 3: get_batch with fallback=False
                vr_no_fb = PyVideoReader(vid_path)
                try:
                    batch_no_fb = vr_no_fb.get_batch(batch_indices, with_fallback=False)
                    results["no_fallback"] = np.array_equal(batch_no_fb, gt_batch)
                except Exception:
                    results["no_fallback"] = False

                # Test 4: getitem (fresh reader)
                vr_gi = PyVideoReader(vid_path)
                try:
                    getitem_frames = [vr_gi[i] for i in batch_indices]
                    batch_gi = np.array(getitem_frames)
                    results["getitem"] = np.array_equal(batch_gi, gt_batch)
                except Exception:
                    results["getitem"] = False

                # Test 5: slice vr[indices] (list indexing)
                vr_slice = PyVideoReader(vid_path)
                try:
                    batch_slice = vr_slice[batch_indices]
                    results["slice"] = np.array_equal(batch_slice, gt_batch)
                except Exception:
                    results["slice"] = False

                # Report round result
                round_ok = all(results.values())
                if round_ok:
                    print(f"  Round {run + 1}: ✓")
                else:
                    all_runs_ok = False
                    failed_tests = [k for k, v in results.items() if not v]
                    print(f"  Round {run + 1}: ✗ ({', '.join(failed_tests)})")

            if all_runs_ok:
                total_pass += 1
            else:
                total_fail += 1

        except Exception as e:
            print(f"{name}: ERROR ({str(e)[:60]})")
            total_fail += 1

    print()
    print("-" * 80)
    print(f"success: {total_pass}, fail: {total_fail}")


if __name__ == "__main__":
    run_tests()
