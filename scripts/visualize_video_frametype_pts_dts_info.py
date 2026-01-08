"""Visualize video packet PTS/DTS and mark I-frames.

This script:
1) Uses ffprobe to extract packet info for video stream 0.
2) Plots PTS and DTS curves by packet index, marking keyframes (I-frames).
3) Optionally saves the plot as packets_pts_dts.png.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple
import matplotlib.pyplot as plt


def _fps_str_to_float(value: str | None) -> float | None:
    """Convert FPS string like '30000/1001' or '30' to float."""
    if not value:
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            num_f = float(num)
            den_f = float(den)
            if den_f == 0:
                return None
            return num_f / den_f
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def probe_packets(video_path: Path) -> List[dict]:
    """Call ffprobe to get packet-level info (v:0 stream only)."""
    cmd = [
        "ffprobe",
        "-i",
        str(video_path),
        "-select_streams",
        "v:0",
        "-show_packets",
        "-print_format",
        "json",
        "-v",
        "quiet",
    ]
    output = subprocess.check_output(cmd)
    data = json.loads(output)
    return data.get("packets", [])


def probe_fps(video_path: Path) -> Tuple[float | None, str | None]:
    """Read stream FPS (prefer avg_frame_rate, fallback to r_frame_rate). Returns float and original string."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-print_format",
        "json",
        "-i",
        str(video_path),
    ]
    output = subprocess.check_output(cmd)
    data = json.loads(output)
    streams = data.get("streams", [])
    if not streams:
        return None, None
    stream = streams[0]
    candidates = [
        stream.get("avg_frame_rate"),
        stream.get("r_frame_rate"),
    ]
    chosen_str: str | None = None
    chosen_val: float | None = None
    for cand in candidates:
        if not cand:
            continue
        val = _fps_str_to_float(cand)
        if val is None:
            continue
        chosen_str = cand
        chosen_val = val
        break
    return chosen_val, chosen_str


def probe_frame_types(video_path: Path) -> Tuple[dict, dict, int, int]:
    """Get frame pict_type using pkt_pos as index (more stable). Returns:
    - pos_to_type: pkt_pos -> pict_type
    - pos_to_frame_indices: pkt_pos -> list[frame_idx]
    - frame_total: total frame count
    - frames_missing_pos: frames without pkt_pos
    """
    cmd = [
        "ffprobe",
        "-i",
        str(video_path),
        "-select_streams",
        "v:0",
        "-show_frames",
        "-print_format",
        "json",
        "-v",
        "quiet",
    ]
    output = subprocess.check_output(cmd)
    data = json.loads(output)
    frames = data.get("frames", [])
    pos_to_type: dict = {}
    pos_to_frame_indices: dict = {}
    frame_idx = 0
    frames_missing_pos = 0
    for f in frames:
        pos = f.get("pkt_pos")
        pict_type = f.get("pict_type")
        if pos is None:
            frames_missing_pos += 1
            frame_idx += 1
            continue
        pos_str = str(pos)
        if pict_type is not None:
            pos_to_type.setdefault(pos_str, pict_type)
        pos_to_frame_indices.setdefault(pos_str, []).append(frame_idx)
        frame_idx += 1
    return pos_to_type, pos_to_frame_indices, len(frames), frames_missing_pos


def analyze_packet_frame_relation(
    packet_count: int, frame_count: int, tol_ratio: float = 0.05, tol_abs: int = 2
) -> str:
    """Judge packet/frame relationship: fragmentation, aggregation, or 1:1."""
    if frame_count <= 0:
        return "Frame count is 0, cannot judge"
    thresh = max(tol_abs, int(frame_count * tol_ratio))
    if packet_count > frame_count + thresh:
        return "Packets > Frames: likely fragmentation (multi-packet per frame)"
    if packet_count + thresh < frame_count:
        return "Packets < Frames: likely aggregation (multi-frame per packet)"
    return "Packets ≈ Frames: roughly one-to-one"


def find_fragmentation_aggregation(
    packets: Sequence[dict], pos_to_frame_indices: dict
) -> Tuple[
    List[int],
    List[int],
    List[int],
    List[int],
    int,
    int,
]:
    """Find packet indices for fragmentation/aggregation scenarios.

    - fragmentation_packets: multiple packets map to same frame (same frame_idx)
    - aggregation_packets: one packet maps to multiple frames (same pos -> multiple frame_idx)
    - packets_no_pos: packet indices without pos
    - packets_unmapped: packets with pos but no matching frame
    - frames_with_pos: number of frames having pos
    - frames_mapped: number of frames mapped via packets
    """
    fragmentation_packets: List[int] = []
    aggregation_packets: List[int] = []
    packets_no_pos: List[int] = []
    packets_unmapped: List[int] = []

    frame_to_packet: dict = {}  # frame_idx -> first packet idx
    frames_mapped_set: set[int] = set()
    frames_with_pos = sum(len(v) for v in pos_to_frame_indices.values())

    for pkt_idx, pkt in enumerate(packets):
        pos = pkt.get("pos")
        if pos is None:
            packets_no_pos.append(pkt_idx)
            continue
        pos_str = str(pos)
        frame_indices = pos_to_frame_indices.get(pos_str)
        if not frame_indices:
            packets_unmapped.append(pkt_idx)
            continue
        if len(frame_indices) > 1:
            aggregation_packets.append(pkt_idx)
            # Skip fragmentation check since this is an aggregation scenario
            continue
        frame_idx = frame_indices[0]
        frames_mapped_set.add(frame_idx)
        prev_pkt = frame_to_packet.get(frame_idx)
        if prev_pkt is None:
            frame_to_packet[frame_idx] = pkt_idx
        else:
            fragmentation_packets.append(pkt_idx)
    return (
        fragmentation_packets,
        aggregation_packets,
        packets_no_pos,
        packets_unmapped,
        frames_with_pos,
        len(frames_mapped_set),
    )


def extract_pts_dts_flags_types(
    packets: Sequence[dict], pos_to_type: dict
) -> Tuple[List[int], List[int], List[bool], List[str]]:
    """Extract pts/dts, keyframe flags and frame types (I/P/B/?) from ffprobe packet results."""
    pts_list: List[int] = []
    dts_list: List[int] = []
    is_key_list: List[bool] = []
    frame_types: List[str] = []
    for pkt in packets:
        pts_val = pkt.get("pts")
        dts_val = pkt.get("dts")
        pts_list.append(int(pts_val) if pts_val is not None else 0)
        dts_list.append(int(dts_val) if dts_val is not None else 0)
        flags = pkt.get("flags", "")
        pos = pkt.get("pos")
        pict_type = pos_to_type.get(str(pos)) if pos is not None else None
        # fallback: use flags to determine keyframe
        frame_type = pict_type or ("I" if "K" in flags else "?")
        is_key_list.append("K" in flags or frame_type == "I")  # 'K' in ffprobe flags indicates keyframe
        frame_types.append(frame_type)
    return pts_list, dts_list, is_key_list, frame_types


def plot_packets(
    pts: Sequence[int],
    dts: Sequence[int],
    is_key: Sequence[bool],
    frame_types: Sequence[str],
    fps_val: float | None = None,
    fps_str: str | None = None,
    packet_count: int | None = None,
    frame_count: int | None = None,
    frames_missing_pos: int | None = None,
    frames_with_pos: int | None = None,
    frames_mapped: int | None = None,
    packets_no_pos: int | None = None,
    packets_unmapped: int | None = None,
    relation_status: str | None = None,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot packet index vs pts/dts with keyframe markers.

    If save_path is None, don't save. If show=True, display the plot window.
    """
    indices = range(len(pts))
    fig, ax = plt.subplots(figsize=(12, 6))

    (line_pts,) = ax.plot(indices, pts, label="PTS", color="tab:blue")
    (line_dts,) = ax.plot(indices, dts, label="DTS", color="tab:orange", alpha=0.7)
    ax.set_xlabel("Packet index (from 0)")
    ax.set_ylabel("PTS / DTS (shared)")
    ax.grid(True, linestyle="--", alpha=0.3)
    title_parts = ["PTS & DTS vs packets"]
    if fps_val is not None:
        title_fps = f"{fps_val:.3f}"
        if fps_str:
            title_fps = f"{fps_str} (~{fps_val:.3f})"
        title_parts.append(f"fps={title_fps}")
    if packet_count is not None and frame_count is not None:
        title_parts.append(f"pkts={packet_count}, frames={frame_count}")
    ax.set_title(" | ".join(title_parts))

    # Mark keyframes (I-frames)
    key_idx = [i for i, k in enumerate(is_key) if k or frame_types[i] == "I"]
    key_pts = [pts[i] for i in key_idx]
    ax.scatter(key_idx, key_pts, color="red", s=20, marker="o", label="I (keyframe)")

    ax.legend(loc="upper left")

    # Display statistics
    info_lines = []
    if packet_count is not None and frame_count is not None:
        info_lines.append(f"Packets: {packet_count}")
        info_lines.append(f"Frames: {frame_count}")
    if frames_missing_pos is not None and frames_missing_pos > 0:
        info_lines.append(f"Frames without pos: {frames_missing_pos}")
    if frames_with_pos is not None and frames_mapped is not None:
        info_lines.append(f"Frames mapped: {frames_mapped}/{frames_with_pos}")
    if packets_no_pos is not None and packets_no_pos > 0:
        info_lines.append(f"Packets without pos: {packets_no_pos}")
    if packets_unmapped is not None and packets_unmapped > 0:
        info_lines.append(f"Packets unmapped: {packets_unmapped}")
    if relation_status:
        info_lines.append(relation_status)
    if info_lines:
        ax.text(
            1.02,
            0.98,
            "\n".join(info_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.9, lw=0.5),
            clip_on=False,
        )

    # Show values on hover
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    annot.set_visible(False)
    vline = ax.axvline(0, color="gray", linestyle="--", alpha=0.3, visible=False)

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            annot.set_visible(False)
            vline.set_visible(False)
            fig.canvas.draw_idle()
            return
        idx = int(round(event.xdata))
        if idx < 0 or idx >= len(pts):
            annot.set_visible(False)
            vline.set_visible(False)
            fig.canvas.draw_idle()
            return
        vline.set_xdata((idx, idx))  # sequence length must match ydata
        vline.set_visible(True)
        annot.xy = (idx, pts[idx])
        annot.set_text(f"idx={idx}\nPTS={pts[idx]}\nDTS={dts[idx]}\nType={frame_types[idx]}\nI-frame={is_key[idx]}")
        annot.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"Saved plot to: {save_path}")
    if show:
        plt.show()


def main() -> None:
    video_path = "test_videos/ea09afcd-425a-499e-86c3-a88d0b1d70c8.mov"

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    fps_val, fps_str = probe_fps(video_path)
    (
        pos_to_type,
        pos_to_frame_indices,
        frame_count,
        frames_missing_pos,
    ) = probe_frame_types(video_path)
    packets = probe_packets(video_path)
    if not packets:
        raise RuntimeError("No packet data retrieved")

    pts, dts, is_key, frame_types = extract_pts_dts_flags_types(packets, pos_to_type)
    packet_count = len(packets)
    relation_status = analyze_packet_frame_relation(packet_count, frame_count)
    (
        frag_packets,
        aggr_packets,
        packets_no_pos,
        packets_unmapped,
        frames_with_pos,
        frames_mapped,
    ) = find_fragmentation_aggregation(packets, pos_to_frame_indices)
    print(
        f"Packets: {packet_count}, Frames: {frame_count}, Relation: {relation_status}",
    )
    print(f"Fragmentation packets (multi-packet per frame): count={len(frag_packets)}, indices={frag_packets[:30]}")
    print(f"Aggregation packets (multi-frame per packet): count={len(aggr_packets)}, indices={aggr_packets[:30]}")
    print(
        f"Packets without pos: {len(packets_no_pos)}, packets unmapped: {len(packets_unmapped)}, "
        f"Frames missing pos: {frames_missing_pos}, frames with pos mapped: {frames_mapped}/{frames_with_pos}"
    )

    # Print unmapped packet details
    print("=== Unmapped packets details (first 8) ===")
    for idx in packets_unmapped[:8]:
        pkt = packets[idx]
        print(f"Packet {idx}: pos={pkt.get('pos')}, size={pkt.get('size')}, flags={pkt.get('flags')}")
    plot_packets(
        pts,
        dts,
        is_key,
        frame_types,
        fps_val=fps_val,
        fps_str=fps_str,
        packet_count=packet_count,
        frame_count=frame_count,
        frames_missing_pos=frames_missing_pos,
        frames_with_pos=frames_with_pos,
        frames_mapped=frames_mapped,
        packets_no_pos=len(packets_no_pos),
        packets_unmapped=len(packets_unmapped),
        relation_status=relation_status,
        save_path=None,
        show=True,
    )


if __name__ == "__main__":
    main()
