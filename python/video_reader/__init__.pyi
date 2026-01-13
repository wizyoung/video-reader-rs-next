"""Type stubs for video_reader module."""

from typing import Iterator, Optional, Union, overload

import numpy as np
from numpy.typing import NDArray

class PyVideoReader:
    """
    A video reader for efficient frame-by-frame or batch decoding.

    Parameters
    ----------
    filename : str
        Path to the video file.
    threads : int, optional
        Number of threads to use. If None, FFmpeg chooses the optimal number.
    resize_shorter_side : float, optional
        Resize the shorter side of the video to this value, preserving aspect ratio.
    resize_longer_side : float, optional
        Resize the longer side of the video to this value, preserving aspect ratio.
    target_width : int, optional
        Resize to exact width. Must be used with target_height.
    target_height : int, optional
        Resize to exact height. Must be used with target_width.
    resize_algo : str, optional
        Resize algorithm: 'fast_bilinear' (default), 'bilinear', 'bicubic',
        'nearest', 'area', 'lanczos'.
    device : str, optional
        Hardware acceleration device: 'cuda', 'vdpau', 'drm', etc.
        None or 'cpu' uses CPU decoding.
    filter : str, optional
        Custom FFmpeg filter, e.g. "format=rgb24,scale=w=256:h=256:flags=fast_bilinear".
    log_level : str, optional
        FFmpeg log level: 'quiet', 'panic', 'fatal', 'error' (default), 'warning',
        'info', 'verbose', 'debug', 'trace'.
    oob_mode : str, optional
        Out-of-bounds handling: 'error' (default), 'skip', or 'black'.
    """

    def __init__(
        self,
        filename: str,
        threads: Optional[int] = None,
        resize_shorter_side: Optional[float] = None,
        resize_longer_side: Optional[float] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        resize_algo: Optional[str] = None,
        device: Optional[str] = None,
        filter: Optional[str] = None,
        log_level: Optional[str] = None,
        oob_mode: Optional[str] = None,
    ) -> None: ...
    def __iter__(self) -> Iterator[NDArray[np.uint8]]: ...
    def __next__(self) -> NDArray[np.uint8]: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int) -> NDArray[np.uint8]:
        """Get a single frame by index. Returns array of shape (H, W, 3)."""
        ...

    @overload
    def __getitem__(self, key: slice) -> NDArray[np.uint8]:
        """Get frames by slice. Returns array of shape (N, H, W, 3)."""
        ...

    @overload
    def __getitem__(self, key: list[int]) -> NDArray[np.uint8]:
        """Get frames by list of indices. Returns array of shape (N, H, W, 3)."""
        ...

    def __getitem__(self, key: Union[int, slice, list[int]]) -> NDArray[np.uint8]: ...
    def get_pts(self, index: Optional[Union[int, slice, list[int]]] = None) -> list[float]:
        """
        Get the PTS (Presentation Time Stamp) for frame(s).

        Parameters
        ----------
        index : int, slice, list[int], optional
            Frame index or indices. If None, returns all PTS values.

        Returns
        -------
        list[float]
            PTS values in seconds. -1 for out-of-bounds indices.
        """
        ...

    def get_info(self) -> dict[str, str]:
        """
        Get video metadata information.

        Returns
        -------
        dict[str, str]
            Dictionary containing video metadata (all values are strings).
            Keys include: 'width', 'height', 'frame_count', 'time_base', etc.
        """
        ...

    def get_fps(self) -> float:
        """
        Get the average frame rate of the video.

        Returns
        -------
        float
            Average frames per second.
        """
        ...

    def get_shape(self) -> list[int]:
        """
        Get the shape of the video.

        Returns
        -------
        list[int]
            [frame_count, height, width]
        """
        ...

    def decode(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        compression_factor: Optional[float] = None,
    ) -> NDArray[np.uint8]:
        """
        Decode the video to a numpy array.

        Parameters
        ----------
        start_frame : int, optional
            Starting frame index.
        end_frame : int, optional
            Ending frame index (inclusive).
        compression_factor : float, optional
            Temporal compression. E.g., 0.25 decodes 1 frame out of 4.
            Default is 1.0 (all frames).

        Returns
        -------
        NDArray[np.uint8]
            Video frames with shape (N, H, W, 3).
        """
        ...

    def decode_fast(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        compression_factor: Optional[float] = None,
    ) -> list[NDArray[np.uint8]]:
        """
        Decode using YUV420P format with async YUV-to-RGB conversion.

        Faster than `decode()` for high-resolution videos.

        Parameters
        ----------
        start_frame : int, optional
            Starting frame index.
        end_frame : int, optional
            Ending frame index (inclusive).
        compression_factor : float, optional
            Temporal compression factor.

        Returns
        -------
        list[NDArray[np.uint8]]
            List of frames, each with shape (H, W, 3).
        """
        ...

    def decode_gray(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        compression_factor: Optional[float] = None,
    ) -> NDArray[np.uint8]:
        """
        Decode the video to grayscale.

        Parameters
        ----------
        start_frame : int, optional
            Starting frame index.
        end_frame : int, optional
            Ending frame index (inclusive).
        compression_factor : float, optional
            Temporal compression factor.

        Returns
        -------
        NDArray[np.uint8]
            Grayscale frames with shape (N, H, W).
        """
        ...

    def get_batch(
        self,
        indices: list[int],
        with_fallback: Optional[bool] = None,
    ) -> NDArray[np.uint8]:
        """
        Decode specific frames by index.

        Parameters
        ----------
        indices : list[int]
            List of frame indices to decode.
        with_fallback : bool, optional
            - None (default): automatically choose the faster method.
            - True: use sequential decoding (iterate through all frames).
            - False: use seek-based decoding (seek to keyframes).

        Returns
        -------
        NDArray[np.uint8]
            Frames with shape (N, H, W, 3), where N = len(indices).
        """
        ...

    def estimate_decode_cost(self, indices: list[int]) -> tuple[int, int]:
        """
        Estimate decode cost for both methods.

        Parameters
        ----------
        indices : list[int]
            Frame indices to decode.

        Returns
        -------
        tuple[int, int]
            (seek_cost, sequential_cost) - estimated number of frames to decode.
        """
        ...

    def estimate_decode_cost_detailed(self, indices: list[int]) -> dict[str, int]:
        """
        Detailed decode cost estimation.

        Parameters
        ----------
        indices : list[int]
            Frame indices to decode.

        Returns
        -------
        dict[str, int]
            Dictionary with keys: 'seek_frames', 'seek_count', 'sequential_frames',
            'unique_count', 'max_index', 'recommendation', 'seek_total_cost'.
        """
        ...

    def count_actual_frames(self) -> int:
        """
        Count actual decodable frames by decoding without color conversion.

        This is slower than reading metadata but gives accurate results
        for B-frame videos. Equivalent to ffprobe's `-count_frames` option.

        Returns
        -------
        int
            Number of decodable frames.
        """
        ...
