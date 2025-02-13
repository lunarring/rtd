# class to interpolate one or more images between trwo frames

import numpy as np
from PIL import Image


class AverageFrameInterpolator:
    def __init__(self, num_frames: int = 1):
        """
        Initialize the interpolator with the number of frames to interpolate.

        Args:
            num_frames: The number of frames to interpolate between the previous and new frame.
        """
        self.num_frames = num_frames
        self.prev_frame = None

    def interpolate(self, new_frame: Image.Image | np.ndarray) -> list[np.ndarray]:
        """
        Interpolate between the previous frame and new frame.

        Args:
            new_frame: The new frame to interpolate towards

        Returns:
            List of interpolated frames between previous and new frame as uin8 arrays.
        """
        # cast and copy
        new_frame = np.asarray(new_frame).astype(np.float32).copy()

        # If this is the first frame, just return copies of the target frame
        if self.prev_frame is None:
            self.prev_frame = new_frame
            return [new_frame.copy() for _ in range(self.num_frames)]

        interpolated = []

        # Generate num_frames interpolated frames
        for i in range(1, self.num_frames + 1):
            # Calculate interpolation factor (0 to 1)
            factor = i / (self.num_frames + 1)

            # Linear interpolation between frames
            frame_interp = (
                (new_frame * factor + self.prev_frame * (1 - factor))
                .clip(0, 255)
                .astype(np.uint8)
            )

            interpolated.append(frame_interp)

        # Store the new frame for next interpolation
        self.prev_frame = new_frame

        return interpolated
