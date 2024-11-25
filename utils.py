import cv2
import numpy as np
from collections import deque


def preprocess_frame(frame):
    """Preprocesses a raw frame."""
    if frame.shape[0] == 3:  # RGB frame
        frame = frame.transpose(1, 2, 0)  # Convert to HWC
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        resized = cv2.resize(gray, (80, 80))  # Resize to 80x80
        return resized  # Shape: [80, 80]
    elif frame.shape == (80, 80):  # Already preprocessed
        return frame
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")


def stack_frames(frame, stacked_frames, stack_size):
    """Stacks frames along the channel dimension."""
    if stacked_frames is None:
        stacked_frames = deque([np.zeros_like(frame, dtype=np.float32) for _ in range(stack_size)], maxlen=stack_size)

    stacked_frames.append(frame)
    stacked_state = np.stack(stacked_frames, axis=0)  # Shape: [Stack Size, 80, 80]
    return stacked_state, stacked_frames

