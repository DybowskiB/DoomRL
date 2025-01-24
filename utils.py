import cv2
import numpy as np
from collections import deque


# Preprocessing klatki
def preprocess_frame(frame):
    if frame.shape[0] == 3:  # Klatka w formacie RGB
        frame = frame.transpose(1, 2, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Konwersja na czarno-biały
        resized = cv2.resize(gray, (80, 80))  # Skalowanie do 80x80
        return resized  # Shape: [80, 80]
    elif frame.shape == (80, 80):  # klatka była już przeprocesowana
        return frame
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")


# Stackowanie klatek
def stack_frames(frame, stacked_frames, stack_size):
    if stacked_frames is None:
        stacked_frames = deque(
            [np.zeros_like(frame, dtype=np.float32) for _ in range(stack_size)],
            maxlen=stack_size,
        )

    stacked_frames.append(frame)
    stacked_state = np.stack(stacked_frames, axis=0)  # Shape: [Stack Size, 80, 80]
    return stacked_state, stacked_frames
