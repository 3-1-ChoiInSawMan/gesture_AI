"""Convert raw frontend coordinates to the training coordinate system."""
import numpy as np


def prepare_sequence(sequence, *, sequence_normalization=True):
    x = np.asarray(sequence, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 88 or len(x) == 0:
        raise ValueError("Expected a nonempty (T, 88) sequence")
    if not np.isfinite(x).all():
        raise ValueError("Coordinates must be finite")
    points = x.reshape(-1, 44, 2)
    shoulders = points[:, 42:44]
    center = shoulders.mean(axis=1, keepdims=True)
    distance = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    if np.any(distance < 1e-6):
        raise ValueError("Shoulder distance is too small")
    # Apply to all points, including zero-filled missing hands, as collect.py does.
    x = ((points - center) / distance[:, None, None]).reshape(-1, 88)
    if sequence_normalization:
        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
    return x.astype(np.float32)
