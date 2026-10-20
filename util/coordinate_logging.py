"""Summarize received frontend coordinates without dumping every frame."""
import numpy as np


def log_received_coordinates(logger, session_id, room, count, frame, valid):
    points = np.asarray(frame).reshape(44, 2)
    shoulders = points[42:44]
    distance = float(np.linalg.norm(shoulders[0] - shoulders[1]))
    reason = "ok"
    if not np.isfinite(points).all():
        reason = "non_finite_coordinates"
    elif distance < 0.1:
        reason = "shoulders_too_close_or_missing"
    elif len(np.unique(np.round(points, 5), axis=0)) < 10:
        reason = "too_few_distinct_points"
    elif max(len(np.unique(np.round(points[:21],5),axis=0)),
             len(np.unique(np.round(points[21:42],5),axis=0))) < 5:
        reason = "hands_missing_or_collapsed"
    logger.info(
        "CC coordinates received session=%s room=%s received_frames=%s shape=%s "
        "valid=%s reason=%s left_wrist=%s right_wrist=%s shoulders=%s shoulder_distance=%.4f",
        session_id, room, count, frame.shape, valid, reason,
        np.round(points[0],4).tolist(), np.round(points[21],4).tolist(),
        np.round(shoulders,4).tolist(), distance,
    )
