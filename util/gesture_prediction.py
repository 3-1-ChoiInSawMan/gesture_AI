"""Filter gesture predictions before they can become translated words."""

from collections import Counter, deque

from util.config import CC_CONF_THRESHOLD, CC_NONE_LABELS


def is_none_label(label: str | None) -> bool:
    return isinstance(label, str) and label.strip().casefold() in CC_NONE_LABELS


def select_word_candidates(
    predictions: list[tuple[int, float]],
    idx2label: dict[int, str],
    pred_history: deque[int],
) -> tuple[list[str] | None, dict]:
    if not predictions:
        pred_history.clear()
        return None, {"accepted": False, "reason": "empty_predictions", "voted": None}

    prediction, confidence = predictions[0]
    candidate = idx2label.get(prediction)
    candidate_words = [
        word
        for index, _ in predictions
        if (word := idx2label.get(index)) and not is_none_label(word)
    ]
    debug = {
        "candidate": candidate,
        "candidates": candidate_words,
        "confidence": round(confidence, 4),
        "accepted": False,
        "voted": None,
    }

    # Reject the current prediction before voting: an old gesture must never
    # survive a none/uncertain prediction through the smoothing history.
    if is_none_label(candidate):
        reason = "none_label"
    elif not candidate:
        reason = "unknown_label"
    elif not confidence >= CC_CONF_THRESHOLD:
        reason = "low_confidence"
    else:
        reason = None

    if reason:
        pred_history.clear()
        debug["reason"] = reason
        return None, debug

    pred_history.append(prediction)
    voted = Counter(pred_history).most_common(1)[0][0]
    voted_word = idx2label.get(voted)
    if not voted_word or is_none_label(voted_word):
        pred_history.clear()
        debug["reason"] = "invalid_vote"
        return None, debug

    debug.update(accepted=True, voted=voted_word, reason="gesture")
    return [voted_word] + [word for word in candidate_words if word != voted_word], debug
