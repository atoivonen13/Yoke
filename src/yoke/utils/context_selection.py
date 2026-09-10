"""Context-window subsampling shared across the dataset, training, and eval.

The 9-band scalar-temporal forecaster selects its context by a trailing time
window: keep every event within ``context_window_days`` of the anchor
(most-recent) event, then, when more than ``max_context_len`` qualify, keep a
bounded subset. The legacy rule kept the most-recent ``max_context_len`` events
(``sel_idx[-M:]``), which — at a wide window with dense early observing nights —
dropped the early light-curve rise.

This module provides the single source of truth for the subsample so the dataset
(``_getitem_window`` / ``_getitem_window_rollout``), the batched training rollout
(``_rollout_pass_9band_window``), and the eval/inference harnesses all select
bit-identical events. The batched training path reimplements the same integer
arithmetic on tensors (see ``_rollout_pass_9band_window``); keep the two in sync.
"""

import numpy as np


def window_select_positions(
    n_qualifying: int, max_context_len: int
) -> np.ndarray:
    """Local positions of the time-sorted qualifying run to keep.

    Returns indices into ``[0, n_qualifying)`` (oldest first) selecting which of
    the in-window events to retain. When ``n_qualifying <= max_context_len`` this
    is ``arange(n_qualifying)`` — byte-identical to keeping every event, so the
    legacy behavior (and prior runs where the window never over-filled) is
    reproduced exactly. When ``n_qualifying > max_context_len`` it is an
    anchor-pinned integer linspace: ``max_context_len`` positions spread evenly by
    index, with position 0 (the earliest in-window event, i.e. the rise) and
    position ``n_qualifying - 1`` (the anchor / most-recent event) always kept.

    The selection uses only non-negative integer arithmetic (floor division), so
    it is bit-identical to the vectorized torch twin in the training rollout — no
    float rounding-mode ambiguity. It is provably strictly increasing when
    subsampling (no duplicate events).

    Args:
        n_qualifying (int): Number of events inside the trailing time window
            (>= 1; the anchor is always inside its own window).
        max_context_len (int): Maximum number of events to keep (M).

    Returns:
        np.ndarray: 1-D int64 array of local positions to keep, length
        ``min(n_qualifying, max_context_len)``, strictly increasing.
    """
    n = int(n_qualifying)
    m = int(max_context_len)

    if n <= m:
        return np.arange(n, dtype=np.int64)

    # Legacy trailing selection: keep the most-recent m events (sel_idx[-m:]).
    #
    # The anchor-pinned integer-linspace subsample below was tested in studies 077
    # (2 d/12) and 078 (5 d/24) and LOST to the legacy rule at every window: 075
    # (2 d/12, legacy) = 1.84 RMSE, 077 (2 d/12, subsample) = 2.11, 078 (5 d/24,
    # subsample) = 2.14, 072 (5 d/24, legacy) = 2.25. Late-time forecasting is
    # driven by recent, dense sampling; trading recent density to spread the
    # context (or to keep the early rise) dilutes exactly the local-slope signal
    # the model needs. So the default reverts to the trailing rule -- this makes
    # the code reproduce the 075 champion exactly. The spread subsample is
    # preserved below (commented) so re-testing it later is a one-function change;
    # the batched twin in _rollout_pass_9band_window must be flipped to match if
    # it is ever re-enabled.
    return np.arange(n - m, n, dtype=np.int64)

    # Anchor-pinned integer linspace (disabled -- see above). Round-half-up over
    # [0, n-1] with m points; consecutive numerators grow by (n - 1) >= m so each
    # floor step increases by >= 1 -> strictly increasing, endpoints pinned to 0
    # and n - 1.
    # if m == 1:
    #     return np.array([n - 1], dtype=np.int64)
    # j = np.arange(m, dtype=np.int64)
    # return (j * (n - 1) + (m - 1) // 2) // (m - 1)
