"""ZigZag K-bar labeling algorithm implementation.

This module exposes a single public function `find_zigzag_labels(df)` that
implements the exact algorithm specified in the prompt. It operates on a
Pandas DataFrame with OHLCV columns and returns a NumPy array of integer
labels in {0, 1, 2, 3} for each bar.

Label meaning:
- 0: HH (High to High)
- 1: HL (High to Low)
- 2: LH (Low to High)
- 3: LL (Low to Low)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


THRESHOLD: float = 0.02
LOOKBACK_WINDOW: int = 20
VALID_LABELS = (0, 1, 2, 3)


def _validate_input_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input DataFrame satisfies all required constraints.

    This function raises ValueError with an informative message if any
    requirement is violated.
    """

    required_cols = ["open", "high", "low", "close", "volume"]

    # Check columns exist and are lowercase
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected {required_cols}.")

    # Ensure column names are lowercase (prompt requirement)
    for c in required_cols:
        if c.lower() != c:
            raise ValueError("All required column names must be lowercase: open, high, low, close, volume.")

    # Check for NaN / infinite values on required columns
    subset = df[required_cols]
    if subset.isna().values.any():
        raise ValueError("Input DataFrame contains NaN values in OHLCV columns.")

    if not np.isfinite(subset.to_numpy(dtype="float64")).all():
        raise ValueError("Input DataFrame contains non-finite values in OHLCV columns.")

    high = subset["high"].to_numpy(dtype="float64")
    low = subset["low"].to_numpy(dtype="float64")
    open_ = subset["open"].to_numpy(dtype="float64")
    close = subset["close"].to_numpy(dtype="float64")

    # high >= low for all bars
    if not np.all(high >= low):
        raise ValueError("All bars must satisfy high >= low.")

    # open and close within [low, high]
    if not np.all((open_ >= low) & (open_ <= high)):
        raise ValueError("All bars must satisfy open in [low, high].")
    if not np.all((close >= low) & (close <= high)):
        raise ValueError("All bars must satisfy close in [low, high].")


def _safe_random_int(low: int, high: int, size: int | None = None) -> np.ndarray:
    """Generate random integers using NumPy's Generator API.

    This is used only for the fallback labeling cases described in the spec
    (when local structure is not decisive, or for padding at the end).
    """

    rng = np.random.default_rng()
    return rng.integers(low=low, high=high + 1, size=size, endpoint=True)


def find_zigzag_labels(df: pd.DataFrame) -> np.ndarray:
    """Compute ZigZag labels for each bar in an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns [open, high, low, close, volume].

    Returns
    -------
    np.ndarray
        Array of shape (len(df),) with integer labels in {0, 1, 2, 3}.
    """

    _validate_input_dataframe(df)

    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    n = len(df)

    # Edge-case: empty input -> empty output
    if n == 0:
        return np.array([], dtype="int64")

    labels: list[int] = []
    i = 0

    while i < n - 1:
        current_high = high[i]
        current_low = low[i]
        next_high = current_high
        next_low = current_low
        found_reversal = False

        # 2.2: scanning loop
        upper_k = min(i + LOOKBACK_WINDOW, n - 1)
        for k in range(i + 1, upper_k + 1):
            # 2.2.1: update tracked high/low
            if high[k] > next_high:
                next_high = high[k]
            if low[k] < next_low:
                next_low = low[k]

            # 2.2.2: compute change_up and change_down (with zero protection)
            if current_low > 0:
                change_up = (next_high - current_low) / current_low
            else:
                change_up = 0.0

            if current_high > 0:
                change_down = (current_high - next_low) / current_high
            else:
                change_down = 0.0

            # 2.2.3: check upward reversal
            if change_up > THRESHOLD:
                found_reversal = True
                if next_low < current_low:
                    labels.append(2)  # LH
                else:
                    labels.append(0)  # HH
                i = k
                break

            # 2.2.4: check downward reversal
            if change_down > THRESHOLD:
                found_reversal = True
                if next_high > current_high:
                    labels.append(0)  # HH
                else:
                    labels.append(3)  # LL
                i = k
                break

        # 2.3: post-scan handling
        if not found_reversal:
            if i + 2 < n:
                h1 = high[i]
                h2 = high[i + 1]
                h3 = high[i + 2]
                l1 = low[i]
                l2 = low[i + 1]
                l3 = low[i + 2]

                # middle bar is local high
                if (h2 >= h1) and (h2 >= h3):
                    if (l2 >= l1) and (l2 >= l3):
                        labels.append(0)  # HH
                    else:
                        labels.append(1)  # HL
                # middle bar is local low
                elif (l2 <= l1) and (l2 <= l3):
                    if (h2 <= h1) and (h2 <= h3):
                        labels.append(3)  # LL
                    else:
                        labels.append(2)  # LH
                else:
                    # Undecided local structure: random label in [0, 3]
                    labels.append(int(_safe_random_int(0, 3, size=1)[0]))

                i += 1
            else:
                # Near the end: no local 3-bar structure check
                i += 1

    # Step 3: pad remaining labels if necessary (safety net)
    while len(labels) < n:
        labels.append(int(_safe_random_int(0, 3, size=1)[0]))

    result = np.asarray(labels[:n], dtype="int64")

    # Output validation: ensure correctness of returned labels
    if result.shape[0] != n:
        raise RuntimeError("Internal error: label array length does not match number of bars.")
    if not np.isin(result, VALID_LABELS).all():
        raise RuntimeError("Internal error: label array contains values outside [0, 3].")
    if not np.isfinite(result).all():
        raise RuntimeError("Internal error: label array contains non-finite values.")

    return result


def _build_test_df_from_ohl(low_list: Iterable[float], high_list: Iterable[float]) -> pd.DataFrame:
    """Utility function for quick manual tests.

    It builds a minimal OHLCV frame with synthetic open/close/volume.
    """

    low_arr = np.asarray(low_list, dtype="float64")
    high_arr = np.asarray(high_list, dtype="float64")

    if low_arr.shape != high_arr.shape:
        raise ValueError("low_list and high_list must have the same length.")

    # Simple mid-price open/close; volume all ones.
    mid = (low_arr + high_arr) / 2.0

    return pd.DataFrame(
        {
            "open": mid,
            "high": high_arr,
            "low": low_arr,
            "close": mid,
            "volume": np.ones_like(mid),
        }
    )


def _run_basic_examples() -> None:
    """Run the three example scenarios described in the specification.

    This function is only for local, manual verification and exploratory tests.
    """

    # Test 1: clear upward reversal
    highs_1 = [100.0, 101.0, 105.0, 108.0]
    lows_1 = [95.0, 96.0, 100.0, 102.0]
    df1 = _build_test_df_from_ohl(lows_1, highs_1)
    labels_1 = find_zigzag_labels(df1)
    print("Test 1 labels:", labels_1)

    # Test 2: clear downward reversal
    highs_2 = [100.0, 99.0, 98.0, 97.0]
    lows_2 = [95.0, 94.0, 90.0, 88.0]
    df2 = _build_test_df_from_ohl(lows_2, highs_2)
    labels_2 = find_zigzag_labels(df2)
    print("Test 2 labels:", labels_2)

    # Test 3: small fluctuations, no clear reversal
    highs_3 = [100.0, 100.5, 100.2]
    lows_3 = [95.0, 95.5, 95.3]
    df3 = _build_test_df_from_ohl(lows_3, highs_3)
    labels_3 = find_zigzag_labels(df3)
    print("Test 3 labels:", labels_3)


if __name__ == "__main__":
    _run_basic_examples()
