"""Exponential moving average filter for temporal smoothing."""

from __future__ import annotations


class ExponentialMovingAverage:
    """Simple exponential moving average (EMA) filter.

    Smooths a scalar time series to reduce frame-to-frame jitter:

        smoothed_t = alpha * value_t + (1 - alpha) * smoothed_{t-1}

    On the first update, the smoothed value equals the raw input.
    """

    def __init__(self, alpha: float) -> None:
        """Initialize the EMA filter.

        Args:
            alpha: Smoothing factor in ``(0.0, 1.0]``. Higher values react
                faster to new measurements; lower values produce smoother output.

        Raises:
            ValueError: If ``alpha`` is outside ``(0.0, 1.0]``.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0.0, 1.0], got {alpha}")
        self._alpha = alpha
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        """Current smoothed value, or ``None`` before the first update."""
        return self._value

    def update(self, measurement: float) -> float:
        """Incorporate a new measurement and return the smoothed value.

        Args:
            measurement: New raw scalar observation.

        Returns:
            Updated smoothed value.
        """
        if self._value is None:
            self._value = measurement
        else:
            self._value = (
                self._alpha * measurement + (1.0 - self._alpha) * self._value
            )
        return self._value

    def reset(self) -> None:
        """Clear filter state."""
        self._value = None
