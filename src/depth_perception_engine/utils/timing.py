"""A tiny stopwatch for populating DepthPerceptionResult.processing_time_ms."""

import time


class Stopwatch:
    """Context-manager style timer, in milliseconds."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0
