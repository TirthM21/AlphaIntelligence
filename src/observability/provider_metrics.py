"""Provider-level telemetry for data-source reliability and fallback visibility."""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Optional


@dataclass
class ProviderStats:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0

    def record(self, success: bool, latency_ms: float) -> None:
        self.attempts += 1
        self.total_latency_ms += latency_ms
        if success:
            self.successes += 1
        else:
            self.failures += 1

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.attempts if self.attempts else 0.0


class ProviderMetrics:
    """Tracks attempts/success/failure/latency per provider."""

    def __init__(self) -> None:
        self._stats: Dict[str, ProviderStats] = {}

    def _get(self, provider: str) -> ProviderStats:
        if provider not in self._stats:
            self._stats[provider] = ProviderStats()
        return self._stats[provider]

    def timed_call(self, provider: str, success: bool, started_at: float) -> None:
        latency_ms = (perf_counter() - started_at) * 1000.0
        self._get(provider).record(success=success, latency_ms=latency_ms)

    def summary(self) -> Dict[str, Dict[str, float]]:
        payload: Dict[str, Dict[str, float]] = {}
        for provider, st in self._stats.items():
            payload[provider] = {
                "attempts": st.attempts,
                "successes": st.successes,
                "failures": st.failures,
                "avg_latency_ms": round(st.avg_latency_ms, 2),
                "success_rate": round((st.successes / st.attempts) * 100.0, 2) if st.attempts else 0.0,
            }
        return payload
