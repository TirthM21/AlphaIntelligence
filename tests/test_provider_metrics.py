from time import perf_counter

from src.observability.provider_metrics import ProviderMetrics


def test_provider_metrics_summary_tracks_counts():
    pm = ProviderMetrics()
    t1 = perf_counter()
    pm.timed_call("fmp", True, t1)
    t2 = perf_counter()
    pm.timed_call("fmp", False, t2)

    summary = pm.summary()
    assert "fmp" in summary
    assert summary["fmp"]["attempts"] == 2
    assert summary["fmp"]["successes"] == 1
    assert summary["fmp"]["failures"] == 1
