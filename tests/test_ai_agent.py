from src.ai.ai_agent import AIAgent


def test_generate_deep_dive_report_has_fallback_without_client():
    agent = AIAgent(api_key=None)
    agent.client = None
    report = agent.generate_deep_dive_report('sample context')
    assert 'Strategic Market Regime' in report
    assert 'endpoint unavailable' in report.lower()
