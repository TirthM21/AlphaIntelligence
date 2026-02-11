from src.reporting.newsletter_generator import NewsletterGenerator


def test_dedupe_news_removes_duplicates_preserves_order():
    items = [
        {'title': 'A', 'url': 'u1', 'site': 'S'},
        {'title': 'A', 'url': 'u1', 'site': 'S2'},
        {'title': 'B', 'url': 'u2', 'site': 'S'},
    ]
    deduped = NewsletterGenerator._dedupe_news(items, max_items=10)
    assert len(deduped) == 2
    assert deduped[0]['title'] == 'A'
    assert deduped[1]['title'] == 'B'


def test_normalize_news_item_supports_headline_and_link():
    raw = {'headline': 'Fed Update', 'link': 'http://x', 'publisher': 'Yahoo'}
    normalized = NewsletterGenerator._normalize_news_item(raw, title_key='headline', url_key='link', source_key='publisher')
    assert normalized['title'] == 'Fed Update'
    assert normalized['url'] == 'http://x'
    assert normalized['site'] == 'Yahoo'
