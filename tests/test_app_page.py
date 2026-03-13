from streamlit.testing.v1 import AppTest


def test_app_loads_without_error():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert not at.exception


def test_app_has_title():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    # Check title element exists
    assert len(at.title) > 0


def test_app_has_four_metrics():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert len(at.metric) == 4
