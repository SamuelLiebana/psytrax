from streamlit.testing.v1 import AppTest


def test_streamlit_app_shell_renders_without_exceptions(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    at = AppTest.from_file("app.py", default_timeout=30).run()

    assert not at.exception
    assert [title.value for title in at.title[:2]] == ["psytrax", "psytrax"]
    assert at.radio[0].label == "Navigation"
    assert at.radio[0].options == [
        "Instructions",
        "Fit Model",
        "Visualise Results",
        "Compare Models",
        "Model Recovery",
        "IBL Explorer",
    ]
