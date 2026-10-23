from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_playground_renders_and_validates_empty_input_without_loading_model():
    """Verify the playground renders and rejects empty input without model loading."""
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "main.py")).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "Laya action playground"
    button = next(button for button in app.button if button.label == "Translate and check permissions")
    button.click().run()
    assert not app.exception
    assert any("needs_clarification" in heading.value for heading in app.subheader)


def test_report_tab_renders_downloads_and_runtime_errors():
    """Verify the report tab displays errors and provides both report downloads."""
    from evaluation import evaluate

    class Unavailable:
        def load(self):
            """Simulate an unavailable model for report-tab rendering.

            Raises:
                RuntimeError: Always raised to create visible evaluation error rows.
            """
            raise RuntimeError("No test model")

        def metadata(self):
            """Identify the offline UI test backend.

            Returns:
                A model metadata dictionary labelling this instance as a UI test double.
            """
            return {"model": "unavailable (UI test)"}

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "main.py"))
    app.session_state.report = evaluate(Unavailable(), limit=3)
    app.run(timeout=20)
    assert not app.exception
    assert any("No test model" in item.value for item in app.error)
    assert len(app.get("download_button")) == 2
