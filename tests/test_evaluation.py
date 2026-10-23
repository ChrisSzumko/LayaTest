import json
from collections import Counter

from domain import ACTIONS
from evaluation import evaluate, report_html, report_json
from scenarios import scenarios


class UnavailableBackend:
    def load(self):
        """Simulate a model-loading failure with HTML-like text in its message.

        Raises:
            RuntimeError: Always raised to exercise report failure handling and escaping.
        """
        raise RuntimeError("model not available <script>")

    def metadata(self):
        """Return identifying metadata for the unavailable test backend.

        Returns:
            A dictionary identifying this offline test double, not a real model.
        """
        return {"model": "unavailable (test only)"}


def test_fixtures_cover_every_action_with_three_phrasings():
    """Verify complete action coverage and unique scenario identifiers."""
    counts = Counter(case.action for case in scenarios() if "/" in case.name)
    assert set(counts) == set(ACTIONS)
    assert all(value >= 3 for value in counts.values())
    assert len({case.name for case in scenarios()}) == len(scenarios())


def test_model_failure_is_reported_without_fabricated_predictions():
    """Verify load failures become error rows and escaped, serializable reports."""
    report = evaluate(UnavailableBackend())
    assert report["summary"]["cases"] == 79
    assert report["summary"]["errors"] == 79
    assert report["summary"]["passed"] == 0
    assert report["summary"]["action_accuracy"] == 0
    assert report["summary"]["warm_latency_ms"]["mean"] is None
    assert all(row["actual"]["status"] == "error" for row in report["results"])
    assert json.loads(report_json(report))["load_error"].startswith("RuntimeError")
    html = report_html(report)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_limited_smoke_report_preserves_expected_labels():
    """Verify limiting a run preserves fixture labels and independent policy checks."""
    report = evaluate(UnavailableBackend(), limit=3)
    assert report["summary"]["cases"] == 3
    assert report["results"][0]["expected"]["action"] == "view_alarm_rules"
    assert report["summary"]["authorization_accuracy_oracle_targets"] == 1
