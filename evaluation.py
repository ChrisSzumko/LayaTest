"""Repeatable real-model evaluation and portable reports, also used by the UI."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
import json
from math import ceil
from pathlib import Path
from statistics import mean, median

from domain import authorize
from inference import LayaBackend
from pipeline import Pipeline
from scenarios import scenarios


def evaluate(backend, threshold=0.8, limit=None, progress=None):
    """Evaluate labelled scenarios against a backend and summarize the results.

    Load the backend and issue one unscored warmup request before timing cases.
    Loading or warmup failures produce error rows rather than fake predictions.
    Authorization is also checked independently using labelled actions/targets.

    Args:
        backend: Adapter providing load(), predict(state, questions), and metadata().
        threshold: Per-question confidence cutoff, defaulting to 0.8.
        limit: Optional prefix length of the scenario list to evaluate.
        progress: Optional callback receiving completed and total case counts
            after each evaluated case. Callback exceptions propagate.

    Returns:
        A dictionary containing run metadata, any load error, aggregate metrics,
        per-action accuracy, a confusion matrix, and expected/actual case results.
        Latency metrics exclude model loading and the unscored warmup request.
    """
    cases = scenarios()
    if limit is not None:
        cases = cases[:limit]
    pipeline = Pipeline(backend)
    load_error = None
    try:
        backend.load()
        # One unscored request warms kernels; case latency excludes loading.
        warmup = pipeline.translate("Show the system logs", threshold=threshold)
        if warmup["status"] == "error":
            load_error = warmup["reason"]
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
    rows, confusion, per_action = [], defaultdict(Counter), defaultdict(list)
    policy_matches, end_to_end_policy = [], []
    for index, case in enumerate(cases):
        result = ({"action": None, "target": {}, "status": "error", "reason": load_error,
                   "inference_ms": 0, "scores": {}, "raw": {}} if load_error else
                  pipeline.translate(case.text, case.actor, case.selected_user, case.selected_group, threshold))
        action_ok = result["status"] != "error" and result["action"] == case.action
        target_ok = result["status"] != "error" and result["target"] == case.target
        status_ok = result["status"] == case.status
        rows.append({"name": case.name, "input": case.text, "actor": case.actor,
                     "selected_user": case.selected_user, "selected_group": case.selected_group,
                     "expected": {"action": case.action, "target": case.target, "status": case.status},
                     "actual": result, "action_correct": action_ok, "target_correct": target_ok,
                     "status_correct": status_ok, "passed": action_ok and target_ok and status_ok})
        confusion[case.action or "<none>"][result["action"] or "<none>"] += 1
        if case.action:
            per_action[case.action].append(action_ok)
        if case.status in ("allowed", "denied"):
            # Oracle action/target tests policy independently of classification errors.
            allowed, _ = authorize(case.actor, case.action, case.target)
            policy_matches.append(allowed == (case.status == "allowed"))
            end_to_end_policy.append(status_ok)
        if progress:
            progress(index+1, len(cases))
    latencies = sorted(row["actual"]["inference_ms"] for row in rows if row["actual"]["status"] != "error" and row["actual"]["inference_ms"] > 0)
    def rate(values):
        """Compute the mean of metric values when observations exist.

        Args:
            values: Sequence of numeric values or boolean correctness indicators.

        Returns:
            The arithmetic mean, or None for an empty sequence.
        """
        return mean(values) if values else None
    return {"created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {**backend.metadata(), "threshold": threshold, "warmup_requests": 1,
                         "fixture_version": 1, "limit": limit},
            "load_error": load_error,
            "summary": {"cases": len(rows), "passed": sum(row["passed"] for row in rows),
                        "errors": sum(row["actual"]["status"] == "error" for row in rows),
                        "action_accuracy": rate([row["action_correct"] for row in rows]),
                        "end_to_end_accuracy": rate([row["passed"] for row in rows]),
                        "authorization_accuracy_oracle_targets": rate(policy_matches),
                        "end_to_end_permission_status_accuracy": rate(end_to_end_policy),
                        "abstention_rate": rate([row["actual"]["status"] == "needs_clarification" for row in rows]),
                        "warm_latency_ms": {"mean": mean(latencies) if latencies else None,
                                            "median": median(latencies) if latencies else None,
                                            "p95": latencies[max(0, ceil(len(latencies)*0.95)-1)] if latencies else None}},
            "per_action_accuracy": {key: rate(values) for key, values in per_action.items()},
            "confusion_matrix": dict(confusion), "results": rows}


def report_json(report):
    """Serialize an evaluation report as indented, Unicode-preserving JSON.

    Args:
        report: Evaluation report dictionary returned by evaluate().

    Returns:
        A JSON string containing the complete report, including raw predictions.

    Raises:
        TypeError: The report contains a value that JSON cannot serialize.
        ValueError: The report contains non-finite numbers or circular references.
    """
    return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)


def report_html(report):
    """Render an evaluation report as a standalone HTML document.

    Args:
        report: Evaluation report dictionary returned by evaluate().

    Returns:
        HTML containing metadata, metrics, the confusion matrix, and a case table.
        Dynamic text is HTML-escaped; raw model responses remain in JSON reports.

    Raises:
        KeyError: A required report or case field is missing.
        TypeError: A rendered value cannot be serialized as JSON.
    """
    def pretty(value):
        """Serialize a value as formatted JSON and escape it for HTML.

        Args:
            value: JSON-serializable value to display in a preformatted block.

        Returns:
            An HTML-escaped JSON string.

        Raises:
            TypeError: The value is not JSON-serializable.
            ValueError: The value contains a circular reference.
        """
        return escape(json.dumps(value, indent=2, ensure_ascii=False))
    rows = "".join(
        f"<tr><td>{escape(row['name'])}</td><td>{escape(row['input'])}</td>"
        f"<td><pre>{pretty(row['expected'])}</pre></td>"
        f"<td><pre>{pretty({k: row['actual'].get(k) for k in ('action', 'target', 'status', 'reason', 'inference_ms')})}</pre></td>"
        f"<td>{'PASS' if row['passed'] else 'FAIL'}</td></tr>" for row in report["results"])
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Laya evaluation</title>"
            "<style>body{font:15px system-ui;margin:2rem}table{border-collapse:collapse;width:100%}"
            "td,th{border:1px solid #bbb;padding:.5rem;text-align:left;vertical-align:top}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Laya evaluation</h1>"
            "<p>Dry run only. Authorization on oracle targets measures policy independently of model accuracy.</p>"
            f"<h2>Run metadata</h2><pre>{pretty(report['metadata'])}</pre>"
            f"<h2>Summary</h2><pre>{pretty(report['summary'])}</pre>"
            f"<h2>Per-action accuracy</h2><pre>{pretty(report['per_action_accuracy'])}</pre>"
            f"<h2>Confusion matrix (expected → predicted)</h2><pre>{pretty(report['confusion_matrix'])}</pre>"
            f"<h2>Cases</h2><table><tr><th>Case</th><th>Input</th><th>Expected</th><th>Actual</th><th>Result</th></tr>{rows}</table></html>")


def main():
    """Run the evaluation CLI and write JSON and HTML report files.

    Read options from sys.argv, validate them, print case progress and summary
    metrics, and create the selected report directory if necessary.

    Returns:
        Exit code 1 if any case has a runtime/model error, otherwise 0. Poor
        model accuracy alone does not produce a nonzero exit code.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid options.
        OSError: The report directory or files cannot be created or written.
    """
    parser = argparse.ArgumentParser(description="Evaluate the actual local Laya model; no simulated inference.")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test case limit")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or (args.limit is not None and args.limit < 1):
        parser.error("threshold must be in [0,1] and limit must be positive")
    report = evaluate(LayaBackend(), args.threshold, args.limit,
                      lambda done, total: print(f"Evaluated {done}/{total}", flush=True))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "evaluation.json").write_text(report_json(report), encoding="utf-8")
    (args.report_dir / "evaluation.html").write_text(report_html(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Reports: {args.report_dir.resolve()}")
    if report["load_error"]:
        print(f"Model error: {report['load_error']}")
    return 1 if report["summary"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
