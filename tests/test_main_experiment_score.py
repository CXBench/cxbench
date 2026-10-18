from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from main_experiment import score
from cxbench import statistics as paper


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


@pytest.fixture
def batch_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(score.TaskSpec, "load", lambda path: SimpleNamespace(id=Path(path).stem))

    def offline_grade(task, trace, state, mode, *, grader):
        assert grader == "rule"
        return state["grade"]

    monkeypatch.setattr(score, "grade_run", offline_grade)

    def make(name="batch", cases=None):
        cases = cases or [{}]
        batch_dir = tmp_path / name
        task_dir = tmp_path / "tasks"
        task_dir.mkdir(exist_ok=True)
        planned, results = [], []
        for index, case in enumerate(cases, 1):
            task_id = f"task-{index}"
            path = task_dir / f"{task_id}.yaml"
            path.write_text(task_id, encoding="utf-8")
            planned.append(
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
            run_dir = batch_dir / "task_runs" / f"{task_id}-success"
            run_dir.mkdir(parents=True)
            _write_json(
                run_dir / "config.json",
                {"task_path": str(path), "grader": "hybrid", "mode": "least_privilege"},
            )
            _write_json(
                run_dir / "grade.json",
                {
                    "task_id": task_id,
                    "grader": "hybrid",
                    "llm_judge": {
                        "process_pass": case.get("judge_raw", True),
                        "findings": [
                            {
                                "passed": case.get("judge", True),
                                "rule_id": "judge-check",
                                "evidence": "trace evidence",
                            }
                        ],
                    },
                },
            )
            _write_jsonl(run_dir / "trace.jsonl", [])
            _write_json(
                run_dir / "final_state.json",
                {
                    "grade": {
                        "outcome_pass": case.get("outcome", True),
                        "rules": [
                            {
                                "rule_id": "policy_summary_before_authority_check",
                                "passed": case.get("hard", True),
                            },
                            {
                                "rule_id": "disclosure_before_confirmation",
                                "passed": case.get("soft", True),
                            },
                            {"rule_id": "no_coverage_promise", "passed": case.get("soft", True)},
                        ],
                        "grading_policy": {
                            "semantic_judge_rules": ["no_coverage_promise"],
                            "hybrid_hint_rules": ["disclosure_before_confirmation"],
                        },
                    }
                },
            )
            results.append(
                {
                    "task_id": task_id,
                    "task_path": str(path),
                    "run_dir": str(run_dir),
                    "ok": True,
                    "verdict": "pass",
                    "final_grader": "hybrid",
                }
            )
        _write_json(
            batch_dir / "manifest.json",
            {"grader": "hybrid", "tasks": planned, "task_count": len(planned)},
        )
        _write_jsonl(batch_dir / "results.jsonl", results)
        return batch_dir

    return make


def test_paper_formula_ignores_retry_dirs_and_preserves_recorded_grades(batch_factory, tmp_path):
    batch = batch_factory(
        cases=[
            {"soft": False, "judge_raw": False},
            {"hard": False},
            {"judge": False, "judge_raw": True},
            {"outcome": False},
        ]
    )
    abandoned = batch / "task_runs" / "task-1-failed-attempt"
    abandoned.mkdir()
    (abandoned / "grade.json").write_text("invalid JSON", encoding="utf-8")
    original = {path: path.read_bytes() for path in batch.rglob("*") if path.is_file()}

    summary = score.score_batches({"model-a": batch}, tmp_path / "scores")

    assert summary["per_model"]["model-a"] == {
        "completed": 4,
        "outcome_pass": 3,
        "both_pass": 1,
        "false_pass": 2,
        "rate": 0.25,
        "out_rate": 0.75,
        "ci95": paper.wilson_ci(1, 4),
    }
    assert summary["n_hard_rules"] == 1
    assert summary["taxonomy"] == {"ordering": {"rules": 1, "active": 1, "fails": 1, "sole": 1}}
    assert summary["loo"] == {"active": 1, "sensitive": 0, "max_pp": 25.0}
    assert summary["mcnemar"]["pairs"] == []
    assert summary["false_pass"]["weak_pct"] is None
    assert len(score._jsonl(tmp_path / "scores" / "rules-model-a.jsonl")) == 4
    assert (tmp_path / "scores" / "report.md").is_file()
    assert original == {path: path.read_bytes() for path in batch.rglob("*") if path.is_file()}


def test_multiple_models_and_optional_reference(batch_factory, tmp_path):
    a = batch_factory("a", [{}, {}])
    b = batch_factory("b", [{}, {"hard": False}])
    summary = score.score_batches({"a": a, "b": b}, tmp_path / "scores")
    assert summary["mcnemar"]["pairs"] == [{"pair": "a vs b", "n10": 1, "n01": 0, "p": 1.0}]
    assert summary["frontier"]["spread_pp"] == 50.0
    referenced = score.score_batches({"a": a, "b": b}, tmp_path / "referenced", reference_model="b")
    assert referenced["mcnemar"]["pairs"] == []
    assert referenced["mcnemar"]["tier_max_p"] == "1.0e+00"
    assert referenced["frontier"]["tier_gap_pp"] == 50.0
    assert referenced["false_pass"]["weak_pct"] == 50.0


@pytest.mark.parametrize(
    "problem, message",
    [
        ("count", "tasks/task_count"),
        ("missing", "incomplete"),
        ("duplicate", "duplicate result"),
        ("duplicate_plan", "duplicate planned"),
        ("error", "did not complete"),
        ("error_verdict", "did not complete"),
        ("grader", "hybrid batch"),
        ("run_grader", "hybrid grading"),
        ("judge", "judge evidence"),
        ("findings", "judge findings"),
        ("hash", "sha256 mismatch"),
        ("trace", "trace.jsonl"),
    ],
)
def test_invalid_batches_do_not_write_outputs(batch_factory, tmp_path, problem, message):
    batch = batch_factory()
    manifest = score._json(batch / "manifest.json")
    results = score._jsonl(batch / "results.jsonl")
    run = Path(results[0]["run_dir"])
    grade = score._json(run / "grade.json")
    if problem == "count":
        manifest["task_count"] = 2
    elif problem == "missing":
        results.clear()
    elif problem == "duplicate":
        results.append(results[0])
    elif problem == "duplicate_plan":
        manifest["tasks"].append(manifest["tasks"][0])
        manifest["task_count"] = 2
    elif problem == "error":
        results[0]["ok"] = False
    elif problem == "error_verdict":
        results[0]["verdict"] = "error"
    elif problem == "grader":
        manifest["grader"] = "rule"
    elif problem == "run_grader":
        grade["grader"] = "rule"
    elif problem == "judge":
        del grade["llm_judge"]
    elif problem == "findings":
        del grade["llm_judge"]["findings"][0]["passed"]
    elif problem == "hash":
        manifest["tasks"][0]["sha256"] = "0" * 64
    elif problem == "trace":
        (run / "trace.jsonl").unlink()
    _write_json(batch / "manifest.json", manifest)
    _write_jsonl(batch / "results.jsonl", results)
    _write_json(run / "grade.json", grade)
    output = tmp_path / "scores"

    with pytest.raises(score.ScoreError, match=message):
        score.score_batches({"a": batch}, output)

    assert not output.exists()


def test_unpaired_models_and_unknown_reference_are_rejected(batch_factory, tmp_path):
    a = batch_factory("a", [{}, {}])
    b = batch_factory("b")
    with pytest.raises(score.ScoreError, match="same planned tasks"):
        score.score_batches({"a": a, "b": b}, tmp_path / "scores")
    with pytest.raises(score.ScoreError, match="reference model"):
        score.score_batches({"a": a}, tmp_path / "scores", reference_model="missing")
    assert not (tmp_path / "scores").exists()


def test_scorer_cli(batch_factory, tmp_path):
    batch = batch_factory()
    assert score.main(["--batch", f"model={batch}", "--output", str(tmp_path / "scores")]) == 0
    with pytest.raises(SystemExit):
        score.main(
            [
                "--batch",
                f"model={batch}",
                "--batch",
                f"model={batch}",
                "--output",
                str(tmp_path / "bad"),
            ]
        )


def test_slash_labels_are_preserved_with_safe_artifact_names(batch_factory, tmp_path):
    batch = batch_factory()
    summary = score.score_batches({"provider/model": batch}, tmp_path / "scores")
    assert "provider/model" in summary["per_model"]
    assert (tmp_path / "scores" / "rules-provider_model.jsonl").is_file()
    with pytest.raises(score.ScoreError, match="collide"):
        score.score_batches(
            {"provider/model": batch, "provider_model": batch}, tmp_path / "collision"
        )


def test_existing_output_is_preserved(batch_factory, tmp_path):
    batch = batch_factory()
    output = tmp_path / "scores"
    output.mkdir()
    (output / "summary.json").write_text("existing scores", encoding="utf-8")
    with pytest.raises(score.ScoreError, match="already exists"):
        score.score_batches({"a": batch}, output)
    assert (output / "summary.json").read_text() == "existing scores"
