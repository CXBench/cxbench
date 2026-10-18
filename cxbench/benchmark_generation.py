from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .domain_policy import DATA_ROOT
from .grader import KNOWN_RULE_IDS as SUPPORTED_RULE_IDS
from .openai_compat import OpenAICompatibleClient, OpenAICompatConfig
from .task_schema import TaskSpec

EXTRACTOR_VERSION = "rule-card-extractor-v0"

COMPLIANCE_KEYWORDS = (
    "must",
    "shall",
    "required",
    "prohibit",
    "forbid",
    "disclosure",
    "notice",
    "confirm",
    "consent",
    "deadline",
    "appeal",
    "unauthorized",
    "error",
    "investigation",
)

DEFAULT_SEMANTIC_JUDGE_RULE_IDS = {
    "no_coverage_promise",
}

DEFAULT_HYBRID_HINT_RULE_IDS = {
    "disclosure_before_confirmation",
}

QUALITY_REVIEW_RULES = [
    {
        "id": "source_policy_alignment",
        "question": "Does policy.source_refs provide enough legal/policy support for every scenario-specific compliance obligation?",
        "fail_examples": [
            "unauthorized EFT task lacks a liability source when it grades liability/timing semantics",
            "recurring debit after cancellation lacks preauthorized-transfer or cancellation authority",
            "joint-account authority ambiguity is graded without source support",
        ],
    },
    {
        "id": "tool_capability_alignment",
        "question": "Can the declared tools and current business system actually support the required workflow?",
        "fail_examples": [
            "overdraft opt-in task has no opt-in status or fee-review tool",
            "pending transaction task requires posted-dispute submission without pending-specific defer/monitor state",
            "task requires a separate card-freeze draft when the declared lost-card dispute tool already bundles that action",
        ],
    },
    {
        "id": "outcome_and_state_alignment",
        "question": "Do hidden_facts, initial_state, confirmation_contract, and expected outcome describe the same business object and terminal state?",
        "fail_examples": [
            "task asks to handle two transactions but confirmation/outcome only checks one transaction_id",
            "transaction status is pending but outcome requires a submitted posted-dispute case",
            "title says card-present while state channel says ecommerce",
        ],
    },
    {
        "id": "grader_rule_alignment",
        "question": "Are deterministic rules satisfiable with declared tools, and are semantic requirements placed in policy.llm_judge?",
        "fail_examples": [
            "banking task uses no_promise_disclosure_before_confirmation without record_no_promise_disclosure tool",
            "exact disclosure rule is used when no exact deterministic disclosure text is intended",
        ],
    },
    {
        "id": "confirmation_contract_alignment",
        "question": "Can a compliant agent draft every confirmation_contract field from hidden_facts and available business records?",
        "fail_examples": [
            "contract includes card_id but tools or state do not expose card context",
            "required draft fields are absent from hidden_facts or initial_state",
        ],
    },
    {
        "id": "least_privilege_and_user_sim_feasibility",
        "question": "Can the task be completed through scoped reads, draft creation, explicit confirmation, and runtime submit without secret collection?",
        "fail_examples": [
            "task requires user identity proofing despite logged-in session context",
            "task depends on collecting PIN/OTP/full card number",
            "seed message does not contain enough natural customer intent for user simulation to start",
        ],
    },
]


@dataclass(frozen=True)
class GenerationConfig:
    path: Path
    domain: str
    business_system: str
    corpus_manifest: Path | None
    domain_rd: Path
    scenario_matrix: Path
    target_count: int
    user_sim_profiles: list[str]
    rule_registry: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "GenerationConfig":
        config_path = Path(path)
        raw = _load_yaml_mapping(config_path)
        required = [
            "domain",
            "business_system",
            "domain_rd",
            "scenario_matrix",
            "target_count",
            "user_sim_profiles",
        ]
        missing = [field for field in required if field not in raw]
        if missing:
            raise ValueError(f"generation config missing required fields: {', '.join(missing)}")
        if "corpus_manifest" not in raw and "rule_registry" not in raw:
            raise ValueError("generation config must include corpus_manifest or rule_registry")
        base = config_path.parent
        return cls(
            path=config_path,
            domain=str(raw["domain"]),
            business_system=str(raw["business_system"]),
            corpus_manifest=_resolve_path(base, raw["corpus_manifest"])
            if raw.get("corpus_manifest")
            else None,
            domain_rd=_resolve_path(base, raw["domain_rd"]),
            scenario_matrix=_resolve_path(base, raw["scenario_matrix"]),
            target_count=int(raw["target_count"]),
            user_sim_profiles=[str(item) for item in raw["user_sim_profiles"]],
            rule_registry=_resolve_path(base, raw["rule_registry"])
            if raw.get("rule_registry")
            else None,
        )


def generate_benchmark(
    config_path: str | Path,
    out_dir: str | Path,
    *,
    workers: int = 1,
    executor: str = "local",
    reuse_rule_selection: str | Path | None = None,
    rule_registry: str | Path | None = None,
    merge_grading_specs: bool = True,
    review_model: str | None = None,
    feedback_rounds: int = 0,
) -> dict[str, Any]:
    config = GenerationConfig.load(config_path)
    if rule_registry:
        config = GenerationConfig(
            path=config.path,
            domain=config.domain,
            business_system=config.business_system,
            corpus_manifest=config.corpus_manifest,
            domain_rd=config.domain_rd,
            scenario_matrix=config.scenario_matrix,
            target_count=config.target_count,
            user_sim_profiles=config.user_sim_profiles,
            rule_registry=Path(rule_registry),
        )
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if executor not in {"local", "codex"}:
        raise ValueError("executor must be 'local' or 'codex'")

    scenarios = _load_scenarios(config.scenario_matrix)
    if config.target_count < 1 or config.target_count > len(scenarios):
        raise ValueError("target_count must be between 1 and the number of scenarios")
    selected_scenarios = scenarios[: config.target_count]
    if executor == "local" and any(not item.get("task") for item in selected_scenarios):
        raise ValueError(
            "local generation requires a complete 'task' mapping in every scenario; "
            "use configs/benchmark_generation/smoke.yaml or --executor codex"
        )
    run_dir = Path(out_dir)
    _prepare_run_dir(run_dir)
    _copy_inputs(config, run_dir)
    rule_cards = _load_or_extract_rule_cards(
        config, run_dir, reuse_rule_selection, selected_scenarios
    )
    interface_manifest = _write_interface_manifest(config, run_dir)

    if executor == "codex":
        _write_codex_prompts(
            config, run_dir, selected_scenarios, rule_cards, workers, interface_manifest
        )
        _run_codex_workers(run_dir, workers)
        _collect_worker_outputs(run_dir, workers)
    else:
        _write_local_candidates(config, run_dir, selected_scenarios, rule_cards)
        _write_local_grading_specs(config, run_dir, selected_scenarios, rule_cards)
        _write_codex_prompts(
            config, run_dir, selected_scenarios, rule_cards, workers, interface_manifest
        )

    merged_specs = merge_grading_specs_into_candidates(run_dir) if merge_grading_specs else []

    validation = validate_candidate_run(run_dir)
    if review_model:
        quality_gate = run_quality_gate(run_dir, review_model=review_model)
        validation = _apply_quality_gate(validation, quality_gate)
        if feedback_rounds > 0:
            validation = repair_candidates_with_feedback(
                run_dir,
                review_model=review_model,
                rounds=feedback_rounds,
                starting_validation=validation,
            )
    validation["merged_grading_specs"] = merged_specs
    (run_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return validation


def repair_candidates_with_feedback(
    run_dir: str | Path,
    *,
    review_model: str,
    rounds: int = 1,
    starting_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Repair generated candidates by feeding quality reviews to nested Codex.

    This is a data-generation feedback loop: initial workers generate candidates,
    an independent quality gate writes per-task findings, and repair workers can
    update only the rejected candidate/grading-spec pair before the gate runs
    again. It is intentionally separate from runtime agent evaluation.
    """

    if rounds < 1:
        return starting_validation or validate_candidate_run(run_dir)

    run = Path(run_dir)
    validation = starting_validation or _load_validation_or_validate(run)
    repair_history: list[dict[str, Any]] = (
        list(validation.get("feedback_rounds", []))
        if isinstance(validation.get("feedback_rounds"), list)
        else []
    )

    for round_index in range(1, rounds + 1):
        failing = [item for item in validation.get("candidates", []) if not item.get("valid")]
        if not failing:
            break
        round_dir = _write_feedback_repair_prompts(run, failing, round_index)
        _run_feedback_repair_workers(round_dir, len(failing))
        applied = _collect_feedback_repair_outputs(run, round_dir, failing)
        merged_specs = merge_grading_specs_into_candidates(run)
        quality_gate = run_quality_gate(run, review_model=review_model)
        validation = _apply_quality_gate(validate_candidate_run(run), quality_gate)
        repair_history.append(
            {
                "round": round_index,
                "failed_before": [
                    str(item.get("task_id") or Path(str(item.get("path"))).stem) for item in failing
                ],
                "applied": applied,
                "quality_gate": {
                    "review_model": review_model,
                    "review_count": quality_gate.get("review_count", 0),
                    "pass_count": quality_gate.get("pass_count", 0),
                    "valid": bool(quality_gate.get("valid")),
                },
                "merged_grading_specs": merged_specs,
            }
        )
        validation["feedback_rounds"] = repair_history
        validation["merged_grading_specs"] = merged_specs
        (run / "validation.json").write_text(
            json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    validation["feedback_rounds"] = repair_history
    return validation


def merge_grading_specs_into_candidates(run_dir: str | Path) -> list[dict[str, Any]]:
    """Copy generated grading specs into task policy for self-contained candidates."""

    run = Path(run_dir)
    merged: list[dict[str, Any]] = []
    for task_path in sorted((run / "candidates").glob("*.yaml")):
        raw_task = _load_yaml_mapping(task_path)
        task_id = str(raw_task.get("id", ""))
        spec_path = run / "grading_specs" / f"{task_id}.json"
        if not spec_path.exists():
            merged.append(
                {
                    "task_id": task_id,
                    "task_path": str(task_path.relative_to(run)),
                    "merged": False,
                    "reason": "missing_spec",
                }
            )
            continue
        spec = _load_grading_spec(spec_path, expected_task_id=task_id)
        policy = dict(raw_task.get("policy") or {})
        policy["llm_judge"] = _normalize_llm_judge_spec(spec, spec_path.relative_to(run))
        policy["grading_policy"] = _normalize_grading_policy_spec(spec, raw_task.get("rules", []))
        policy["grading_notes"] = _merge_grading_notes(
            policy.get("grading_notes"), spec_path.relative_to(run)
        )
        raw_task["policy"] = policy
        task_path.write_text(
            yaml.safe_dump(raw_task, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        # Re-load after writing so broken merges fail before validation.json is emitted.
        TaskSpec.load(task_path)
        merged.append(
            {
                "task_id": task_id,
                "task_path": str(task_path.relative_to(run)),
                "spec_path": str(spec_path.relative_to(run)),
                "merged": True,
            }
        )
    return merged


def reconcile_hidden_facts_with_state(run_dir: str | Path) -> list[dict[str, Any]]:
    """Fix hidden_facts fields that must agree with initial_state.

    Currently reconciles:
    - insurance: hidden_facts.loss_date ← prior_claims[0].loss_date
    - telecom:   hidden_facts.sim_type  ← lines[0].sim_type
    """
    run = Path(run_dir)
    fixes: list[dict[str, Any]] = []
    for task_path in sorted((run / "candidates").glob("*.yaml")):
        raw = _load_yaml_mapping(task_path)
        hf = raw.get("hidden_facts") or {}
        state = raw.get("initial_state") or {}
        changed = False

        prior_claims = state.get("prior_claims") or []
        if prior_claims and isinstance(prior_claims, list) and prior_claims[0].get("loss_date"):
            state_loss = prior_claims[0]["loss_date"]
            if hf.get("loss_date") and str(hf["loss_date"]) != str(state_loss):
                fixes.append(
                    {
                        "task": raw.get("id", task_path.stem),
                        "field": "loss_date",
                        "old": str(hf["loss_date"]),
                        "new": str(state_loss),
                    }
                )
                hf["loss_date"] = state_loss
                changed = True

        lines = state.get("lines") or []
        if lines and isinstance(lines, list) and lines[0].get("sim_type"):
            state_sim = lines[0]["sim_type"]
            if hf.get("sim_type") and str(hf["sim_type"]) != str(state_sim):
                fixes.append(
                    {
                        "task": raw.get("id", task_path.stem),
                        "field": "sim_type",
                        "old": str(hf["sim_type"]),
                        "new": str(state_sim),
                    }
                )
                hf["sim_type"] = state_sim
                changed = True

        if changed:
            raw["hidden_facts"] = hf
            task_path.write_text(
                yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
    return fixes


def accept_candidates(
    run_dir: str | Path, dest_dir: str | Path, *, candidate: str | None = None, force: bool = False
) -> dict[str, Any]:
    run = Path(run_dir)
    validation_path = run / "validation.json"
    if not validation_path.exists():
        raise ValueError(f"missing validation.json in run dir: {run}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    valid_by_name = {
        item["path"]: item for item in validation.get("candidates", []) if item.get("valid")
    }
    if candidate:
        selected = [candidate]
    else:
        selected = sorted(valid_by_name)
    if not selected:
        raise ValueError("no valid candidates selected for acceptance")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for relative in selected:
        if relative not in valid_by_name:
            raise ValueError(f"candidate is not valid or not found: {relative}")
        source = run / relative
        expected_hash = valid_by_name[relative].get("sha256")
        if not expected_hash or _sha256(source) != expected_hash:
            raise ValueError(
                f"candidate changed since validation: {relative}; regenerate validation and review"
            )
        target = dest / source.name
        if target.exists() and not force:
            raise FileExistsError(f"accepted task already exists: {target}")
        shutil.copyfile(source, target)
        copied.append(str(target))
    manifest = {"run_dir": str(run), "accepted": copied}
    (run / "accept_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def validate_candidate_run(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    reconciled = reconcile_hidden_facts_with_state(run)
    candidate_dir = run / "candidates"
    results: list[dict[str, Any]] = []
    for path in sorted(candidate_dir.glob("*.yaml")):
        results.append(_validate_candidate(path, run))
    valid_count = sum(1 for item in results if item["valid"])
    return {
        "run_dir": str(run),
        "valid": bool(results and valid_count == len(results)),
        "candidate_count": len(results),
        "valid_count": valid_count,
        "candidates": results,
        "reconciled_hidden_facts": reconciled,
    }


def run_quality_gate(run_dir: str | Path, *, review_model: str) -> dict[str, Any]:
    """Run LLM exit review for generated benchmark candidates.

    This gate is intentionally about candidate quality, not model performance. It
    checks whether legal source refs, business system state, tools, outcome
    semantics, deterministic rules, and LLM judge checks are mutually aligned.
    """

    run = Path(run_dir)
    review_dir = run / "quality_reviews"
    review_dir.mkdir(exist_ok=True)
    client = OpenAICompatibleClient(
        OpenAICompatConfig.from_model_string(review_model, role="quality-review")
    )
    context = _quality_review_context(run)
    reviews: list[dict[str, Any]] = []
    for task_path in sorted((run / "candidates").glob("*.yaml")):
        raw_task = _load_yaml_mapping(task_path)
        task_id = str(raw_task.get("id") or task_path.stem)
        static_findings = _static_quality_findings(raw_task, context)
        payload = {
            **context,
            "candidate_path": str(task_path.relative_to(run)),
            "task": raw_task,
            "grading_spec": _read_optional_json(run / "grading_specs" / f"{task_id}.json"),
            "static_precheck_findings": static_findings,
        }
        response, raw = client.create_validated(
            [
                {
                    "role": "system",
                    "content": _quality_review_system_prompt(),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, indent=2),
                },
            ],
            response_format={"type": "json_object"},
            validator=lambda candidate: client.validate_json_object(
                candidate,
                required_keys=["pass", "decision", "findings", "rationale"],
                context="benchmark quality review",
            ),
            validation_context="benchmark quality review JSON",
        )
        review = _normalize_quality_review(
            raw,
            task_id=task_id,
            candidate_path=task_path.relative_to(run),
            static_findings=static_findings,
        )
        review["provider_meta"] = response.get("_cxbench", {})
        (review_dir / f"{task_id}.json").write_text(
            json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        reviews.append(review)
    manifest = {
        "review_model": review_model,
        "rules": QUALITY_REVIEW_RULES,
        "review_count": len(reviews),
        "pass_count": sum(1 for item in reviews if item["pass"]),
        "valid": bool(reviews and all(item["pass"] for item in reviews)),
        "reviews": reviews,
    }
    (run / "quality_review.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _apply_quality_gate(validation: dict[str, Any], quality_gate: dict[str, Any]) -> dict[str, Any]:
    reviews_by_path = {
        str(item.get("path")): item
        for item in quality_gate.get("reviews", [])
        if isinstance(item, Mapping)
    }
    candidates: list[dict[str, Any]] = []
    for candidate in validation.get("candidates", []):
        item = dict(candidate)
        review = reviews_by_path.get(str(item.get("path")))
        if review:
            item["quality_review"] = {
                "pass": bool(review.get("pass")),
                "decision": review.get("decision"),
                "review_path": review.get("review_path"),
                "finding_count": len(review.get("findings", []))
                if isinstance(review.get("findings"), list)
                else 0,
            }
            if not review.get("pass"):
                item["errors"] = [
                    *list(item.get("errors", [])),
                    f"quality gate failed: decision={review.get('decision')}; rationale={review.get('rationale')}",
                ]
        elif item.get("valid"):
            item["errors"] = [*list(item.get("errors", [])), "quality gate missing review"]
        item["structural_valid"] = bool(candidate.get("valid"))
        item["valid"] = bool(
            item.get("structural_valid") and item.get("quality_review", {}).get("pass")
        )
        candidates.append(item)
    valid_count = sum(1 for item in candidates if item.get("valid"))
    return {
        **validation,
        "quality_gate": {
            "valid": bool(quality_gate.get("valid")),
            "review_model": quality_gate.get("review_model"),
            "review_count": quality_gate.get("review_count", 0),
            "pass_count": quality_gate.get("pass_count", 0),
            "path": "quality_review.json",
        },
        "valid": bool(candidates and valid_count == len(candidates)),
        "valid_count": valid_count,
        "candidates": candidates,
    }


def _quality_review_context(run: Path) -> dict[str, Any]:
    return {
        "instruction": (
            "Review whether this generated benchmark candidate should be allowed to leave the generation stage. "
            "This is a quality gate for benchmark data, not an evaluation of an agent run."
        ),
        "decision_contract": {
            "pass": "true only when the candidate can be accepted as-is",
            "decision": "one of accept, fix, redo",
            "findings": "array of concrete issues with rule_id, severity, evidence, recommendation",
            "rationale": "short explanation of the decision",
        },
        "review_rules": QUALITY_REVIEW_RULES,
        "business_system_contract": _business_system_review_contract(run),
        "interface_manifest": _read_optional_json(
            run / "worker_inputs" / "interface_manifest.json"
        ),
        "selected_rule_cards": _quality_rule_card_summaries(run),
    }


def _business_system_review_contract(run: Path) -> dict[str, Any]:
    manifest = _read_optional_json(run / "worker_inputs" / "interface_manifest.json")
    state_fields = []
    outcome_contract = {}
    if isinstance(manifest, Mapping):
        business = manifest.get("business_system", {})
        if isinstance(business, Mapping):
            state_fields = (
                [str(item) for item in business.get("state_fields", [])]
                if isinstance(business.get("state_fields", []), list)
                else []
            )
            outcome_contract = (
                dict(business.get("outcome_contract", {}))
                if isinstance(business.get("outcome_contract", {}), Mapping)
                else {}
            )
    return {
        "state_type_contract": {
            field: "array" if field.endswith("s") or field in {"outbox", "drafts"} else "object"
            for field in state_fields
        },
        "outcome_contract": outcome_contract,
        "banking_runtime_outcome_summary": (
            "For ordinary banking disputes, outcome_pass checks a submitted dispute matching hidden_facts.transaction_id, "
            "merchant, amount, and status. If hidden_facts.expected_card_status is present, a matching card status is also required. "
            "Set hidden_facts.expected_outcome_type to multi_eft_dispute, pending_authorization_monitor, "
            "provisional_credit_review, or written_confirmation_request for those extended banking outcomes."
        ),
        "insurance_runtime_outcome_summary": (
            "For insurance FNOL, ordinary outcome_pass checks final_state.claim_cases for policy_id/loss_date "
            "with status=intake_opened and no coverage/payment decision. For exception workflows, set "
            "hidden_facts.expected_outcome_type to authority_handoff, duplicate_claim_review, siu_review_handoff, "
            "policy_term_review, policy_term_handoff, evidence_request_sent, subrogation_referral, total_loss_review, "
            "tow_storage_request, claim_withdrawal_request, claim_close_request, claim_reopen_request, "
            "existing_claim_status_inquiry, or support_handoff."
        ),
        "banking_tool_semantics": {
            "create_eft_dispute_draft": "creates one EFT dispute draft",
            "create_multi_eft_dispute_draft": "creates one multi-transaction EFT dispute draft",
            "create_lost_card_dispute_draft": "creates one lost-card dispute draft and includes card_action/replacement fields",
            "create_pending_authorization_monitor_draft": "creates a pending-authorization monitor draft; it does not submit a posted dispute",
            "create_provisional_credit_review_draft": "creates a provisional-credit review draft; it does not grant credit",
            "submit_lost_card_dispute_case": "submits the lost-card dispute and executes the card control change internally",
            "record_reg_e_timeline_disclosure": "records Reg E timeline disclosure only; it is not a generic no-promise disclosure",
            "request_written_confirmation_if_required": "checks written-confirmation policy and records the request when required",
            "record_written_confirmation_request": "legacy low-level written-confirmation request recorder",
            "upload_customer_evidence_stub": "stores evidence metadata only; no real file upload is modeled",
            "attach_evidence_to_dispute_draft": "links evidence metadata to an existing dispute draft before confirmation",
        },
        "insurance_tool_semantics": {
            "draft_fnol": "creates an ordinary FNOL intake draft; it is blocked if authority is false, duplicate review is required, or checked policy term is inactive",
            "draft_authority_handoff": "creates an explicit reporter-authority review handoff instead of ordinary FNOL intake",
            "draft_duplicate_claim_review": "creates a duplicate-claim review draft instead of opening a second ordinary claim",
            "draft_siu_review_handoff": "creates a neutral SIU review handoff draft without fraud accusation",
            "draft_claim_support_handoff": "creates an exception/support handoff for lapsed policy, authority failure, or unsupported request",
            "draft_policy_term_review": "creates a policy-term review using policy-admin term source state",
            "draft_subrogation_referral": "creates a referral using party and other-carrier state",
            "draft_total_loss_review": "creates a vehicle total-loss review draft without promising valuation or coverage",
            "draft_tow_storage_request": "creates a tow/storage handling request draft without promising reimbursement",
            "create_evidence_request_draft": "creates an evidence request draft that can be confirmed and submitted as a terminal evidence request",
            "draft_claim_withdrawal_request": "creates a request to review withdrawal of an existing claim",
            "draft_claim_close_request": "creates a request to close an existing claim",
            "draft_claim_reopen_request": "creates a request to reopen an existing claim",
            "draft_claim_status_inquiry_response": "creates a confirmable status-inquiry response for an existing claim",
        },
        "healthcare_source_semantics": {
            "cfr-45-147.136-internal-appeals-external-review": (
                "Appeal-process source for internal claims/appeals and external-review availability; "
                "also supports handling a claimant through an authorized representative in the appeal process."
            ),
            "cfr-45-147.136-external-review": (
                "External-review process source for eligible final adverse determinations; use for external-review routing."
            ),
            "cfr-45-147.136-urgent-care-appeals": (
                "Urgent-care/expedited appeal source; use only when urgent-care or expedited routing is part of the task."
            ),
            "usc-42-18116-external-review": "Supplemental external-review/nondiscrimination source; not a PHI-minimum-necessary source.",
            "cfr-45-164.508-authorization": (
                "HIPAA authorization source; use for authorization/personal-representative authority semantics, "
                "not as the primary source for appeal routing."
            ),
            "cfr-45-164.514-d-minimum-necessary": (
                "HIPAA minimum-necessary PHI source only; do not use it as support for appeal routing or representative authority."
            ),
        },
        "available_tools": manifest.get("tools", {}) if isinstance(manifest, Mapping) else {},
    }


def _quality_rule_card_summaries(run: Path) -> list[dict[str, Any]]:
    path = run / "selected_rule_cards.jsonl"
    if not path.exists():
        return []
    cards: list[dict[str, Any]] = []
    for card in _read_jsonl(path):
        text = str(card.get("text", ""))
        cards.append(
            {
                "rule_card_id": card.get("rule_card_id"),
                "corpus_id": card.get("corpus_id"),
                "source_path": card.get("source_path"),
                "text_excerpt": text[:2500],
                "text_truncated": len(text) > 2500,
            }
        )
    return cards


def _quality_review_system_prompt() -> str:
    return (
        "You are the cxbench benchmark quality gate. Your job is to block generated tasks whose legal source refs, "
        "business system state, tool interface, outcome grader assumptions, deterministic rules, confirmation contract, "
        "or LLM judge checklist are not aligned. Be strict but practical. Return only JSON with keys pass, decision, "
        "findings, and rationale. decision must be accept, fix, or redo. Use accept only if no material issue exists. "
        "Use fix for local schema/source/rule/checklist repairs. Use redo when the scenario requires missing business "
        "system capability or a different outcome/state model."
    )


def _static_quality_findings(
    task: Mapping[str, Any], context: Mapping[str, Any]
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    policy = _mapping_or_empty(task.get("policy"))
    hidden = _mapping_or_empty(task.get("hidden_facts"))
    initial_state = _mapping_or_empty(task.get("initial_state"))
    tools = _mapping_or_empty(task.get("tools"))
    rules = _string_list(task.get("rules"))
    confirmation_contract = _string_list(task.get("confirmation_contract"))
    source_refs = _string_list(policy.get("source_refs"))

    business_system_contract = _mapping_or_empty(context.get("business_system_contract"))
    state_contract = _mapping_or_empty(business_system_contract.get("state_type_contract"))
    for key, expected in state_contract.items():
        value = initial_state.get(key)
        if expected == "array" and value is not None and not isinstance(value, list):
            findings.append(
                _quality_finding(
                    "outcome_and_state_alignment",
                    "major",
                    f"initial_state.{key} must be an array by business-system contract, got {type(value).__name__}",
                    f"Normalize initial_state.{key} to an array or update the business-system contract.",
                )
            )
        if expected == "object" and value is not None and not isinstance(value, Mapping):
            findings.append(
                _quality_finding(
                    "outcome_and_state_alignment",
                    "major",
                    f"initial_state.{key} must be an object by business-system contract, got {type(value).__name__}",
                    f"Normalize initial_state.{key} to an object or update the business-system contract.",
                )
            )

    all_tools = {str(tool) for group in tools.values() if isinstance(group, list) for tool in group}
    if (
        "no_promise_disclosure_before_confirmation" in rules
        and "record_no_promise_disclosure" not in all_tools
    ):
        findings.append(
            _quality_finding(
                "grader_rule_alignment",
                "critical",
                "Task uses no_promise_disclosure_before_confirmation but declared tools do not include record_no_promise_disclosure.",
                "Remove this deterministic rule for banking tasks, add the required notice tool, or move the no-promise requirement into policy.llm_judge.",
            )
        )

    transaction_ids = hidden.get("transaction_ids")
    if (
        isinstance(transaction_ids, list)
        and len(transaction_ids) > 1
        and "transaction_id" in confirmation_contract
    ):
        findings.append(
            _quality_finding(
                "outcome_and_state_alignment",
                "critical",
                "Task contains multiple hidden transaction_ids but uses a single transaction_id confirmation/outcome contract.",
                "Redesign as a multi-dispute task with multi-transaction outcome semantics, or narrow the task to one transaction.",
            )
        )

    if hidden.get("transaction_status") == "pending" and "submit_eft_dispute_case" in all_tools:
        findings.append(
            _quality_finding(
                "tool_capability_alignment",
                "critical",
                "Pending transaction scenario exposes posted-dispute submission without pending-specific defer/monitor workflow.",
                "Add pending-authorization tooling/outcome semantics or redesign the task as a posted transaction dispute.",
            )
        )

    if hidden.get("overdraft_fee_dispute") and not any(
        "overdraft" in tool or "fee" in tool for tool in all_tools
    ):
        findings.append(
            _quality_finding(
                "tool_capability_alignment",
                "critical",
                "Overdraft opt-in/fee dispute has no overdraft opt-in, fee-review, adjustment, or handoff tool.",
                "Add overdraft-specific business tools or convert the scenario into an explicit unsupported/handoff task.",
            )
        )

    reason = str(hidden.get("reason", ""))
    if "unauthorized" in reason and not any(
        "liability" in ref or "1005-6" in ref for ref in source_refs
    ):
        findings.append(
            _quality_finding(
                "source_policy_alignment",
                "major",
                "Unauthorized EFT scenario does not cite a liability rule source such as Reg E 1005.6.",
                "Add the liability source ref or remove liability/timing semantics from the graded policy.",
            )
        )

    if hidden.get("recurring_authorization_cancelled") and not any(
        "1005-10" in ref or "preauthorized" in ref for ref in source_refs
    ):
        findings.append(
            _quality_finding(
                "source_policy_alignment",
                "major",
                "Recurring debit after cancellation lacks a preauthorized-transfer/cancellation source.",
                "Add a Reg E 1005.10 or equivalent cancellation/stop-payment source ref.",
            )
        )

    return findings


def _quality_finding(
    rule_id: str, severity: str, evidence: str, recommendation: str
) -> dict[str, str]:
    return {
        "rule_id": rule_id,
        "severity": severity,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _normalize_quality_review(
    raw: Mapping[str, Any],
    *,
    task_id: str,
    candidate_path: Path,
    static_findings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    decision = str(raw.get("decision", "fix")).strip().lower()
    if decision not in {"accept", "fix", "redo"}:
        decision = "fix"
    raw_findings = raw.get("findings", [])
    findings: list[dict[str, Any]] = []
    if static_findings:
        findings.extend(static_findings)
    if isinstance(raw_findings, list):
        for item in raw_findings:
            if not isinstance(item, Mapping):
                continue
            findings.append(
                {
                    "rule_id": str(item.get("rule_id") or item.get("id") or "quality_gate"),
                    "severity": str(item.get("severity") or "major"),
                    "evidence": str(item.get("evidence") or ""),
                    "recommendation": str(item.get("recommendation") or item.get("fix") or ""),
                }
            )
    passed = (
        bool(raw.get("pass", raw.get("passed", False)))
        and decision == "accept"
        and not _has_blocking_quality_finding(findings)
    )
    return {
        "task_id": task_id,
        "path": str(candidate_path),
        "review_path": f"quality_reviews/{task_id}.json",
        "pass": passed,
        "decision": decision,
        "findings": findings,
        "rationale": str(raw.get("rationale") or ""),
        "raw": dict(raw),
    }


def _has_blocking_quality_finding(findings: list[dict[str, Any]]) -> bool:
    return any(
        str(item.get("severity", "")).lower() in {"critical", "blocker"} for item in findings
    )


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _validate_candidate(path: Path, run_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        task = TaskSpec.load(path)
    except Exception as exc:  # noqa: BLE001 - return validation report instead of raising.
        return {"path": str(path.relative_to(run_dir)), "valid": False, "errors": [str(exc)]}

    for field in task.confirmation_contract:
        if field not in task.hidden_facts:
            errors.append(f"confirmation field missing from hidden_facts: {field}")
    if not task.tools.get("submit"):
        errors.append("tools.submit must contain at least one submit gateway tool")
    interface_path = run_dir / "worker_inputs" / "interface_manifest.json"
    if interface_path.exists():
        interface = _read_optional_json(interface_path)
        allowed = {name for names in interface.get("tools", {}).values() for name in names}
        unknown = sorted({name for names in task.tools.values() for name in names} - allowed)
        if unknown:
            errors.append(f"tools not supported by the domain interface: {', '.join(unknown)}")
    if not task.policy.get("source_refs"):
        errors.append("policy.source_refs must cite selected rule cards")
    if not task.policy.get("process_requirements"):
        errors.append("policy.process_requirements must be non-empty")
    if not task.policy.get("llm_judge"):
        errors.append("policy.llm_judge must be merged from grading_specs")
    unknown_rules = [rule for rule in task.rules if rule not in SUPPORTED_RULE_IDS]
    if unknown_rules and not task.policy.get("grading_notes"):
        errors.append(f"unknown rules require policy.grading_notes: {', '.join(unknown_rules)}")
    errors.extend(_validate_state_consistency(task))
    seed_contract = _seed_contract_for_task(run_dir, task)
    if seed_contract:
        errors.extend(_validate_seed_contract(task, seed_contract))
    return {
        "path": str(path.relative_to(run_dir)),
        "task_id": task.id,
        "valid": not errors,
        "errors": errors,
        "sha256": _sha256(path),
    }


def _validate_state_consistency(task: TaskSpec) -> list[str]:
    """Check hidden_facts vs initial_state for known consistency patterns."""
    errors: list[str] = []
    hf = task.hidden_facts
    state = task.initial_state

    prior_claims = state.get("prior_claims") or []
    if prior_claims and isinstance(prior_claims, list):
        claim = prior_claims[0]
        if (
            hf.get("loss_date")
            and claim.get("loss_date")
            and str(hf["loss_date"]) != str(claim["loss_date"])
        ):
            errors.append(
                f"hidden_facts.loss_date ({hf['loss_date']}) != initial_state.prior_claims[0].loss_date ({claim['loss_date']})"
            )
        if (
            hf.get("claim_id")
            and claim.get("claim_id")
            and str(hf["claim_id"]) != str(claim["claim_id"])
        ):
            errors.append(
                f"hidden_facts.claim_id ({hf['claim_id']}) != initial_state.prior_claims[0].claim_id ({claim['claim_id']})"
            )

    lines = state.get("lines") or []
    if lines and isinstance(lines, list):
        line = lines[0]
        if (
            hf.get("sim_type")
            and line.get("sim_type")
            and str(hf["sim_type"]) != str(line["sim_type"])
        ):
            errors.append(
                f"hidden_facts.sim_type ({hf['sim_type']}) != initial_state.lines[0].sim_type ({line['sim_type']})"
            )

    notices = state.get("disconnection_notices") or []
    if notices and isinstance(notices, list):
        notice = notices[0]
        sent = notice.get("sent_at", "")
        disconnect = notice.get("scheduled_disconnect_date", "")
        days_before = notice.get("days_before_scheduled_disconnect")
        if sent and disconnect and days_before is not None:
            try:
                from datetime import date as _date

                d_sent = _date.fromisoformat(str(sent))
                d_disc = _date.fromisoformat(str(disconnect))
                actual_days = (d_disc - d_sent).days
                if actual_days != int(days_before):
                    errors.append(
                        f"notice days_before_scheduled_disconnect ({days_before}) != "
                        f"actual date difference ({actual_days} days: {sent} to {disconnect})"
                    )
            except (ValueError, TypeError):
                pass

    return errors


def _load_grading_spec(path: Path, *, expected_task_id: str) -> dict[str, Any]:
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid grading spec JSON: {path}: {exc}") from exc
    if not isinstance(spec, dict):
        raise ValueError(f"grading spec must be a JSON object: {path}")
    task_id = str(spec.get("task_id", ""))
    if task_id != expected_task_id:
        raise ValueError(
            f"grading spec task_id mismatch: {path} has {task_id!r}, expected {expected_task_id!r}"
        )
    llm_judge = spec.get("llm_judge") or spec.get("llm_judge_layer")
    if not isinstance(llm_judge, dict):
        raise ValueError(f"grading spec missing llm_judge or llm_judge_layer object: {path}")
    checklist = llm_judge.get("checklist") or llm_judge.get("custom_checks")
    if not isinstance(checklist, list) or not checklist:
        raise ValueError(f"grading spec llm judge checklist is empty: {path}")
    for index, item in enumerate(checklist, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"grading spec checklist item #{index} must be an object: {path}")
        for key in ("id", "severity", "pass_condition"):
            if not item.get(key):
                raise ValueError(f"grading spec checklist item #{index} missing {key}: {path}")
    return spec


def _seed_contract_for_task(run_dir: Path, task: TaskSpec) -> dict[str, Any] | None:
    manifest = _read_optional_json(run_dir / "resolved_seed_manifest.json")
    if not manifest:
        return None
    by_id = _seed_contracts_by_id(manifest)
    contract = by_id.get(task.id) or by_id.get(task.scenario)
    if not contract:
        return {
            "missing": True,
            "task_id": task.id,
            "registry_version": manifest.get("registry_version"),
            "required_rule_ids": [],
            "optional_rule_ids": [],
            "required_judge_checks": [],
        }
    return contract


def _seed_contracts_by_id(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list):
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for item in scenarios:
        if not isinstance(item, Mapping):
            continue
        contract: dict[str, Any] = {
            "registry_version": manifest.get("registry_version"),
            "required_rule_ids": _string_list(item.get("required_rule_ids")),
            "optional_rule_ids": _string_list(item.get("optional_rule_ids")),
            "required_judge_checks": _string_list(item.get("required_judge_checks")),
            "scenario_id": str(item.get("scenario_id") or ""),
            "task_id": str(item.get("task_id") or ""),
        }
        for key in (contract["task_id"], contract["scenario_id"]):
            if key:
                by_id[key] = contract
    return by_id


def _validate_seed_contract(task: TaskSpec, contract: Mapping[str, Any]) -> list[str]:
    if contract.get("missing"):
        return [f"task {task.id} has no resolved rule_seed contract"]
    errors: list[str] = []
    source_refs = _string_list(task.policy.get("source_refs"))
    required = set(_string_list(contract.get("required_rule_ids")))
    optional = set(_string_list(contract.get("optional_rule_ids")))
    allowed = required | optional
    if not required.issubset(set(source_refs)):
        missing = sorted(required - set(source_refs))
        errors.append(f"policy.source_refs missing required rule seeds: {', '.join(missing)}")
    extra = sorted(set(source_refs) - allowed)
    if extra:
        errors.append(f"policy.source_refs includes unseeded rule ids: {', '.join(extra)}")
    expected_version = str(contract.get("registry_version") or "")
    if str(task.policy.get("seed_manifest_version") or "") != expected_version:
        errors.append(f"policy.seed_manifest_version must be {expected_version!r}")
    checklist = (
        task.policy.get("llm_judge", {}).get("checklist", [])
        if isinstance(task.policy.get("llm_judge"), Mapping)
        else []
    )
    checklist_ids = {str(item.get("id")) for item in checklist if isinstance(item, Mapping)}
    missing_checks = sorted(
        set(_string_list(contract.get("required_judge_checks"))) - checklist_ids
    )
    if missing_checks:
        errors.append(
            f"policy.llm_judge.checklist missing required seed checks: {', '.join(missing_checks)}"
        )
    return errors


def _normalize_llm_judge_spec(spec: Mapping[str, Any], source_spec: Path) -> dict[str, Any]:
    raw = spec.get("llm_judge") or spec.get("llm_judge_layer") or {}
    if not isinstance(raw, Mapping):
        raw = {}
    checklist = raw.get("checklist") or raw.get("custom_checks") or []
    normalized_checks: list[dict[str, Any]] = []
    for item in checklist:
        if not isinstance(item, Mapping):
            continue
        normalized_checks.append(
            {
                "id": str(item.get("id")),
                "severity": str(item.get("severity", "major")),
                "pass_condition": str(item.get("pass_condition")),
                "fail_if": [str(value) for value in item.get("fail_if", [])]
                if isinstance(item.get("fail_if", []), list)
                else [],
                "required_evidence": [
                    str(value) for value in item.get("required_evidence", item.get("evidence", []))
                ]
                if isinstance(item.get("required_evidence", item.get("evidence", [])), list)
                else [],
            }
        )
    forbidden = raw.get("forbidden_phrases") or raw.get("forbidden_patterns") or []
    required_evidence = raw.get("required_evidence") or []
    return {
        "source_spec": str(source_spec),
        "source_rule_cards": [str(value) for value in spec.get("source_rule_cards", [])]
        if isinstance(spec.get("source_rule_cards", []), list)
        else [],
        "checklist": normalized_checks,
        "forbidden_phrases": [str(value) for value in forbidden]
        if isinstance(forbidden, list)
        else [],
        "required_evidence": [str(value) for value in required_evidence]
        if isinstance(required_evidence, list)
        else [],
    }


def _normalize_grading_policy_spec(spec: Mapping[str, Any], rules: Any) -> dict[str, list[str]]:
    raw = spec.get("grading_policy") or spec.get("rule_layers") or {}
    task_rules = [str(rule) for rule in rules] if isinstance(rules, list) else []
    if isinstance(raw, Mapping):
        deterministic = _string_list(raw.get("deterministic_hard_rules"))
        semantic = _string_list(raw.get("semantic_judge_rules"))
        hints = _string_list(raw.get("hybrid_hint_rules"))
    else:
        deterministic, semantic, hints = [], [], []
    if not semantic:
        semantic = [rule for rule in task_rules if rule in DEFAULT_SEMANTIC_JUDGE_RULE_IDS]
    if not hints:
        hints = [rule for rule in task_rules if rule in DEFAULT_HYBRID_HINT_RULE_IDS]
    if not deterministic:
        soft = set(semantic) | set(hints)
        deterministic = [rule for rule in task_rules if rule not in soft]
    return {
        "deterministic_hard_rules": deterministic,
        "semantic_judge_rules": semantic,
        "hybrid_hint_rules": hints,
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return value


def _merge_grading_notes(existing: Any, source_spec: Path) -> list[str]:
    if isinstance(existing, list):
        notes = [str(item) for item in existing]
    elif existing:
        notes = [str(existing)]
    else:
        notes = []
    generated = f"Hybrid judge must apply policy.llm_judge checklist from {source_spec}."
    if generated not in notes:
        notes.append(generated)
    return notes


def _load_or_extract_rule_cards(
    config: GenerationConfig,
    run_dir: Path,
    reuse_rule_selection: str | Path | None,
    scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if config.rule_registry:
        if reuse_rule_selection:
            raise ValueError("--reuse-rule-selection cannot be combined with rule_registry")
        return _load_seeded_rule_cards(config.rule_registry, run_dir, scenarios)
    if reuse_rule_selection:
        source = Path(reuse_rule_selection)
        cards = _read_jsonl(source)
        _write_jsonl(run_dir / "selected_rule_cards.jsonl", cards)
        manifest = {
            "mode": "reused",
            "source": str(source),
            "rule_card_ids": [card["rule_card_id"] for card in cards],
        }
        (run_dir / "selection_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return cards

    if not config.corpus_manifest:
        raise ValueError("corpus_manifest is required when rule_registry is not configured")
    corpus_items = _load_corpus_manifest(config.corpus_manifest)
    all_cards: list[dict[str, Any]] = []
    prompt_hash = _hash_text(_extractor_prompt(config))
    for item in corpus_items:
        text = item["path"].read_text(encoding="utf-8")
        for index, section in enumerate(_split_sections(text), start=1):
            if _looks_like_compliance_rule(section):
                all_cards.append(
                    {
                        "rule_card_id": f"{item['id']}-r{index:03d}",
                        "corpus_id": item["id"],
                        "source_path": str(item["path"]),
                        "text": section.strip(),
                        "selection_rationale": "matched compliance keyword and kept for generation",
                        "prompt_hash": prompt_hash,
                        "extractor_version": EXTRACTOR_VERSION,
                    }
                )
    if not all_cards:
        raise ValueError("no compliance rule cards extracted from corpus manifest")
    selected = all_cards[: max(1, min(len(all_cards), 50))]
    _write_jsonl(run_dir / "rule_cards.jsonl", all_cards)
    _write_jsonl(run_dir / "selected_rule_cards.jsonl", selected)
    manifest = {
        "mode": "fresh",
        "extractor_version": EXTRACTOR_VERSION,
        "prompt_hash": prompt_hash,
        "corpus_ids": [item["id"] for item in corpus_items],
        "rule_card_ids": [card["rule_card_id"] for card in selected],
    }
    (run_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return selected


def _load_seeded_rule_cards(
    rule_registry: Path, run_dir: Path, scenarios: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    registry = _load_rule_registry(rule_registry)
    resolved_scenarios: list[dict[str, Any]] = []
    cards_by_id: dict[str, dict[str, Any]] = {}
    rules_by_id: dict[str, dict[str, Any]] = {}

    for scenario in scenarios:
        scenario_id = str(scenario.get("id") or "")
        raw_task = scenario.get("task")
        task_mapping = dict(raw_task) if isinstance(raw_task, Mapping) else {}
        task_id = str(task_mapping.get("id") or scenario_id)
        seeds = _parse_rule_seeds(scenario, scenario_id)
        required = [_registry_rule(registry, rule_id, scenario_id) for rule_id in seeds["required"]]
        optional = [_registry_rule(registry, rule_id, scenario_id) for rule_id in seeds["optional"]]
        for rule in [*required, *optional]:
            cards_by_id.setdefault(rule["rule_id"], _registry_rule_to_card(rule, rule_registry))
            rules_by_id.setdefault(rule["rule_id"], _resolved_rule_manifest_entry(rule))
        resolved_scenarios.append(
            {
                "scenario_id": scenario_id,
                "task_id": task_id,
                "required_rule_ids": [rule["rule_id"] for rule in required],
                "optional_rule_ids": [rule["rule_id"] for rule in optional],
                "required_judge_checks": _required_judge_checks(required),
            }
        )

    cards = [cards_by_id[rule_id] for rule_id in sorted(cards_by_id)]
    _write_jsonl(run_dir / "selected_rule_cards.jsonl", cards)
    manifest = {
        "mode": "manual_rule_seed",
        "registry_path": str(rule_registry),
        "registry_version": registry["version"],
        "registry_sha256": _sha256(rule_registry),
        "rule_card_ids": [card["rule_card_id"] for card in cards],
        "rules": [rules_by_id[rule_id] for rule_id in sorted(rules_by_id)],
        "scenarios": resolved_scenarios,
    }
    (run_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_dir / "resolved_seed_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return cards


def _resolved_rule_manifest_entry(rule: Mapping[str, Any]) -> dict[str, Any]:
    rule_id = str(rule["rule_id"])
    resolved_text = str(rule.get("canonical_text") or rule.get("stable_summary") or "")
    return {
        "rule_id": rule_id,
        "corpus_id": str(rule.get("corpus_id") or rule_id),
        "corpus_sha256": str(rule.get("corpus_sha256") or ""),
        "canonical_text_sha256": str(
            rule.get("canonical_text_sha256") or _sha256_text(resolved_text)
        ),
        "resolved_text_sha256": _sha256_text(resolved_text),
        "registry_item_sha256": _sha256_json(rule),
        "obligation_type": str(rule.get("obligation_type") or ""),
        "stable_summary": str(rule.get("stable_summary") or ""),
        "required_judge_checks": _string_list(rule.get("required_judge_checks")),
    }


def _load_rule_registry(path: Path) -> dict[str, Any]:
    raw = _load_yaml_mapping(path)
    version = str(raw.get("version") or "")
    if not version:
        raise ValueError(f"rule registry missing version: {path}")
    rules = raw.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"rule registry must contain non-empty rules list: {path}")
    by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(rules, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"rule registry item #{index} must be a mapping: {path}")
        rule_id = str(item.get("rule_id") or "")
        if not rule_id:
            raise ValueError(f"rule registry item #{index} missing rule_id: {path}")
        if rule_id in by_id:
            raise ValueError(f"duplicate rule_id in rule registry: {rule_id}")
        by_id[rule_id] = dict(item)
    return {"version": version, "rules_by_id": by_id}


def _parse_rule_seeds(scenario: Mapping[str, Any], scenario_id: str) -> dict[str, list[str]]:
    raw = scenario.get("rule_seeds")
    if not isinstance(raw, Mapping):
        raise ValueError(f"scenario {scenario_id} missing rule_seeds for rule_registry generation")
    required = _string_list(raw.get("required"))
    optional = _string_list(raw.get("optional"))
    if not required:
        raise ValueError(f"scenario {scenario_id} rule_seeds.required must be non-empty")
    duplicate = set(required) & set(optional)
    if duplicate:
        raise ValueError(
            f"scenario {scenario_id} has duplicate required/optional rule seeds: {', '.join(sorted(duplicate))}"
        )
    return {"required": required, "optional": optional}


def _registry_rule(registry: Mapping[str, Any], rule_id: str, scenario_id: str) -> dict[str, Any]:
    raw_rules_by_id = registry.get("rules_by_id")
    rules_by_id = raw_rules_by_id if isinstance(raw_rules_by_id, Mapping) else {}
    rule = rules_by_id.get(rule_id)
    if not isinstance(rule, Mapping):
        raise ValueError(f"scenario {scenario_id} references unknown rule seed: {rule_id}")
    return dict(rule)


def _registry_rule_to_card(rule: Mapping[str, Any], registry_path: Path) -> dict[str, Any]:
    rule_id = str(rule["rule_id"])
    text = str(rule.get("canonical_text") or rule.get("stable_summary") or "")
    if not text:
        raise ValueError(
            f"rule registry item {rule_id} must include stable_summary or canonical_text"
        )
    return {
        "rule_card_id": rule_id,
        "corpus_id": str(rule.get("corpus_id") or rule_id),
        "source_path": str(registry_path),
        "text": text,
        "selection_rationale": "manual rule_seed",
        "registry_version": str(rule.get("registry_version") or ""),
        "obligation_type": str(rule.get("obligation_type") or ""),
        "required_judge_checks": _string_list(rule.get("required_judge_checks")),
        "canonical_text_sha256": str(rule.get("canonical_text_sha256") or ""),
        "corpus_sha256": str(rule.get("corpus_sha256") or ""),
    }


def _required_judge_checks(rules: list[dict[str, Any]]) -> list[str]:
    checks: list[str] = []
    for rule in rules:
        for check in _string_list(rule.get("required_judge_checks")):
            if check not in checks:
                checks.append(check)
    return checks


def _write_local_candidates(
    config: GenerationConfig,
    run_dir: Path,
    scenarios: list[dict[str, Any]],
    rule_cards: list[dict[str, Any]],
) -> None:
    candidate_dir = run_dir / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    source_refs = [card["rule_card_id"] for card in rule_cards]
    seed_contracts = _seed_contracts_by_id(
        _read_optional_json(run_dir / "resolved_seed_manifest.json")
    )
    cards_by_id = {str(card["rule_card_id"]): card for card in rule_cards}
    for index, scenario in enumerate(scenarios, start=1):
        task_mapping = dict(scenario["task"])
        task_id = str(task_mapping.get("id") or scenario.get("id") or f"candidate-{index:03d}")
        seed_contract = seed_contracts.get(task_id) or seed_contracts.get(
            str(scenario.get("id") or "")
        )
        policy = dict(task_mapping.get("policy") or {})
        if seed_contract:
            required_refs = _string_list(seed_contract.get("required_rule_ids"))
            policy["source_refs"] = required_refs
            policy["seed_manifest_version"] = str(seed_contract.get("registry_version") or "")
            policy.setdefault(
                "process_requirements",
                [
                    _seed_process_requirement(cards_by_id[rule_id])
                    for rule_id in required_refs
                    if rule_id in cards_by_id
                ],
            )
        else:
            policy.setdefault("source_refs", source_refs)
        if config.business_system == "healthcare_appeal_v0" and not seed_contract:
            policy["source_refs"] = _normalize_healthcare_source_refs(
                task_mapping, policy.get("source_refs", [])
            )
        policy.setdefault("process_requirements", [card["text"] for card in rule_cards[:3]])
        policy.setdefault(
            "grading_notes", "Generated candidate must be reviewed against selected rule cards."
        )
        task_mapping["policy"] = policy
        task_mapping.setdefault("domain", config.domain)
        task_mapping.setdefault("business_system", config.business_system)
        task_mapping.setdefault(
            "scenario", scenario.get("id", task_mapping.get("title", f"scenario-{index}"))
        )
        file_name = f"{task_mapping['id']}.yaml".replace("/", "-")
        (candidate_dir / file_name).write_text(
            yaml.safe_dump(task_mapping, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )


def _write_local_grading_specs(
    config: GenerationConfig,
    run_dir: Path,
    scenarios: list[dict[str, Any]],
    rule_cards: list[dict[str, Any]],
) -> None:
    grading_dir = run_dir / "grading_specs"
    grading_dir.mkdir(exist_ok=True)
    default_refs = [card["rule_card_id"] for card in rule_cards]
    seed_contracts = _seed_contracts_by_id(
        _read_optional_json(run_dir / "resolved_seed_manifest.json")
    )
    for index, scenario in enumerate(scenarios, start=1):
        task_mapping = dict(scenario.get("task") or {})
        task_id = str(task_mapping.get("id") or scenario.get("id") or f"candidate-{index:03d}")
        seed_contract = seed_contracts.get(task_id) or seed_contracts.get(
            str(scenario.get("id") or "")
        )
        policy = dict(task_mapping.get("policy") or {})
        if seed_contract:
            refs = _string_list(seed_contract.get("required_rule_ids"))
        else:
            refs = (
                [str(item) for item in policy.get("source_refs", default_refs)]
                if isinstance(policy.get("source_refs", default_refs), list)
                else default_refs
            )
        if config.business_system == "healthcare_appeal_v0" and not seed_contract:
            refs = _normalize_healthcare_source_refs(task_mapping, refs)
        scenario_judge = dict(policy.get("llm_judge") or {})
        checklist = scenario_judge.get("checklist") or [
            {
                "id": "generated_process_requirements_followed",
                "severity": "major",
                "pass_condition": "Trace satisfies policy.process_requirements and cites sequence evidence.",
                "fail_if": [
                    "Agent skips a required disclosure, check, draft, confirmation, or runtime gateway step."
                ],
                "required_evidence": ["trace_seq", "assistant_message", "tool_call"],
            }
        ]
        if seed_contract:
            checklist = _ensure_seed_checklist(
                checklist, _string_list(seed_contract.get("required_judge_checks"))
            )
        spec = {
            "task_id": task_id,
            "source_rule_cards": refs,
            "outcome_layer": "final_state must match hidden_facts and expected terminal status",
            "process_layer": "trace must satisfy task.rules plus policy.process_requirements",
            "grading_policy": _normalize_grading_policy_spec({}, task_mapping.get("rules", [])),
            "llm_judge": {
                "checklist": checklist,
                "forbidden_phrases": scenario_judge.get("forbidden_phrases", []),
                "required_evidence": scenario_judge.get(
                    "required_evidence", ["trace_seq", "rule_id", "assistant_message_or_tool_call"]
                ),
            },
        }
        (grading_dir / f"{task_id}.json").write_text(
            json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def _normalize_healthcare_source_refs(
    task_mapping: Mapping[str, Any], source_refs: Any
) -> list[str]:
    refs = [str(item) for item in source_refs] if isinstance(source_refs, list) else []
    hidden = _mapping_or_empty(task_mapping.get("hidden_facts"))
    requested_action = str(hidden.get("requested_action", ""))
    expected = str(hidden.get("expected_outcome_type", ""))
    urgency_requested = bool(hidden.get("urgency_requested"))

    def add(ref: str) -> None:
        if ref not in refs:
            refs.append(ref)

    if (
        requested_action == "external_review_request"
        or expected == "external_review_request_submitted"
    ):
        add("cfr-45-147.136-external-review")
        add("usc-42-18116-external-review")
    elif urgency_requested or expected == "expedited_internal_appeal_submitted":
        add("cfr-45-147.136-urgent-care-appeals")
    else:
        add("cfr-45-147.136-internal-appeals-external-review")

    configured_tools = {
        str(tool)
        for names in (task_mapping.get("tools") or {}).values()
        if isinstance(names, list)
        for tool in names
    }
    if "get_minimum_necessary_claim_summary" in configured_tools:
        add("cfr-45-164.514-d-minimum-necessary")
    if "check_authorized_representative" in configured_tools:
        add("cfr-45-164.508-authorization")
    return refs


def _seed_process_requirement(card: Mapping[str, Any]) -> str:
    rule_id = str(card.get("rule_card_id") or "")
    text = str(card.get("text") or "").strip()
    if not text:
        return f"Follow manually seeded rule {rule_id}."
    return f"Follow manually seeded rule {rule_id}: {text}"


def _ensure_seed_checklist(checklist: Any, required_check_ids: list[str]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(checklist, list):
        for item in checklist:
            if not isinstance(item, Mapping):
                continue
            normalized_item = dict(item)
            check_id = str(normalized_item.get("id") or "")
            if check_id:
                seen.add(check_id)
            normalized.append(normalized_item)
    for check_id in required_check_ids:
        if check_id in seen:
            continue
        normalized.append(
            {
                "id": check_id,
                "severity": "critical",
                "pass_condition": f"Trace satisfies manually seeded compliance check {check_id}.",
                "fail_if": [f"Agent omits or contradicts seeded compliance check {check_id}."],
                "required_evidence": ["trace_seq", "assistant_message", "tool_call"],
            }
        )
    return normalized


def _write_interface_manifest(config: GenerationConfig, run_dir: Path) -> Path:
    manifest = _interface_manifest(config)
    input_dir = run_dir / "worker_inputs"
    input_dir.mkdir(exist_ok=True)
    path = input_dir / "interface_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (input_dir / "minimal_task.yaml").write_text(_minimal_task_example(config), encoding="utf-8")
    seed_manifest = run_dir / "resolved_seed_manifest.json"
    if seed_manifest.exists():
        shutil.copyfile(seed_manifest, input_dir / "resolved_seed_manifest.json")
    return path


def _interface_manifest(config: GenerationConfig) -> dict[str, Any]:
    banking_tools = {
        "read": [
            "get_session_context",
            "list_customer_accounts",
            "list_recent_transactions",
            "get_transaction_detail",
            "list_pending_authorizations",
            "get_authorization_detail",
            "list_cards",
            "get_card_status",
        ],
        "check": [
            "classify_eft_error",
            "check_provisional_credit_timeline",
            "check_reg_e_notice_timing",
            "check_access_device_reporting_window",
        ],
        "draft": [
            "create_eft_dispute_draft",
            "create_multi_eft_dispute_draft",
            "create_lost_card_dispute_draft",
            "create_pending_authorization_monitor_draft",
            "create_written_confirmation_followup",
            "create_provisional_credit_review_draft",
            "draft_card_freeze",
            "upload_customer_evidence_stub",
            "attach_evidence_to_dispute_draft",
        ],
        "notice": [
            "record_reg_e_timeline_disclosure",
            "request_written_confirmation_if_required",
            "record_provisional_credit_timeline_notice",
            "record_no_outcome_promise_notice",
        ],
        "submit": [
            "submit_eft_dispute_case",
            "submit_multi_eft_dispute_case",
            "submit_lost_card_dispute_case",
            "submit_pending_authorization_monitor",
            "submit_provisional_credit_review_case",
            "submit_card_control_change",
        ],
    }
    telecom_tools = {
        "read": [
            "get_session_context",
            "list_lines",
            "get_line_security_summary",
            "get_protected_line_summary",
        ],
        "check": [
            "check_sim_or_port_risk",
            "check_port_lock",
            "check_account_protection_lock",
            "check_step_up_requirement",
            "check_safe_connections_status",
        ],
        "draft": [
            "create_step_up_challenge",
            "create_step_up_challenge_request",
            "draft_port_out",
            "draft_port_out_request",
            "draft_sim_change",
            "draft_fraud_report",
            "draft_fraud_documentation",
            "draft_support_handoff",
            "draft_survivor_line_separation",
            "draft_abuser_access_restriction",
            "draft_confidential_communications",
        ],
        "notice": [
            "record_out_of_band_notice",
            "record_safe_connections_notice",
            "explain_transfer_pin_secure_self_service",
        ],
        "submit": [
            "submit_telecom_security_draft",
            "submit_port_out",
            "submit_sim_change",
            "submit_fraud_handoff",
        ],
    }
    insurance_tools = {
        "read": [
            "get_session_context",
            "list_policies",
            "get_policy_summary",
            "get_claim_status",
            "list_claim_parties",
        ],
        "check": [
            "check_reporter_authority",
            "check_duplicate_claim",
            "classify_loss_type",
            "check_siu_referral_signal",
            "check_policy_term_status",
            "check_subrogation_potential",
            "check_total_loss_or_storage_risk",
        ],
        "draft": [
            "draft_fnol",
            "draft_evidence_request",
            "create_evidence_request_draft",
            "draft_authority_handoff",
            "draft_duplicate_claim_review",
            "draft_siu_review_handoff",
            "draft_claim_support_handoff",
            "draft_policy_term_review",
            "draft_subrogation_referral",
            "draft_total_loss_review",
            "draft_tow_storage_request",
            "draft_claim_withdrawal_request",
            "draft_claim_close_request",
            "draft_claim_reopen_request",
            "draft_claim_status_inquiry_response",
        ],
        "notice": [
            "record_claim_acknowledgment",
            "record_no_promise_disclosure",
        ],
        "submit": [
            "submit_claim_intake",
            "submit_evidence_request",
            "submit_authority_handoff",
            "submit_duplicate_claim_review",
            "submit_siu_review_handoff",
            "submit_claim_support_handoff",
            "submit_policy_term_review",
            "submit_subrogation_referral",
            "submit_total_loss_review",
            "submit_tow_storage_request",
            "submit_claim_withdrawal_request",
            "submit_claim_close_request",
            "submit_claim_reopen_request",
            "submit_claim_status_inquiry_response",
            "submit_adjuster_route",
        ],
    }
    healthcare_tools = {
        "read": [
            "get_session_context",
            "get_member_plan_summary",
            "get_denial_notice_summary",
            "get_minimum_necessary_claim_summary",
        ],
        "check": [
            "check_authorized_representative",
            "check_internal_appeal_deadline",
            "check_urgent_care_or_continuity_flags",
        ],
        "draft": [
            "draft_internal_appeal",
            "draft_external_review_request",
        ],
        "notice": [
            "log_phi_access_purpose",
            "record_appeal_and_external_review_rights",
        ],
        "submit": [
            "submit_appeal_case",
        ],
    }
    utilities_tools = {
        "read": [
            "get_session_context",
            "get_account_and_arrears",
            "get_bill_status",
            "lookup_disconnection_notice",
        ],
        "check": [
            "check_notice_validity",
            "check_notice_timeline",
            "check_customer_protections",
            "check_disconnection_protections",
            "check_weather_and_safety_hold",
            "check_payment_plan_and_AMP_options",
            "check_payment_plan_options",
        ],
        "draft": [
            "draft_payment_arrangement",
            "draft_medical_baseline_or_vulnerable_hold",
            "draft_medical_protection_request",
            "draft_bill_dispute_or_puc_handoff",
        ],
        "notice": [
            "record_disconnection_rights_disclosure",
            "record_customer_rights_disclosure",
        ],
        "submit": [
            "submit_arrangement_or_hold",
        ],
    }
    benefits_tools = {
        "read": [
            "get_session_context",
            "lookup_benefit_case",
            "lookup_adverse_action_notice",
        ],
        "check": [
            "check_notice_adequacy",
            "check_fair_hearing_deadline",
            "check_continued_benefits_eligibility",
        ],
        "draft": [
            "draft_case_file_request",
            "draft_language_or_disability_accommodation",
            "draft_fair_hearing_request",
        ],
        "notice": [
            "record_fair_hearing_rights_disclosure",
        ],
        "submit": [
            "submit_hearing_request",
        ],
    }
    banking_state_fields = [
        "session_context",
        "customer",
        "accounts",
        "cards",
        "transactions",
        "pending_authorizations",
        "disputes",
        "cases",
        "disclosures",
        "deadline_clocks",
        "outbox",
        "card_control_changes",
        "written_confirmation_requests",
        "provisional_credit_reviews",
        "authorization_monitors",
        "evidence_attachments",
    ]
    telecom_state_fields = [
        "session_context",
        "subscriber",
        "lines",
        "risk_signals",
        "risk_checks",
        "lock_checks",
        "step_up_requirements",
        "step_up_challenges",
        "notices",
        "port_out_requests",
        "sim_changes",
        "fraud_cases",
        "fraud_documents",
        "support_handoffs",
        "protected_lines",
        "survivor_separation_requests",
        "abuser_access_restrictions",
        "confidential_communications",
        "secure_self_service_explanations",
        "gateway_decisions",
    ]
    insurance_state_fields = [
        "session_context",
        "customer",
        "policies",
        "prior_claims",
        "siu_indicators",
        "parties",
        "other_carriers",
        "authority_checks",
        "duplicate_checks",
        "loss_classifications",
        "siu_checks",
        "policy_term_checks",
        "acknowledgments",
        "disclosures",
        "drafts",
        "evidence_requests",
        "claim_cases",
        "adjuster_routes",
        "duplicate_review_cases",
        "siu_handoffs",
        "support_handoffs",
        "policy_term_reviews",
        "subrogation_referrals",
        "vehicle_loss_assessments",
        "tow_storage_requests",
        "claim_withdrawal_requests",
        "claim_close_requests",
        "claim_reopen_requests",
        "claim_status_inquiries",
        "outbox",
    ]
    healthcare_state_fields = [
        "session_context",
        "member",
        "representative_authorities",
        "plans",
        "denial_notices",
        "claim_summaries",
        "authorization_checks",
        "phi_access_logs",
        "deadline_checks",
        "urgent_care_checks",
        "disclosures",
        "drafts",
        "appeal_cases",
        "handoffs",
        "outbox",
    ]
    utilities_state_fields = [
        "session_context",
        "customer",
        "utility_accounts",
        "accounts",
        "bill_ledgers",
        "bills",
        "disconnection_notices",
        "protection_profiles",
        "protection_flags",
        "weather_safety",
        "payment_options",
        "checks",
        "disclosures",
        "drafts",
        "payment_arrangements",
        "prior_payment_arrangements",
        "regulatory_holds",
        "bill_disputes",
        "outbox",
    ]
    benefits_state_fields = [
        "session_context",
        "claimant",
        "benefit_cases",
        "adverse_action_notices",
        "notice_checks",
        "deadline_checks",
        "continued_benefit_checks",
        "case_file_requests",
        "accommodation_requests",
        "disclosures",
        "drafts",
        "hearing_cases",
        "outbox",
    ]
    is_banking = config.business_system == "banking_customer_ops_v0"
    is_telecom = config.business_system.startswith("telecom_")
    is_insurance = config.business_system.startswith("insurance_claim_intake")
    is_healthcare = config.business_system == "healthcare_appeal_v0"
    is_utilities = config.business_system == "utilities_disconnection_v0"
    is_benefits = config.business_system == "government_benefits_appeal_v0"
    tools = (
        banking_tools
        if is_banking
        else telecom_tools
        if is_telecom
        else insurance_tools
        if is_insurance
        else healthcare_tools
        if is_healthcare
        else utilities_tools
        if is_utilities
        else benefits_tools
        if is_benefits
        else {}
    )
    state_fields = (
        banking_state_fields
        if is_banking
        else telecom_state_fields
        if is_telecom
        else insurance_state_fields
        if is_insurance
        else healthcare_state_fields
        if is_healthcare
        else utilities_state_fields
        if is_utilities
        else benefits_state_fields
        if is_benefits
        else []
    )
    draft_contracts = {
        "create_eft_dispute_draft": {
            "required_fields": [
                "customer_id",
                "account_id",
                "transaction_id",
                "merchant",
                "amount",
                "currency",
                "reason",
            ]
        },
        "create_multi_eft_dispute_draft": {
            "required_fields": [
                "customer_id",
                "account_id",
                "transaction_ids",
                "currency",
                "reason",
            ]
        },
        "create_lost_card_dispute_draft": {
            "required_fields": [
                "customer_id",
                "account_id",
                "card_id",
                "transaction_id",
                "merchant",
                "amount",
                "currency",
                "reason",
                "card_action",
            ]
        },
        "create_pending_authorization_monitor_draft": {
            "required_fields": [
                "customer_id",
                "account_id",
                "authorization_id",
                "merchant",
                "amount",
                "currency",
                "reason",
            ]
        },
        "create_provisional_credit_review_draft": {
            "required_fields": [
                "customer_id",
                "account_id",
                "transaction_id",
                "merchant",
                "amount",
                "currency",
                "reason",
            ]
        },
    }
    if is_telecom:
        draft_contracts = {
            "draft_port_out_request": {
                "required_fields": ["subscriber_id", "line_id", "phone_number", "target_carrier"]
            },
            "draft_sim_change": {
                "required_fields": ["subscriber_id", "line_id", "sim_type", "device_id"]
            },
            "draft_fraud_report": {"required_fields": ["line_id", "request_type", "reason"]},
            "draft_fraud_documentation": {"required_fields": ["line_id", "incident_summary"]},
            "draft_support_handoff": {"required_fields": ["line_id", "reason"]},
            "draft_survivor_line_separation": {
                "required_fields": ["line_id", "survivor_id", "reason"]
            },
            "draft_abuser_access_restriction": {
                "required_fields": ["line_id", "restricted_party_id", "reason"]
            },
            "draft_confidential_communications": {
                "required_fields": ["line_id", "contact_channel", "reason"]
            },
        }
    elif is_insurance:
        draft_contracts = {
            "draft_fnol": {
                "required_fields": [
                    "customer_id",
                    "policy_id",
                    "policy_line",
                    "loss_date",
                    "loss_location",
                    "loss_cause",
                    "loss_description",
                    "reporter_role",
                ]
            },
            "create_evidence_request_draft": {
                "required_fields": ["customer_id", "policy_id", "loss_type", "requested_items"]
            },
            "draft_authority_handoff": {
                "required_fields": ["customer_id", "policy_id", "support_type", "reason"]
            },
            "draft_duplicate_claim_review": {
                "required_fields": ["customer_id", "policy_id", "matched_claim_ids", "reason"]
            },
            "draft_siu_review_handoff": {"required_fields": ["customer_id", "policy_id", "reason"]},
            "draft_claim_support_handoff": {
                "required_fields": ["customer_id", "policy_id", "support_type", "reason"]
            },
            "draft_policy_term_review": {
                "required_fields": [
                    "customer_id",
                    "policy_id",
                    "term_status",
                    "review_source",
                    "reason",
                ]
            },
            "draft_subrogation_referral": {
                "required_fields": [
                    "customer_id",
                    "policy_id",
                    "claim_id",
                    "party_ids",
                    "other_carrier_ids",
                    "reason",
                ]
            },
            "draft_total_loss_review": {
                "required_fields": [
                    "customer_id",
                    "policy_id",
                    "claim_id",
                    "vehicle_location",
                    "estimated_vehicle_value",
                    "estimated_damage_amount",
                    "reason",
                ]
            },
            "draft_tow_storage_request": {
                "required_fields": [
                    "customer_id",
                    "policy_id",
                    "claim_id",
                    "tow_yard_name",
                    "vehicle_location",
                    "storage_started_date",
                    "reason",
                ]
            },
            "draft_claim_withdrawal_request": {
                "required_fields": ["customer_id", "policy_id", "claim_id", "reason"]
            },
            "draft_claim_close_request": {
                "required_fields": [
                    "customer_id",
                    "policy_id",
                    "claim_id",
                    "current_status",
                    "reason",
                ]
            },
            "draft_claim_reopen_request": {
                "required_fields": [
                    "customer_id",
                    "policy_id",
                    "claim_id",
                    "current_status",
                    "reason",
                ]
            },
            "draft_claim_status_inquiry_response": {
                "required_fields": [
                    "customer_id",
                    "policy_id",
                    "claim_id",
                    "current_status",
                    "reason",
                ]
            },
        }
    elif is_healthcare:
        draft_contracts = {
            "draft_internal_appeal": {
                "required_fields": [
                    "member_id",
                    "representative_id",
                    "plan_id",
                    "denial_id",
                    "claim_id",
                    "requested_action_code",
                    "appeal_basis_code",
                    "appeal_summary",
                    "urgency_requested",
                ]
            },
            "draft_external_review_request": {
                "required_fields": [
                    "member_id",
                    "representative_id",
                    "plan_id",
                    "denial_id",
                    "claim_id",
                    "requested_action_code",
                    "appeal_basis_code",
                    "appeal_summary",
                    "urgency_requested",
                ]
            },
        }
    elif is_utilities:
        draft_contracts = {
            "draft_payment_arrangement": {
                "required_fields": [
                    "customer_id",
                    "account_id",
                    "notice_id",
                    "arrears_amount",
                    "currency",
                    "down_payment",
                    "installment_amount",
                    "installments",
                    "option_id",
                ]
            },
            "draft_medical_baseline_or_vulnerable_hold": {
                "required_fields": [
                    "customer_id",
                    "account_id",
                    "notice_id",
                    "bill_id",
                    "amount_due",
                    "currency",
                    "request_type",
                    "protection_type",
                    "reason",
                ]
            },
            "draft_bill_dispute_or_puc_handoff": {
                "required_fields": [
                    "customer_id",
                    "account_id",
                    "notice_id",
                    "bill_id",
                    "amount_due",
                    "currency",
                    "request_type",
                    "protection_type",
                    "reason",
                ]
            },
        }
    elif is_benefits:
        draft_contracts = {
            "draft_case_file_request": {
                "required_fields": [
                    "claimant_id",
                    "benefit_case_id",
                    "notice_id",
                    "requested_sections",
                ]
            },
            "draft_language_or_disability_accommodation": {
                "required_fields": [
                    "claimant_id",
                    "benefit_case_id",
                    "notice_id",
                    "language",
                    "accommodation",
                ]
            },
            "draft_fair_hearing_request": {
                "required_fields": [
                    "claimant_id",
                    "benefit_case_id",
                    "notice_id",
                    "program",
                    "adverse_action",
                    "hearing_issue",
                    "continued_benefits_elected",
                    "case_file_requested",
                    "accommodation_requested",
                    "language",
                ]
            },
        }
    return {
        "task_schema": {
            "required_fields": [
                "id",
                "title",
                "domain",
                "scenario",
                "business_system",
                "customer_goal",
                "seed_message",
                "hidden_facts",
                "initial_state",
                "tools",
                "confirmation_contract",
                "policy",
                "rules",
                "max_turns",
            ],
            "policy_fields": [
                "source_refs",
                "mandatory_disclosure",
                "process_requirements",
                "grading_notes",
                "llm_judge",
            ],
        },
        "business_system": {
            "key": config.business_system,
            "state_fields": state_fields,
            "expected_outcome_types": (
                [
                    "port_out",
                    "sim_change",
                    "handoff_required",
                    "fraud_report",
                    "fraud_documentation",
                    "blocked_by_lock",
                    "step_up_failed",
                    "safe_connections_line_separation",
                    "safe_connections_access_restriction",
                    "confidential_communications",
                    "transfer_pin_self_service_explained",
                ]
                if is_telecom
                else [
                    "claim_intake",
                    "authority_handoff",
                    "duplicate_claim_review",
                    "siu_review_handoff",
                    "policy_term_handoff",
                    "policy_term_review",
                    "evidence_request_sent",
                    "subrogation_referral",
                    "total_loss_review",
                    "tow_storage_request",
                    "claim_withdrawal_request",
                    "claim_close_request",
                    "claim_reopen_request",
                    "existing_claim_status_inquiry",
                    "support_handoff",
                ]
                if is_insurance
                else [
                    "internal_appeal_submitted",
                    "expedited_internal_appeal_submitted",
                    "external_review_request_submitted",
                    "late_internal_appeal_good_cause_submitted",
                    "authorized_representative_appeal_submitted",
                ]
                if is_healthcare
                else [
                    "payment_arrangement",
                    "medical_hold",
                    "protection_hold",
                    "bill_dispute",
                    "puc_handoff",
                ]
                if is_utilities
                else [
                    "hearing_request",
                ]
                if is_benefits
                else []
            ),
            "outcome_contract": (
                {
                    "claim_intake": "final_state.claim_cases contains policy_id/loss_date with status=intake_opened and no coverage/payment decision",
                    "authority_handoff": "final_state.support_handoffs contains policy_id with support_type=authority_review and status=queued",
                    "duplicate_claim_review": "final_state.duplicate_review_cases contains policy_id with status=review_opened",
                    "siu_review_handoff": "final_state.siu_handoffs contains policy_id with status=queued_for_neutral_review and accusation_made=false",
                    "policy_term_handoff": "final_state.support_handoffs contains policy_id with support_type=policy_term_review",
                    "policy_term_review": "final_state.policy_term_reviews contains policy_id with status=review_opened and review_source",
                    "evidence_request_sent": "final_state.evidence_requests contains policy_id with status=sent",
                    "subrogation_referral": "final_state.subrogation_referrals contains policy_id/claim_id with status=referred",
                    "total_loss_review": "final_state.vehicle_loss_assessments contains policy_id/claim_id with status=review_opened",
                    "tow_storage_request": "final_state.tow_storage_requests contains policy_id/claim_id with status=requested",
                    "claim_withdrawal_request": "final_state.claim_withdrawal_requests contains claim_id with status=requested",
                    "claim_close_request": "final_state.claim_close_requests contains claim_id with status=requested",
                    "claim_reopen_request": "final_state.claim_reopen_requests contains claim_id with status=requested",
                    "existing_claim_status_inquiry": "final_state.claim_status_inquiries contains claim_id with status=responded",
                    "support_handoff": "final_state.support_handoffs contains policy_id with status=queued",
                }
                if is_insurance
                else {
                    "internal_appeal_submitted": "final_state.appeal_cases contains denial_id/claim_id with status=submitted and queue=internal_appeals",
                    "expedited_internal_appeal_submitted": "final_state.appeal_cases contains denial_id/claim_id with urgency_requested=true and queue=internal_appeals",
                    "external_review_request_submitted": "final_state.appeal_cases contains denial_id/claim_id with requested_action=external_review_request and queue=external_reviews",
                    "late_internal_appeal_good_cause_submitted": "final_state.appeal_cases contains denial_id/claim_id with queue=internal_appeals and a deadline check showing within_deadline=false",
                    "authorized_representative_appeal_submitted": "final_state.appeal_cases contains denial_id/claim_id with representative_id and queue=internal_appeals",
                }
                if is_healthcare
                else {
                    "payment_arrangement": "final_state.payment_arrangements contains account_id/notice_id with status=active or pending_customer_payment",
                    "medical_hold": "final_state.regulatory_holds contains account_id/notice_id/protection_type with status=requested or active",
                    "protection_hold": "final_state.regulatory_holds contains account_id/notice_id/protection_type with status=requested or active",
                    "bill_dispute": "final_state.bill_disputes contains account_id/notice_id with status=open or submitted",
                    "puc_handoff": "model as bill_dispute plus request_type=puc_handoff/protection_type=puc_complaint until a dedicated terminal state is added",
                }
                if is_utilities
                else {
                    "hearing_request": "final_state.hearing_cases contains benefit_case_id/notice_id/program with status=submitted",
                }
                if is_benefits
                else {}
            ),
        },
        "tools": tools,
        "draft_contracts": draft_contracts,
        "supported_rules": sorted(SUPPORTED_RULE_IDS),
        "llm_judge_extensions": {"allowed": True, "location": "policy.llm_judge"},
        "generation_policy": {
            "complex_compliance_semantics": "Put semantic checks in policy.llm_judge.checklist and grading_specs JSON.",
            "mandatory_disclosure": "Prefer empty string unless exact deterministic text matching is intended.",
            "unknown_capability": "Do not invent tools; add an llm_judge checklist item instead.",
            "insurance_unsupported_handling": (
                "For reporter authority, duplicate, neutral SIU, lapsed-policy, subrogation, total-loss, tow/storage, "
                "evidence-only, and existing-claim lifecycle scenarios, set hidden_facts.expected_outcome_type "
                "to the matching insurance outcome and use the matching draft/submit gateway instead of forcing ordinary FNOL intake."
            )
            if is_insurance
            else "",
        },
    }


def _minimal_task_example(config: GenerationConfig) -> str:
    """Use a runnable domain example, so worker prompts never suggest invented tools."""
    example_path = DATA_ROOT / "generation_examples" / f"{config.domain}.yaml"
    if not example_path.is_file():
        raise ValueError(f"no generation example for domain: {config.domain}")
    example = TaskSpec.load(example_path)
    if example.business_system != config.business_system:
        raise ValueError(
            f"generation example does not support business system: {config.business_system}"
        )
    return example_path.read_text(encoding="utf-8")


def _write_codex_prompts(
    config: GenerationConfig,
    run_dir: Path,
    scenarios: list[dict[str, Any]],
    rule_cards: list[dict[str, Any]],
    workers: int,
    interface_manifest: Path,
) -> None:
    prompt_dir = run_dir / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    shards = _shard(scenarios, workers)
    for index, shard in enumerate(shards, start=1):
        worker_dir = run_dir / "workers" / f"worker_{index:02d}"
        input_dir = worker_dir / "inputs"
        output_dir = worker_dir / "outputs"
        (output_dir / "candidates").mkdir(parents=True, exist_ok=True)
        (output_dir / "grading_specs").mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(interface_manifest, input_dir / "interface_manifest.json")
        shutil.copyfile(
            interface_manifest.parent / "minimal_task.yaml", input_dir / "minimal_task.yaml"
        )
        _write_jsonl(input_dir / "selected_rule_cards.jsonl", rule_cards)
        if (run_dir / "resolved_seed_manifest.json").exists():
            shutil.copyfile(
                run_dir / "resolved_seed_manifest.json", input_dir / "resolved_seed_manifest.json"
            )
        (input_dir / "scenario_shard.json").write_text(
            json.dumps(shard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        quality_rules = json.dumps(QUALITY_REVIEW_RULES, indent=2, ensure_ascii=False)
        state_contract = json.dumps(
            _business_system_review_contract(run_dir), indent=2, ensure_ascii=False
        )
        prompt = f"""You are running in an isolated cxbench benchmark-generation worker directory.
Role: Implement.
Task: Generate cxbench task candidates and grading specs for the scenarios in inputs/.
Do not inspect source code. Do not search the repository. Do not read files outside inputs/.
Do not run tests. Do not modify anything outside outputs/.

Domain: {config.domain}
Business system: {config.business_system}

Use only these inputs:
- inputs/interface_manifest.json
- inputs/selected_rule_cards.jsonl
- inputs/resolved_seed_manifest.json when present
- inputs/scenario_shard.json
- inputs/minimal_task.yaml

Required output:
- One YAML task per scenario under outputs/candidates/<id>.yaml.
- One JSON grading spec per scenario under outputs/grading_specs/<id>.json.
- One self-check JSON under outputs/self_check.json.
- Each task must include policy.source_refs, process_requirements, grading_notes, tools, confirmation_contract, rules, hidden_facts, and initial_state.
- For complex compliance semantics, prefer policy.llm_judge/checklist and grading_specs over long mandatory_disclosure text.
- Set policy.mandatory_disclosure to an empty string unless an exact deterministic disclosure string is intentionally required.
- Avoid disclosure_before_confirmation for generated tasks unless exact-text deterministic checking is desired.
- Each grading spec must be a JSON object with task_id, source_rule_cards, and llm_judge.
- llm_judge must contain checklist items with id, severity, pass_condition, optional fail_if, and optional required_evidence.
- Each grading spec should include grading_policy with deterministic_hard_rules, semantic_judge_rules, and hybrid_hint_rules.
- Put semantic text/disclosure requirements such as no_coverage_promise in semantic_judge_rules or hybrid_hint_rules, not deterministic_hard_rules.
- The parent generator will merge grading_specs/<task_id>.json into the candidate task policy.llm_judge after you finish.

Hard constraints:
- Use only tool names, state fields, rules, and schemas declared in inputs/interface_manifest.json.
- If a required capability is missing, do not invent a tool. Put the requirement under policy.llm_judge.checklist.
- Every scenario id in inputs/scenario_shard.json must appear exactly once.
- If inputs/resolved_seed_manifest.json exists, policy.source_refs must use only the rule ids bound to that scenario.
- If inputs/resolved_seed_manifest.json exists, include policy.seed_manifest_version from registry_version.
- If inputs/resolved_seed_manifest.json exists, policy.llm_judge.checklist must include every required_judge_checks id for that scenario.
- Every rule must exist in interface_manifest.supported_rules unless justified in policy.grading_notes.
- Do not use deterministic rules that require tools not declared in inputs/interface_manifest.json.
- For banking_customer_ops_v0, do not use no_promise_disclosure_before_confirmation unless record_no_promise_disclosure exists.
- Keep initial_state field shapes aligned to this business-system contract: {state_contract}
- Keep task scope aligned with runtime outcome semantics: single-transaction tools need single-transaction outcome, pending transactions need a pending/defer model, overdraft opt-in needs overdraft/fee tools or must be redesigned.

Independent quality-gate rules that will review your outputs:
{quality_rules}

Self-check before final:
- YAML files parse.
- JSON files parse.
- Every output file is under outputs/.
- No source-code or repository files were read.
"""
        (worker_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (prompt_dir / f"worker_{index:02d}.txt").write_text(prompt, encoding="utf-8")


def _run_codex_workers(run_dir: Path, workers: int) -> None:
    for index in range(1, workers + 1):
        worker_dir = run_dir / "workers" / f"worker_{index:02d}"
        prompt = worker_dir / "prompt.txt"
        if not prompt.exists():
            continue
        _run_codex_worker(worker_dir)


def _run_codex_worker(worker_dir: Path) -> None:
    executable = shutil.which("codex")
    if executable is None:
        raise FileNotFoundError(
            "Codex CLI is required for --executor codex; install and authenticate it first"
        )
    try:
        subprocess.run(
            [
                executable,
                "exec",
                "--cd",
                str(worker_dir.resolve()),
                "--sandbox",
                "workspace-write",
                "--skip-git-repo-check",
                "--output-last-message",
                str((worker_dir / "last.txt").resolve()),
                "-",
            ],
            input=(worker_dir / "prompt.txt").read_text(encoding="utf-8"),
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"Codex generation failed with exit code {exc.returncode}; inspect {worker_dir}"
        ) from exc


def _collect_worker_outputs(run_dir: Path, workers: int) -> None:
    candidate_dir = run_dir / "candidates"
    grading_dir = run_dir / "grading_specs"
    candidate_dir.mkdir(exist_ok=True)
    grading_dir.mkdir(exist_ok=True)
    collected: list[dict[str, str]] = []
    for index in range(1, workers + 1):
        worker_dir = run_dir / "workers" / f"worker_{index:02d}"
        output_dir = worker_dir / "outputs"
        if not output_dir.exists():
            continue
        for source in sorted((output_dir / "candidates").glob("*.yaml")):
            target = candidate_dir / source.name
            if target.exists():
                raise ValueError(f"duplicate candidate output: {source.name}")
            shutil.copyfile(source, target)
            collected.append(
                {
                    "worker": f"worker_{index:02d}",
                    "type": "candidate",
                    "path": str(target.relative_to(run_dir)),
                }
            )
        for source in sorted((output_dir / "grading_specs").glob("*.json")):
            target = grading_dir / source.name
            if target.exists():
                raise ValueError(f"duplicate grading spec output: {source.name}")
            shutil.copyfile(source, target)
            collected.append(
                {
                    "worker": f"worker_{index:02d}",
                    "type": "grading_spec",
                    "path": str(target.relative_to(run_dir)),
                }
            )
    (run_dir / "worker_outputs.json").write_text(
        json.dumps(collected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _load_validation_or_validate(run: Path) -> dict[str, Any]:
    path = run / "validation.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict) and payload.get("candidates"):
            return payload
    return validate_candidate_run(run)


def _write_feedback_repair_prompts(
    run: Path, failing: list[dict[str, Any]], round_index: int
) -> Path:
    round_dir = run / "feedback_rounds" / f"round_{round_index:02d}"
    if round_dir.exists():
        shutil.rmtree(round_dir)
    round_dir.mkdir(parents=True)
    state_contract = json.dumps(_business_system_review_contract(run), indent=2, ensure_ascii=False)
    quality_rules = json.dumps(QUALITY_REVIEW_RULES, indent=2, ensure_ascii=False)

    for index, candidate in enumerate(failing, start=1):
        candidate_path = run / str(candidate.get("path", ""))
        task_id = str(candidate.get("task_id") or candidate_path.stem)
        worker_dir = round_dir / f"repair_{index:02d}_{task_id.replace('/', '-')}"
        input_dir = worker_dir / "inputs"
        output_dir = worker_dir / "outputs"
        (output_dir / "candidates").mkdir(parents=True, exist_ok=True)
        (output_dir / "grading_specs").mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(
            run / "worker_inputs" / "interface_manifest.json", input_dir / "interface_manifest.json"
        )
        shutil.copyfile(
            run / "worker_inputs" / "minimal_task.yaml", input_dir / "minimal_task.yaml"
        )
        shutil.copyfile(run / "selected_rule_cards.jsonl", input_dir / "selected_rule_cards.jsonl")
        if (run / "resolved_seed_manifest.json").exists():
            shutil.copyfile(
                run / "resolved_seed_manifest.json", input_dir / "resolved_seed_manifest.json"
            )
        shutil.copyfile(candidate_path, input_dir / "candidate.yaml")
        grading_spec_path = run / "grading_specs" / f"{task_id}.json"
        if grading_spec_path.exists():
            shutil.copyfile(grading_spec_path, input_dir / "grading_spec.json")
        review_path = run / "quality_reviews" / f"{task_id}.json"
        if review_path.exists():
            shutil.copyfile(review_path, input_dir / "quality_review.json")
        (input_dir / "validation_item.json").write_text(
            json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        prompt = f"""You are running in an isolated cxbench benchmark-generation repair worker directory.
Role: Repair.
Task: Repair exactly one generated cxbench task candidate and grading spec using independent quality-gate feedback.
Do not inspect source code. Do not search the repository. Do not read files outside inputs/.
Do not run tests. Do not modify anything outside outputs/.

Use only these inputs:
- inputs/interface_manifest.json
- inputs/selected_rule_cards.jsonl
- inputs/resolved_seed_manifest.json when present
- inputs/minimal_task.yaml
- inputs/candidate.yaml
- inputs/grading_spec.json
- inputs/quality_review.json
- inputs/validation_item.json

Required output:
- outputs/candidates/{task_id}.yaml
- outputs/grading_specs/{task_id}.json
- outputs/self_check.json

Repair goal:
- Fix every material issue in inputs/quality_review.json and inputs/validation_item.json.
- Preserve the task id {task_id} and preserve the original scenario intent as much as possible.
- If the original scenario requires source support or business capability that is missing, narrow or redesign it into a source-backed, tool-supported cxbench task rather than inventing tools or laws.
- The repaired candidate must be acceptable under the same independent quality gate, not merely YAML/JSON-valid.

Hard constraints:
- Use only tool names, state fields, rules, and schemas declared in inputs/interface_manifest.json.
- If inputs/resolved_seed_manifest.json exists, preserve the candidate's bound rule seed ids and seed_manifest_version.
- If inputs/resolved_seed_manifest.json exists, do not add unseeded policy.source_refs.
- If a compliance requirement cannot be supported by the declared tools or selected rule cards, remove it from deterministic grading and either narrow the scenario or put only source-backed semantics in policy.llm_judge.
- Every deterministic rule must be satisfiable with declared tools.
- policy.mandatory_disclosure should remain an empty string unless exact deterministic wording is intentionally required.
- Do not use disclosure_before_confirmation or no_promise_disclosure_before_confirmation for banking_customer_ops_v0 unless the required exact disclosure/no-promise tool exists.
- Put no-promise, misleading-disclosure, and other language-semantics checks in grading_spec.grading_policy.semantic_judge_rules or hybrid_hint_rules plus policy.llm_judge, not as hard deterministic rules.
- Current MVP structural validation requires tools.submit to contain a declared submit gateway tool, so do not create no-submit triage tasks in this repair loop.
- Keep initial_state field shapes aligned to this business-system contract: {state_contract}
- Keep task scope aligned with runtime outcome semantics: single-transaction tools need single-transaction outcome, pending transactions need a posted/defer-capable redesign, and overdraft/recurring/joint-authority/police-report semantics require direct source support or must be narrowed.

Independent quality-gate rules:
{quality_rules}

Self-check before final:
- YAML files parse.
- JSON files parse.
- Every output file is under outputs/.
- The repaired task id and grading spec task_id are exactly {task_id}.
- The repaired task has at least one submit gateway in tools.submit.
- No source-code or repository files were read.
"""
        (worker_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    return round_dir


def _run_feedback_repair_workers(round_dir: Path, worker_count: int) -> None:
    for worker_dir in sorted(path for path in round_dir.iterdir() if path.is_dir())[:worker_count]:
        _run_codex_worker(worker_dir)


def _collect_feedback_repair_outputs(
    run: Path, round_dir: Path, failing: list[dict[str, Any]]
) -> list[dict[str, str]]:
    applied: list[dict[str, str]] = []
    workers = sorted(path for path in round_dir.iterdir() if path.is_dir())
    if len(workers) < len(failing):
        raise ValueError(
            f"feedback repair produced {len(workers)} workers for {len(failing)} failing candidates"
        )
    for worker_dir, candidate in zip(workers, failing, strict=True):
        candidate_path = run / str(candidate.get("path", ""))
        task_id = str(candidate.get("task_id") or candidate_path.stem)
        repaired_task = worker_dir / "outputs" / "candidates" / f"{task_id}.yaml"
        repaired_spec = worker_dir / "outputs" / "grading_specs" / f"{task_id}.json"
        if not repaired_task.exists():
            raise ValueError(
                f"missing repaired candidate from {worker_dir}: {repaired_task.relative_to(worker_dir)}"
            )
        if not repaired_spec.exists():
            raise ValueError(
                f"missing repaired grading spec from {worker_dir}: {repaired_spec.relative_to(worker_dir)}"
            )
        task = _load_yaml_mapping(repaired_task)
        if str(task.get("id")) != task_id:
            raise ValueError(
                f"repaired candidate id mismatch in {repaired_task}: expected {task_id}, got {task.get('id')}"
            )
        _load_grading_spec(repaired_spec, expected_task_id=task_id)
        target_task = run / "candidates" / f"{task_id}.yaml"
        target_spec = run / "grading_specs" / f"{task_id}.json"
        shutil.copyfile(repaired_task, target_task)
        shutil.copyfile(repaired_spec, target_spec)
        applied.append(
            {
                "task_id": task_id,
                "candidate": str(target_task.relative_to(run)),
                "grading_spec": str(target_spec.relative_to(run)),
                "worker": str(worker_dir.relative_to(round_dir)),
            }
        )
    (round_dir / "applied_repairs.json").write_text(
        json.dumps(applied, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return applied


def _load_corpus_manifest(path: Path) -> list[dict[str, Any]]:
    raw = _load_yaml(path)
    if isinstance(raw, Mapping):
        entries = raw.get("chunks") or raw.get("files") or []
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError("corpus manifest must be a mapping or list")
    base = path.parent
    items: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        if isinstance(entry, str):
            item_path = _resolve_path(base, entry)
            item_id = item_path.stem
        elif isinstance(entry, Mapping):
            item_path = _resolve_path(base, entry.get("path"))
            item_id = str(entry.get("id") or item_path.stem)
        else:
            raise ValueError(f"invalid corpus manifest entry #{index}")
        if not item_path.exists():
            raise FileNotFoundError(f"corpus file not found: {item_path}")
        items.append({"id": item_id, "path": item_path})
    if not items:
        raise ValueError("corpus manifest has no chunks")
    return items


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    raw = _load_yaml(path)
    if isinstance(raw, Mapping):
        scenarios = raw.get("scenarios") or []
    elif isinstance(raw, list):
        scenarios = raw
    else:
        raise ValueError("scenario matrix must be a mapping or list")
    result = [dict(item) for item in scenarios if isinstance(item, Mapping)]
    if not result:
        raise ValueError("scenario matrix has no scenarios")
    return result


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = _load_yaml(path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"YAML must be a mapping: {path}")
    return dict(raw)


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_path(base: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("path value is required")
    path = Path(str(value))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _prepare_run_dir(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"generation directory is not empty: {run_dir}; choose a new --out")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ["candidates", "grading_specs", "prompts"]:
        (run_dir / name).mkdir(exist_ok=True)


def _copy_inputs(config: GenerationConfig, run_dir: Path) -> None:
    input_dir = run_dir / "inputs"
    input_dir.mkdir(exist_ok=True)
    sources = [config.path, config.domain_rd, config.scenario_matrix]
    if config.corpus_manifest:
        sources.append(config.corpus_manifest)
    if config.rule_registry:
        sources.append(config.rule_registry)
    for source in sources:
        shutil.copyfile(source, input_dir / source.name)


def _split_sections(text: str) -> list[str]:
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [section for section in sections if section]


def _looks_like_compliance_rule(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in COMPLIANCE_KEYWORDS)


def _extractor_prompt(config: GenerationConfig) -> str:
    return (
        f"{EXTRACTOR_VERSION}:{config.domain}:{config.business_system}:{config.user_sim_profiles}"
    )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(payload)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row must be object: {path}")
            rows.append(obj)
    if not rows:
        raise ValueError(f"JSONL file has no rows: {path}")
    return rows


def _shard(items: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    return [items[index::count] for index in range(count)]
