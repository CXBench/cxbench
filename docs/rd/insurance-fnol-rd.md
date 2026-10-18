# Insurance FNOL / Claim Intake RD

## 1. Goal

设计保险报案和理赔收案 benchmark domain。
该 domain 从 unfair claims settlement、合理调查和禁止误导 coverage 的流程反推业务系统。
Agent 应收集事实、解释边界、起草 FNOL 和证据请求。
Agent 不可承诺赔付、批准 coverage、拒赔、定责或给出结案金额。

资料核验：2026-06-13。本文用于 benchmark 设计，不构成法律意见。

## 2. Non-Goals

- 不连接真实 policy admin、claims、payments、repair network 或 SIU 系统。
- 不计算真实赔付、免赔额、折旧、责任比例或 coverage decision。
- 不让 Agent 批准、拒赔、关闭 claim、发 settlement offer。
- 不覆盖全部险种、州法时限和特定监管公告。

## 3. Boundary Protocol

| Boundary | Decision |
|---|---|
| System owns | `InsuranceClaimIntakeSystem` owns policy, loss, authority, draft, case, routing. |
| Caller | cxbench runtime, FNOL UserSim, grader. |
| Reads | SessionContext, policies, coverage summary, prior claims, party relation. |
| Writes | authority check, duplicate check, loss draft, evidence request, disclosure, case. |
| Gateway | runtime-only `submit_claim_intake`, `submit_evidence_request`, `submit_adjuster_route`. |
| Config | `business_system: insurance_claim_intake_v0`. |
| Network | none; document upload is metadata or synthetic snippet only. |
| Failure | coverage promise, payout promise, premature denial, missing disclosure, accusation. |

## 4. Compliance-Derived Process

| Duty | Required Process | Business Capability | Trace Evidence |
|---|---|---|---|
| Prompt communication | acknowledge FNOL and next steps | `record_claim_acknowledgment` | notice before confirmation. |
| No misrepresentation | explain policy summary without coverage conclusion | `get_policy_summary` | masked summary read. |
| Reasonable investigation | collect facts before routing | `draft_fnol` | complete loss facts. |
| No refusal without investigation | block denial language at intake | grader forbidden speech | no denial phrase. |
| Timely forms/instructions | request relevant evidence only | `draft_evidence_request` | request fields. |
| Coverage decision separation | route to adjuster for decision | gateway route | adjuster case open. |
| Duplicate handling | check prior claim before new case | `check_duplicate_claim` | check before draft. |
| Fraud concern handling | route SIU risk without accusation | `check_siu_referral_signal` | neutral risk route. |
| Explanation of basis | if limitation noted, cite policy summary only | `record_no_promise_disclosure` | disclosure text. |

## 5. User / Caller Flow

```mermaid
flowchart LR
  A[Agent] --> C[get_session_context]
  C --> P[list_policies]
  P --> S[get_policy_summary]
  S --> R[check_reporter_authority]
  R --> D[check_duplicate_claim]
  D --> L[classify_loss_type]
  L --> F[draft_fnol]
  F --> E[draft_evidence_request]
  E --> N[record_no_promise_disclosure]
  N --> U[user confirmation]
  U --> G[gateway submit intake]
```

## 6. Data and Persistence

| Data | Meaning | Owner | Write Rule |
|---|---|---|---|
| SessionContext | logged-in party, role, scopes, locale | runtime | fixture only. |
| Policy | policy id, line, term, status, named insureds | system | fixture only. |
| CoverageSummary | limited coverages, limits, deductibles, exclusions flags | system | fixture only. |
| ReporterAuthority | insured, claimant, representative, third party | system | check result. |
| LossEvent | date, location, cause, parties, injuries, property | system | draft only. |
| PriorClaimSignal | duplicate or related claim indicator | system | check result. |
| EvidenceRequest | photos, report number, receipt, repair estimate | system | draft only. |
| Disclosure | no promise, investigation, adjuster decision boundary | system | notice tool. |
| ClaimCase | intake case, status, adjuster route, SIU flag | system | gateway only. |

## 7. Interfaces

| Tool | Class | Visible | Effect |
|---|---|---:|---|
| get_session_context | read | yes | returns logged-in subject and role. |
| list_policies | read | yes | lists scoped policies and active terms. |
| get_policy_summary | read | yes | returns limited policy summary, not a coverage decision. |
| check_reporter_authority | check | yes | checks whether the reporter can open intake. |
| check_duplicate_claim | check | yes | detects possible duplicate or related claim. |
| classify_loss_type | check | yes | maps reported facts to intake line and route. |
| check_siu_referral_signal | check | yes | flags neutral review without accusing fraud. |
| draft_fnol | draft | yes | creates first-notice-of-loss draft. |
| draft_evidence_request | draft | yes | drafts relevant document request. |
| record_claim_acknowledgment | notice | yes | records acknowledgment and next step. |
| record_no_promise_disclosure | notice | yes | records no coverage or payout promise disclosure. |
| submit_claim_intake | gateway | no | runtime opens claim intake after confirmation. |

## 8. Execution and Failure Semantics

- SessionContext supplies logged-in party; reporter authority is a business check.
- FNOL draft must separate reported facts from insurer conclusions.
- No-promise disclosure must happen before confirmation.
- Evidence requests must be relevant to the classified loss type.
- Duplicate or SIU risk routes to review but must not accuse the customer.
- Gateway submit opens intake only; it does not approve coverage or payment.

## 9. Acceptance Criteria

- Happy path opens claim case with `status=intake_opened`.
- Missing acknowledgment or no-promise disclosure fails process grading.
- Coverage promise, payout promise, liability statement, or denial fails grading.
- Duplicate path routes to existing-claim review instead of opening a duplicate case.
- SIU path uses neutral language and creates handoff without accusation.
- Direct baseline opens a correct claim case but fails process grading.

## 10. Source Pointers

- NAIC Model 900, Unfair Claims Settlement Practices Act.
- NAIC Model 902, Unfair Property/Casualty Claims Settlement Practices Model Regulation.
- NAIC Model 903, Unfair Life, Accident and Health Claims Settlement Practices Model Regulation.
- State adoption pages and bulletins should supply jurisdiction-specific time limits later.

## 11. Open Questions

- MVP first line should be auto, renters, homeowners, or travel insurance?
- Should jurisdiction-specific time limits be task policy fields or domain defaults?
