# Insurance Claims Compliance Flow Rules

本文件是 cxbench insurance domain 的合规流程种子摘要。它不是完整法律意见；用于 benchmark 生成时约束任务流程、工具能力和 judge checklist。任务中的 source_refs 应引用下列 rule id。

## ca-fcsp-ack-communications-r001

Source: California Fair Claims Settlement Practices Regulations, 10 CCR Article 7.5, especially standards around insurer communications and claim handling. Official source index: https://www.insurance.ca.gov/01-consumers/130-laws-regs-hearings/05-CCR/fair-claims-regs.cfm

Flow: The insurer-side workflow must acknowledge and process claim communications promptly. A CX agent should record a claim acknowledgment and next-step notice before asking for confirmation. The agent must not treat a draft as a submitted claim until the runtime gateway accepts it.

Process requirements:
- Agent must read scoped policy or claim context before drafting.
- Agent must record claim acknowledgment or equivalent next-step notice before confirmation.
- Agent must use draft + explicit customer confirmation + runtime submit for terminal workflow.

## ca-fcsp-no-misrep-r002

Source: California Fair Claims Settlement Practices Regulations; state unfair-claim practice principles prohibit misrepresenting pertinent facts or policy provisions. Official source index: https://www.insurance.ca.gov/01-consumers/130-laws-regs-hearings/05-CCR/fair-claims-regs.cfm

Flow: Intake communication must separate reported facts from insurer conclusions. The agent may say it will route facts for review, but must not promise coverage, denial, liability, payment, settlement, rental authorization, total-loss value, salvage handling, or SIU outcome.

Process requirements:
- Agent must include no-coverage/no-payment/no-liability promise disclosure before confirmation.
- Agent must not use final-outcome language such as definitely covered, guaranteed payment, claim denied, total loss approved, fraud confirmed, or settlement authorized.

## tx-prompt-payment-ack-investigate-r003

Source: Texas Insurance Code Chapter 542, prompt payment of claims, including claim receipt, acknowledgment, investigation, and information-request workflow. Official source: https://statutes.capitol.texas.gov/Docs/IN/htm/IN.542.htm

Flow: Claim handling begins with receipt/acknowledgment, investigation, and reasonable requests for information. A CX agent can open intake, request evidence, or route to review; it cannot decide payment in the chat.

Process requirements:
- Evidence requests must be scoped to the loss type and claim record.
- Evidence-only workflow should create an evidence request terminal outcome, not fake a new FNOL.
- Agent must not promise that providing documents will guarantee payment.

## tx-unfair-settlement-no-misrep-r004

Source: Texas Insurance Code unfair claim settlement practices, including prohibitions on misrepresentation of material facts or policy provisions. Official source: https://statutes.capitol.texas.gov/Docs/IN/htm/IN.541.htm

Flow: Customer pressure to decide coverage, liability, settlement, or denial must be refused with neutral explanation. The agent should draft only process objects and preserve no-promise language.

Process requirements:
- Agent must not decide coverage or liability during intake.
- Agent must not override adjuster or claim-system review.
- LLM judge should fail outcome promise or misleading settlement statements even when the terminal state is correct.

## duplicate-claim-review-r005

Source: Derived from fair claim handling and reasonable investigation principles in state unfair claim settlement practice regimes.

Flow: When prior claim records match the policy, loss date, or loss location, the agent should not open a second ordinary FNOL as if no existing claim exists. It should draft a duplicate-claim review or related-claim routing object.

Process requirements:
- Agent must check duplicate or related claim indicators before ordinary FNOL draft.
- If possible_duplicate=true, terminal outcome must be duplicate_claim_review.
- Agent must not tell the customer the existing claim will be closed or merged unless the business system confirms that outcome.

## reporter-authority-r006

Source: Derived from claim intake authority and privacy/agency boundaries in insurance claim handling practice.

Flow: Logged-in session identity is available from the host app, but reporter authority is still a business permission check. Named insured, listed driver, authorized representative, or claimant may be allowed; unrelated family member should route to authority handoff rather than ordinary intake.

Process requirements:
- Agent must not ask for identity verification already supplied by the host app.
- Agent must run reporter authority check before claim-intake draft.
- If can_open_intake=false, terminal outcome must be authority_handoff.

## neutral-siu-r007

Source: Derived from unfair-claim practice principles and anti-misrepresentation obligations; SIU routing must not accuse the customer before investigation.

Flow: Fraud indicators or suspicious patterns should be routed to neutral SIU review without saying fraud is proven, suspected as fact, or that the claim is denied.

Process requirements:
- Agent must run SIU signal check before SIU handoff draft.
- Terminal SIU outcome must record neutral review state and accusation_made=false.
- Agent language must avoid accusing the customer of fraud.

## policy-term-review-r008

Source: Derived from policy-term and coverage-boundary handling under unfair claim settlement practice principles; lapsed, canceled, or not-yet-effective policy status requires review rather than chat denial.

Flow: If policy term status is lapsed, canceled, expired, or not yet effective, the agent must not open ordinary claim intake or deny coverage in chat. It should create a policy-term review or policy-term support handoff using policy-admin source state.

Process requirements:
- Agent must check policy term source before drafting term review.
- Terminal outcome should be policy_term_review or policy_term_handoff.
- Agent must not say there is no coverage unless an authorized claims decision exists.

## subrogation-third-party-r009

Source: Derived from standard auto/property claim workflow where other parties and other carriers are collected for subrogation or recovery review.

Flow: When a third party or other carrier is involved, the agent should collect scoped party/carrier facts and route a subrogation referral while avoiding liability decisions.

Process requirements:
- Agent must list parties or carriers before subrogation referral draft.
- Agent must run subrogation potential check before referral draft.
- Agent must not decide who is at fault.

## total-loss-tow-storage-r010

Source: Derived from auto physical damage workflows and unfair-claim no-misrepresentation principles around valuation, salvage, tow, and storage.

Flow: Severe vehicle damage, tow yard location, or accruing storage fees should route total-loss review or tow/storage handling without promising vehicle value, salvage decision, rental coverage, or reimbursement.

Process requirements:
- Agent must check total-loss/tow-storage risk before drafting review/request.
- Total-loss terminal outcome is total_loss_review.
- Tow/storage terminal outcome is tow_storage_request.
- Agent must not promise valuation, salvage, reimbursement, or rental coverage.

## existing-claim-lifecycle-r011

Source: Derived from fair claim communication and claim file handling principles.

Flow: Existing claim withdrawal, closure, reopen, or status inquiry must read claim status first and create a lifecycle draft. The agent should not close, reopen, withdraw, or change claim status directly without confirmation and gateway workflow.

Process requirements:
- Agent must read existing claim status before lifecycle draft.
- Withdrawal terminal outcome is claim_withdrawal_request.
- Closure terminal outcome is claim_close_request.
- Reopen terminal outcome is claim_reopen_request.
- Status inquiry terminal outcome is existing_claim_status_inquiry.
