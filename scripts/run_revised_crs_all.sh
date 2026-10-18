#!/usr/bin/env bash
set -Eeuo pipefail

# Run every model role through the CRS endpoint. This is the smallest
# self-contained smoke/full-run entry point for a CRS-only environment.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"

agent_model="${CXBENCH_AGENT_MODEL:-gpt-5.6-sol}"
user_model="${CXBENCH_USER_MODEL:-crs:gpt-5.6-sol}"
judge_model="${CXBENCH_JUDGE_MODEL:-crs:gpt-5.6-sol}"
task_glob="benchmarks/cxbench-banking-judge-v20260917/tasks/*.yaml"
smoke_task="benchmarks/cxbench-banking-judge-v20260917/tasks/BK-RAW-001.us.yaml"
concurrency="${CXBENCH_CONCURRENCY:-1}"
task_timeout="${CXBENCH_TASK_TIMEOUT:-1200}"
run_id="${CXBENCH_RUN_ID:-}"
runs_dir="${CXBENCH_RUNS_DIR:-runs/main_experiment}"
dry_run=false
smoke_test=false
resume=false

usage() {
  cat <<'EOF'
Run the revised 120-task rubric with agent, user simulator, and judge on CRS.

Usage:
  scripts/run_revised_crs_all.sh [options]

Options:
  --dry-run                 Validate tasks without API calls or run artifacts.
  --smoke-test              Run one banking task end to end.
  --run-id ID               Set the output directory name.
  --runs-dir PATH           Parent output directory (relative to repository or absolute).
  --resume                  Resume the run identified by --run-id.
  --concurrency N           Worker count (default: CXBENCH_CONCURRENCY or 1).
  --task-timeout SECONDS    Per-attempt timeout (default: CXBENCH_TASK_TIMEOUT or 1200).
  -h, --help                Show this help.

Environment:
  CRS_OAI_KEY               Required CRS credential (load it from a private .env).
  CXBENCH_AGENT_MODEL       Defaults to gpt-5.6-sol.
  CXBENCH_USER_MODEL        Defaults to crs:gpt-5.6-sol.
  CXBENCH_JUDGE_MODEL       Defaults to crs:gpt-5.6-sol.
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "error: $1 requires a value" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --smoke-test) smoke_test=true; shift ;;
    --resume) resume=true; shift ;;
    --run-id) require_value "$@"; run_id="$2"; shift 2 ;;
    --runs-dir) require_value "$@"; runs_dir="$2"; shift 2 ;;
    --concurrency) require_value "$@"; concurrency="$2"; shift 2 ;;
    --task-timeout) require_value "$@"; task_timeout="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: concurrency must be a positive integer" >&2
  exit 2
fi
if ! [[ "$task_timeout" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: task timeout must be a positive integer" >&2
  exit 2
fi

task_selection="$task_glob"
run_kind="revised-120"
if [[ "$smoke_test" == true ]]; then
  task_selection="$smoke_task"
  run_kind="single-task-smoke"
fi
if [[ -z "$run_id" ]]; then
  if [[ "$resume" == true ]]; then
    echo "error: --resume requires --run-id or CXBENCH_RUN_ID" >&2
    exit 2
  fi
  run_id="crs-${run_kind}-$(date +%Y%m%d-%H%M%S)"
fi

entrypoint="main_experiment.run"
if [[ "$resume" == true ]]; then
  entrypoint="main_experiment.resume"
fi

command=(
  uv run --frozen python -m "$entrypoint"
  --model "$agent_model"
  --agent-provider crs
  --user-model "$user_model"
  --judge-model "$judge_model"
  --judge-provider crs
  --task "$task_selection"
  --wire-api responses
  --mode least_privilege
  --grader hybrid
  --executor process
  --concurrency "$concurrency"
  --task-timeout "$task_timeout"
  --run-id "$run_id"
  --runs-dir "$runs_dir"
)

if [[ "$dry_run" == true ]]; then
  command+=(--dry-run)
fi

cd "$repo_root"
export CRS_OAI_WIRE_API="responses"
export CRS_OAI_RESPONSES_STREAM="true"
echo "run_id: $run_id"
echo "run_kind: $run_kind"
echo "entrypoint: $entrypoint"
echo "models: agent=$agent_model user=$user_model judge=$judge_model"
echo "provider: CRS/Responses for all roles"
exec "${command[@]}"
