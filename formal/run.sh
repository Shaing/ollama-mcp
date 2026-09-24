#!/usr/bin/env bash
# Run every verification layer described in formal/README.md. Exit 1 if any disagrees with SPEC.md.
set -uo pipefail
cd "$(dirname "$0")/.."
status=0
echo "== 1/4  SMT proofs (Z3): formal/smt/verify.py"
uv run python formal/smt/verify.py || status=1
echo; echo "== 2/4  model checking (TLC): formal/tla/*.cfg"
formal/tla/run.sh || status=1
echo; echo "== 3/4  symbolic execution (CrossHair, bounded; CROSSHAIR_TIMEOUT=${CROSSHAIR_TIMEOUT:-30}s per condition)"
uv run crosshair check formal/crosshair/contracts.py --analysis_kind=PEP316 \
    --per_condition_timeout="${CROSSHAIR_TIMEOUT:-30}" --report_all || status=1
echo; echo "== 4/4  property-based and static tests"
uv run pytest tests/test_properties.py tests/test_spec_static.py -q || status=1
echo; [ $status = 0 ] && echo "formal: everything agrees with formal/SPEC.md" || echo "formal: DISAGREEMENT with formal/SPEC.md (see above)"
exit $status
