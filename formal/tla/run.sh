#!/usr/bin/env bash
# Model-check formal/tla/OllamaAgent.tla with TLC under every OllamaAgent_*.cfg in this directory.
#
# Each cfg starts with a `\* expect:` line: `pass`, or `violation <Invariant>` for a configuration
# that is *meant* to break an invariant (it shows the invariant is not vacuous).  The script exits
# 1 when any run disagrees with its expectation.
#
# Needs Java 11+.  tla2tools.jar (the TLA+ tools, BSD/MIT) is fetched once into ~/.cache/tla2tools/;
# set TLA2TOOLS_JAR to use another copy.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
jar=${TLA2TOOLS_JAR:-$HOME/.cache/tla2tools/tla2tools.jar}
if [ ! -s "$jar" ]; then
    mkdir -p "$(dirname "$jar")"
    curl -fsSL -o "$jar" https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
status=0
for cfg in "$here"/OllamaAgent_*.cfg; do
    name=$(basename "$cfg" .cfg)
    expect=$(sed -n 's/^\\\* expect: *//p' "$cfg" | head -1)
    out=$(cd "$work" && java -XX:+UseParallelGC -cp "$jar" tlc2.TLC -config "$cfg" -workers auto \
          -metadir "$work/$name" -cleanup "$here/OllamaAgent.tla" 2>&1) || true
    states=$(sed -n 's/^\([0-9]*\) states generated, \([0-9]*\) distinct states found.*/\2 distinct states/p' <<<"$out" | tail -1)
    if grep -q "No error has been found" <<<"$out"; then got="pass"
    elif grep -q "is violated" <<<"$out"; then got="violation $(sed -n 's/.*Invariant \([A-Za-z0-9_]*\) is violated.*/\1/p;s/.*Temporal properties were violated.*/liveness/p' <<<"$out" | head -1)"
    else got="error"; fi
    if [ "$got" = "$expect" ]; then mark=ok; else mark=XX; status=1; fi
    printf '%s  %-24s %-22s expected: %-22s %s\n' "$mark" "$name" "$got" "$expect" "$states"
    if [ "$mark" = XX ] || [ -n "${VERBOSE:-}" ]; then printf '%s\n' "$out" | sed 's/^/      /' | tail -60; fi
done
exit $status
