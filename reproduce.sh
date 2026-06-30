#!/usr/bin/env bash
# Reproduce every headline number from the released logs. No API keys, standard library only.
#   ./reproduce.sh
# Each script prints its own EXPECTED -> got checks and exits non-zero on any mismatch.
set -u
cd "$(dirname "$0")/reproduce" || exit 2

SCRIPTS=(headline reproduction route_pin framing confab resample)
declare -a RESULT
pass=0
for s in "${SCRIPTS[@]}"; do
    echo "──────────────────────────────────────────────────────────────────────"
    echo "▶ ${s}.py"
    if python3 "${s}.py"; then RESULT+=("PASS  ${s}"); pass=$((pass+1)); else RESULT+=("FAIL  ${s}"); fi
done

echo "══════════════════════════════════════════════════════════════════════"
echo "REPRODUCTION SUMMARY"
for r in "${RESULT[@]}"; do echo "  ${r}"; done
echo "  ${pass}/${#SCRIPTS[@]} reproduced"
echo "══════════════════════════════════════════════════════════════════════"
[ "${pass}" -eq "${#SCRIPTS[@]}" ] || exit 1
