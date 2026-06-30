"""Tiny shared helper: print a PASS/FAIL line and remember if anything failed.

Every reproduce/ script ends by calling done(), which exits non-zero if any check
failed, so reproduce.sh can tally the suite.  No third-party dependencies anywhere
in reproduce/ — standard library only, no API keys, runs offline against ./data.
"""
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
_failed = []

def check(label, got, expected):
    """Compare got vs expected, print one aligned PASS/FAIL line, record failures."""
    ok = got == expected
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label:38s} expected {str(expected):>12}   got {str(got):>12}   {'OK' if ok else 'MISMATCH <<<'}")
    if not ok:
        _failed.append(label)
    return ok

def note(msg):
    print(f"        {msg}")

def done(title):
    if _failed:
        print(f"\n  {title}: {len(_failed)} MISMATCH(es): {', '.join(_failed)}\n")
        sys.exit(1)
    print(f"\n  {title}: all checks reproduced.\n")
    sys.exit(0)
