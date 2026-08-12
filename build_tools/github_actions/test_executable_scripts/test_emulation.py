# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Checks that the emulated GPU comes up and identifies itself correctly.

The job's `test_script` is already wrapped in a mirage session, so simply
running a ROCm tool here exercises the whole stack end to end:

  mirage CLI -> builtin profile -> rocjitsu emulator -> emulated KFD/HSA ->
  ROCr runtime from the artifacts -> the tool

When this fails, every other emulated job is expected to fail too.

This is not a duplicate of the "Driver / GPU sanity check" step in
test_component.yml. That step is skipped on emulated jobs: it reports on the
host's amdgpu driver, and an emulated job runs on a CPU runner that has none.
The device this job's tests run against exists only inside the mirage session,
so it can only be inspected from in here.

TEST_TYPE selects how much to run; see docs/development/test_filtering.md.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import emulation

logging.basicConfig(level=logging.INFO)

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR", "./build/bin"))

# Test categories in increasing depth. The ffm-* tiers are the same depths
# under a different name, so they select the same checks.
CATEGORY_DEPTH = {
    "quick": 1,
    "standard": 2,
    "comprehensive": 3,
    "full": 4,
}

# Tools to run, with the shallowest category that includes each. Both
# enumerate the emulated agent; they differ in how much of the runtime they
# walk to do it, so the deeper tier is strictly more of the stack.
CHECKS = (
    ("quick", "rocm_agent_enumerator"),
    ("standard", "rocminfo"),
)

# A tool that has not exited by now is wedged, not slow. Bounded here rather
# than left to the step timeout so the log still shows how far it got.
CHECK_TIMEOUT_SECONDS = 10 * 60


def get_test_depth(test_type: str) -> int:
    """Depth for `test_type`, defaulting to the shallowest."""
    category = test_type.lower().removeprefix("ffm-")
    depth = CATEGORY_DEPTH.get(category)
    if depth is None:
        logging.warning(f"Unknown TEST_TYPE '{test_type}', running quick checks")
        return CATEGORY_DEPTH["quick"]
    return depth


def get_expected_targets() -> set[str]:
    """gfx targets this job was built and scheduled for.

    Taken from the job's own inputs rather than from a table of what each
    mirage profile emulates: the assertion worth making is that the emulated
    device matches the artifacts under test, and a second table would only
    duplicate the profile definitions in rocm-systems.
    """
    return {
        target.strip()
        for target in os.getenv("AMDGPU_TARGETS", "").split(",")
        if target.strip()
    }


def run_check(tool: str) -> str:
    """Run `tool` inside the mirage session and return its stdout."""
    cmd = [str(THEROCK_BIN_DIR / tool)]
    logging.info(f"++ Exec $ {' '.join(cmd)}")
    # Already inside the mirage session, so this is a plain invocation.
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
        timeout=CHECK_TIMEOUT_SECONDS,
    )
    # Always echo it; the emulated run's output is the point of the job.
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.stdout


def main() -> int:
    if not emulation.is_emulated():
        print(
            "ERROR: test_emulation.py only runs in an emulated job "
            "(TEST_EMULATOR is unset).",
            file=sys.stderr,
        )
        return 1

    emulation.log_emulator_banner()

    depth = get_test_depth(os.getenv("TEST_TYPE") or "quick")
    reported_targets = set()
    for category, tool in CHECKS:
        if CATEGORY_DEPTH[category] > depth:
            logging.info(f"Skipping {tool}, which needs TEST_TYPE >= {category}")
            continue
        output = run_check(tool)
        reported_targets |= {w for w in output.split() if w.startswith("gfx")}

    if not reported_targets:
        print("ERROR: the emulated stack reported no gfx agent", file=sys.stderr)
        return 1

    expected_targets = get_expected_targets()
    if not expected_targets:
        # Warn rather than fail: the checks above already proved the stack
        # comes up, and there is nothing to compare against.
        logging.warning("AMDGPU_TARGETS is unset, not checking the agent identity")
    elif not expected_targets & reported_targets:
        print(
            f"ERROR: expected an agent for {sorted(expected_targets)}, but the "
            f"emulator reported {sorted(reported_targets)}",
            file=sys.stderr,
        )
        return 1

    print(f"# Emulated agent(s) present: {sorted(reported_targets)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
