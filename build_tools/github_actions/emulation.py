# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Running component tests against an emulated GPU.

Some GPU targets have no test hardware in CI. For those, a test component can
opt into an *emulated variant* that runs on the CPU cluster, with the mirage
CLI driving the rocjitsu software GPU emulator instead of a real device.

Everything specific to that lives here, so fetch_test_configurations.py stays a
list of components:

  * :func:`get_mirage_profile` / :data:`EMULATED_PROFILES` decide whether a
    family is emulated at all,
  * :func:`build_emulated_job` derives the extra matrix entry, and
  * :func:`is_emulated` and friends let a test script ask, at run time,
    what it is running on.

A component opts in with one matrix field::

    "rocrtst": {..., "emulate": "rocjitsu"}

See docs/development/adding_tests.md for the full field list.
"""

import os
import re
import sys
from collections.abc import Mapping
from copy import deepcopy

# Mirage builtin profiles, keyed by AMDGPU family prefix. Each profile pins one
# emulated GPU agent; see rocm-systems/emulation/mirage/builtin/src/profiles.rs.
# Matched longest key first, so a future target-specific entry wins over the
# family-label entry it sits under.
MIRAGE_PROFILE_BY_FAMILY_PREFIX = {
    "gfx94": "mi300x",
    "gfx950": "mi350x",
    "gfx125": "mi450x",
}

# Profiles we actually schedule jobs for. gfx94X is mapped above so local runs
# can resolve it, but it has ample CI hardware, so emulating it would only burn
# CPU-cluster time for coverage we already have.
EMULATED_PROFILES = frozenset({"mi350x", "mi450x"})

# Emulated jobs are far slower than the same tests on hardware, so scale the
# component's declared timeout rather than making it maintain a second one...
TIMEOUT_MULTIPLIER = 10

# ...but never past an hour. An hour is the budget an emulated component has to
# fit in to be worth scheduling at all, which makes it the right point to stop
# waiting on a wedged emulator. A component that hits the cap needs a cheaper
# `emulate_test_type`, not a longer leash.
MAX_TIMEOUT_MINUTES = 60

# Artifacts an emulated job needs on top of its component's own.
FETCH_ARTIFACT_ARGS = ("--mirage", "--rocjitsu")

# Matrix keys that configure emulation. Consumed here, never emitted: leaking
# them would make workflow expressions like `component.emulate` truthy on
# hardware jobs too.
MATRIX_KEYS = (
    "emulate",
    "emulate_only",
    "emulate_test_type",
    "emulate_env",
)

# `mirage host` starts the workload with env_clear(), re-inheriting only
# PATH/HOME/USER/LANG/LC_ALL/TERM/TMPDIR, so everything test_component.yml put
# in the environment has to cross the session boundary by name.
#
# The cross-check when adding here is test_component.yml's `env:` blocks, not
# the test scripts -- anything consumed by the ROCm runtime rather than by
# Python (ROCM_KPACK_DEBUG, any future HSA_*/ROCR_*) is invisible to a grep of
# the scripts.
FORWARDED_ENV = (
    "THEROCK_BIN_DIR",
    "OUTPUT_ARTIFACTS_DIR",
    "AMDGPU_FAMILIES",
    "AMDGPU_TARGETS",
    "TEST_TYPE",
    "TEST_COMPONENT",
    "SHARD_INDEX",
    "TOTAL_SHARDS",
    # Thread limits test_component.yml derives from the pod's CPU budget.
    # Emulated jobs are the most CPU-hungry in the matrix, and without these
    # their test binaries size thread pools from the node's core count rather
    # than the share this job may use.
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    # Set for every job in the matrix, so emulated jobs would otherwise be the
    # only ones running without it. Read by the ROCm runtime, not by a script.
    "ROCM_KPACK_DEBUG",
)

# What an `emulate_env` NAME=VALUE pair may contain. Deliberately narrow: the
# wrapped command is passed through the workflow and a shell before mirage sees
# it, so anything needing quotes is a latent bug rather than a feature.
_ENV_LITERAL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_.:/,+-]+")


# ---------------------------------------------------------------------------
# Matrix side: used by fetch_test_configurations.py in the configure job.
# ---------------------------------------------------------------------------


def get_mirage_profile(amdgpu_families: str | None) -> str | None:
    """Return the mirage builtin profile emulating `amdgpu_families`, if any."""
    if not amdgpu_families:
        return None
    family = amdgpu_families.lower()
    for prefix in sorted(MIRAGE_PROFILE_BY_FAMILY_PREFIX, key=len, reverse=True):
        if family.startswith(prefix):
            return MIRAGE_PROFILE_BY_FAMILY_PREFIX[prefix]
    return None


def get_emulated_profile(amdgpu_families: str | None, platform: str) -> str | None:
    """Return the profile to schedule emulated jobs for, or None for hardware.

    None whenever this family+platform combination is not one we emulate, which
    is the single switch the caller needs.
    """
    if platform != "linux":
        # The mirage and rocjitsu artifacts are Linux only.
        return None
    profile = get_mirage_profile(amdgpu_families)
    return profile if profile in EMULATED_PROFILES else None


def wrap_in_mirage_run(
    test_script: str,
    emulator: str,
    profile: str,
    emulate_env: dict[str, str] | None = None,
) -> str:
    """Wrap `test_script` so the whole thing runs inside a mirage session.

    The emulator and profile are baked in as literals rather than read from the
    environment, so the command reproduces a run on its own -- which is what
    the failure-reproduction output and the component repositories' copies of
    test_component.yml rely on.

    The result deliberately contains no quote characters, matching the rest of
    the matrix, whose test_script values are already unquoted. No path in it
    may contain spaces.
    """
    # $THEROCK_BIN_DIR rather than a fixed path: test_component.yml unpacks
    # artifacts to ./build, the reproduction container to ./therock-build.
    parts = [
        "$THEROCK_BIN_DIR/mirage",
        "run",
        "--profile",
        profile,
        "--emulator",
        emulator,
        f"--env TEST_EMULATOR={emulator}",
        f"--env TEST_EMULATOR_PROFILE={profile}",
    ]
    # Sorted so the emitted command is stable across configure runs.
    for name, value in sorted((emulate_env or {}).items()):
        if not _ENV_LITERAL_RE.fullmatch(f"{name}={value}"):
            raise ValueError(
                f"emulate_env entry {name}={value!r} is not a quote-free, "
                "space-free literal; mirage receives this as a bare shell word"
            )
        parts.append(f"--env {name}={value}")
    # `${NAME:+--env NAME=$NAME}`, not a bare `--env NAME=$NAME`: the latter
    # emits `--env NAME=` for anything the workflow left unset, and mirage
    # inserts that as an empty *set* value rather than leaving it unset.
    # OMP_NUM_THREADS is the live case -- libgomp rejects an empty value and
    # falls back to the node core count, the opposite of why it is forwarded.
    parts += [f"${{{name}:+--env {name}=${name}}}" for name in FORWARDED_ENV]
    parts += ["--", test_script]
    return " ".join(parts)


def build_emulated_job(job_config: dict, emulator: str, profile: str) -> dict:
    """Derive the emulated variant of an already-expanded job config.

    Deriving from the expanded config rather than the matrix entry means the
    variant inherits every earlier decision -- exclude_family, test labels,
    project selection, container image -- for free.
    """
    emulated = deepcopy(job_config)
    base_job_name = job_config["job_name"]

    emulated["job_name"] = f"{base_job_name} (emulated {profile})"
    # test_component.yml derives TEST_COMPONENT from job_name unless this is
    # set, and test_runner.py maps TEST_COMPONENT to a test directory. The
    # decorated job name is not a directory.
    emulated["test_component"] = job_config.get("test_component", base_job_name)
    emulated["emulator"] = emulator
    emulated["emulator_profile"] = profile
    # rocjitsu emulates the GPU in software, so an emulated job on GPU hardware
    # would hold a scarce runner for its whole timeout and use none of it.
    emulated["linux_cpu_runner"] = True
    emulated["timeout_minutes"] = min(
        job_config["timeout_minutes"] * TIMEOUT_MULTIPLIER,
        MAX_TIMEOUT_MINUTES,
    )
    # install_rocm_from_artifacts.py treats --base-only as exclusive: it takes
    # an if/elif branch that skips the extra-artifact block entirely, so
    # "--base-only --mirage" would fetch neither mirage nor rocjitsu. Dropping
    # it loses nothing, since that block pulls the base patterns too.
    base_args = [
        arg
        for arg in job_config.get("fetch_artifact_args", "").split()
        if arg != "--base-only"
    ]
    emulated["fetch_artifact_args"] = " ".join(base_args + list(FETCH_ARTIFACT_ARGS))
    # Emulated jobs already run a reduced subset; splitting it across CPU
    # cluster jobs costs more in artifact fetches than it saves.
    emulated["total_shards"] = 1
    emulated["shard_arr"] = [1]
    # A pin, not a default: which categories an emulator can get through is a
    # property of the emulator, so a nightly asking for "comprehensive" must
    # not drag the emulated variant along with it.
    emulate_test_type = job_config.get("emulate_test_type")
    if emulate_test_type:
        emulated["test_type"] = emulate_test_type
    emulated["test_script"] = wrap_in_mirage_run(
        job_config["test_script"],
        emulator,
        profile,
        job_config.get("emulate_env"),
    )
    # test_artifacts.yml checks multi_gpu_runner and is_benchmark *before*
    # linux_cpu_runner when picking a runner, so leaving either in place would
    # route this job onto GPU hardware it cannot use.
    emulated.pop("multi_gpu", None)
    emulated.pop("multi_gpu_runner", None)
    emulated.pop("is_benchmark", None)
    for key in MATRIX_KEYS:
        emulated.pop(key, None)
    return emulated


# ---------------------------------------------------------------------------
# Run-time side: used by test scripts already inside the mirage session.
# ---------------------------------------------------------------------------

# `os.environ` is an `os._Environ`, not a `dict`.
Env = Mapping[str, str]


def emulator_name(env: Env = os.environ) -> str:
    """Emulator backend for this job, or "" when running on hardware."""
    return (env.get("TEST_EMULATOR") or "").strip()


def emulator_profile(env: Env = os.environ) -> str:
    """mirage profile for this job, or "" when running on hardware."""
    return (env.get("TEST_EMULATOR_PROFILE") or "").strip()


def is_emulated(env: Env = os.environ) -> bool:
    """True when this job's tests run under an emulator."""
    return bool(emulator_name(env))


def log_emulator_banner(env: Env = os.environ) -> None:
    """Print what is being emulated, for readability in CI logs."""
    print(f"# TEST_EMULATOR: {emulator_name(env) or '<none>'}")
    print(f"# TEST_EMULATOR_PROFILE: {emulator_profile(env) or '<none>'}")
    print(f"# AMDGPU_FAMILIES: {env.get('AMDGPU_FAMILIES', '<unset>')}")
    print(f"# AMDGPU_TARGETS: {env.get('AMDGPU_TARGETS', '<unset>')}")
    sys.stdout.flush()
