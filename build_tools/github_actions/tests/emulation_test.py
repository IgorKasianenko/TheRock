# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the emulated test scripts under test_executable_scripts/."""

import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parent.parent / "test_executable_scripts"))

# Modules under test call logging.basicConfig() at import, by convention across
# test_executable_scripts/. That is right for a script run as `python
# test_foo.py` and wrong here: it would otherwise leave the root logger at INFO
# with a stderr handler for every *other* module in the `pytest build_tools`
# session, which is a collection-order dependency.
#
# The snapshot must be taken before the FIRST polluting import, not between
# them: fetch_test_configurations also calls basicConfig() at import, so
# capturing after it would restore the pollution rather than undo it.
_root_logger = logging.getLogger()
_prior_level = _root_logger.level
_prior_handlers = list(_root_logger.handlers)

import fetch_test_configurations
import emulation
import test_emulation_smoke

_root_logger.setLevel(_prior_level)
_root_logger.handlers[:] = _prior_handlers


class EmulationEnvTest(unittest.TestCase):
    """TEST_EMULATOR / TEST_EMULATOR_PROFILE reading."""

    def test_unset_is_not_emulated(self):
        self.assertFalse(emulation.is_emulated({}))
        self.assertEqual(emulation.emulator_name({}), "")
        self.assertEqual(emulation.emulator_profile({}), "")

    def test_blank_is_not_emulated(self):
        # GitHub Actions renders an unset matrix field as an empty string.
        env = {"TEST_EMULATOR": "", "TEST_EMULATOR_PROFILE": ""}
        self.assertFalse(emulation.is_emulated(env))

    def test_values_are_stripped(self):
        env = {"TEST_EMULATOR": " rocjitsu ", "TEST_EMULATOR_PROFILE": " mi350x\n"}
        self.assertTrue(emulation.is_emulated(env))
        self.assertEqual(emulation.emulator_name(env), "rocjitsu")
        self.assertEqual(emulation.emulator_profile(env), "mi350x")


class RocmPathTest(unittest.TestCase):
    def test_derived_from_therock_bin_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"THEROCK_BIN_DIR": str(Path(tmp) / "bin")}
            self.assertEqual(emulation.rocm_path(env), Path(tmp).resolve())

    def test_missing_bin_dir_raises(self):
        with self.assertRaises(RuntimeError):
            emulation.rocm_path({})


class EmulationSmokeTest(unittest.TestCase):
    """rocminfo output validation for test_emulation_smoke.py."""

    GOOD_MI350X = """
    Agent 1
      Name:  AMD EPYC
      Device Type: CPU
    Agent 2
      Name:  gfx950
      Device Type: GPU
    """

    def test_matching_agent_passes(self):
        self.assertEqual(
            test_emulation_smoke.check_rocminfo_output(self.GOOD_MI350X, "mi350x"), []
        )

    def test_wrong_agent_is_reported(self):
        problems = test_emulation_smoke.check_rocminfo_output(
            self.GOOD_MI350X, "mi450x"
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("expected a gfx1250 agent", problems[0])

    def test_no_agents_is_reported(self):
        problems = test_emulation_smoke.check_rocminfo_output("", "mi350x")
        self.assertIn("no agents at all", problems[0])

    def test_unknown_profile_only_checks_for_an_agent(self):
        # An unmapped profile should not fail the smoke test outright; it just
        # loses the agent-identity assertion.
        self.assertEqual(
            test_emulation_smoke.check_rocminfo_output(self.GOOD_MI350X, "future-gpu"),
            [],
        )

    def test_every_scheduled_profile_has_an_expected_agent(self):
        # fetch_test_configurations.py picks the profile from the AMDGPU
        # family; this map turns it back into the gfx target the emulated
        # agent must report. A profile missing here silently downgrades the
        # smoke test to "some agent exists", which is the one thing it is not
        # supposed to settle for.
        for profile in fetch_test_configurations._EMULATED_PROFILES:
            self.assertIn(profile, test_emulation_smoke.EXPECTED_GFX_BY_PROFILE)

    def test_expected_agent_matches_the_family_it_is_scheduled_for(self):
        # The two maps are written from opposite ends -- family -> profile in
        # fetch_test_configurations.py, profile -> gfx target here -- so a typo
        # in either shows up as a round trip that does not close. Compared as a
        # prefix because the family key is a family label ("gfx125") and the
        # target is a specific chip ("gfx1250").
        prefixes = fetch_test_configurations._MIRAGE_PROFILE_BY_FAMILY_PREFIX
        for family_prefix, profile in prefixes.items():
            with self.subTest(profile=profile):
                gfx_target = test_emulation_smoke.EXPECTED_GFX_BY_PROFILE[profile]
                self.assertTrue(
                    gfx_target.startswith(family_prefix),
                    f"family prefix {family_prefix} maps to profile {profile}, "
                    f"which is expected to present {gfx_target}",
                )


class EmptySelectionGuardTest(unittest.TestCase):
    """test_runner.py's guard against a label selection that matches nothing.

    ctest exits 0 and prints "No tests were found!!!" in that case, so without
    this an emulated job whose category is missing from the artifacts would
    report success having run nothing.
    """

    @staticmethod
    def _import_test_runner():
        # test_runner.py does its work at import time and needs these two.
        # THEROCK_BIN_DIR is only path-manipulated, so it need not exist.
        #
        # The module is dropped from sys.modules on the way out, not left
        # cached: it freezes TEST_COMPONENT, TEST_DIR, ROCM_PATH, TEST_TYPE and
        # environ_vars at import, and mock.patch.dict rewinds os.environ but
        # not those. Leaving it cached would mean whichever test module
        # imported test_runner first silently decided that configuration for
        # every later one (build_tools/github_actions/tests/unit_test_runner.py
        # imports it with a different TEST_COMPONENT and a real temp dir).
        env = {"THEROCK_BIN_DIR": "/nonexistent/bin", "TEST_COMPONENT": "rocrtst"}
        prior_module = sys.modules.pop("test_runner", None)
        with mock.patch.dict(os.environ, env, clear=False):
            prior_level = _root_logger.level
            prior_handlers = list(_root_logger.handlers)
            try:
                import test_runner
            finally:
                _root_logger.setLevel(prior_level)
                _root_logger.handlers[:] = prior_handlers
                # Restore whatever another module had imported, if anything.
                if prior_module is not None:
                    sys.modules["test_runner"] = prior_module
                else:
                    sys.modules.pop("test_runner", None)
        return test_runner

    def _selection_is_empty(self, stdout, returncode=0, cmd=None):
        test_runner = self._import_test_runner()
        completed = subprocess.CompletedProcess([], returncode, stdout, "")
        with mock.patch.object(
            test_runner.subprocess, "run", return_value=completed
        ) as run:
            result = test_runner.selection_is_empty(
                cmd if cmd is not None else ["ctest", "-V", "-L", "^x$"]
            )
        return result, run.call_args[0][0]

    def test_no_matching_tests_is_empty(self):
        empty, _ = self._selection_is_empty("Test project /x\nNo tests were found!!!\n")
        self.assertTrue(empty)

    def test_matching_tests_is_not_empty(self):
        empty, _ = self._selection_is_empty("  Test #7: rocrtst64_ffm-quick_suite\n")
        self.assertFalse(empty)

    def test_listing_failure_is_not_reported_as_empty(self):
        # If ctest cannot even list, let the real run surface the problem
        # rather than blaming an empty selection for it.
        empty, _ = self._selection_is_empty("", returncode=1)
        self.assertFalse(empty)

    def test_listing_reuses_the_real_command(self):
        # The check must ask ctest about the command that is actually going to
        # run; re-deriving the labels here could disagree with it.
        _, list_cmd = self._selection_is_empty("No tests were found!!!\n")
        self.assertEqual(list_cmd[-1], "-N")
        self.assertIn("-L", list_cmd)
        self.assertIn("^x$", list_cmd)
        # -V would make listing noisy and serves no purpose with -N.
        self.assertNotIn("-V", list_cmd)

    def test_listing_drops_the_shard_stride(self):
        # --tests-information is a property of *this shard*, not of the label
        # selection. A tail shard of a suite with fewer entries than shards is
        # legitimately empty; keeping the stride here would report that as
        # "the category matches nothing" -- a hard failure on emulated jobs.
        _, list_cmd = self._selection_is_empty(
            "No tests were found!!!\n",
            cmd=["ctest", "-L", "^x$", "--tests-information", "4,,4", "--timeout", "7"],
        )
        self.assertNotIn("--tests-information", list_cmd)
        self.assertNotIn("4,,4", list_cmd)
        # The rest of the command survives, including the value of a flag that
        # merely follows the stripped one.
        self.assertEqual(list_cmd, ["ctest", "-L", "^x$", "--timeout", "7", "-N"])

    def test_listing_failure_is_not_reported_as_empty_on_oserror(self):
        test_runner = self._import_test_runner()
        with mock.patch.object(
            test_runner.subprocess, "run", side_effect=FileNotFoundError("ctest")
        ):
            self.assertFalse(test_runner.selection_is_empty(["ctest", "-L", "^x$"]))


if __name__ == "__main__":
    unittest.main()
