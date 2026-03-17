import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pattern_study_batch


class PatternStudyBatchTests(unittest.TestCase):
    def tearDown(self):
        pattern_study_batch._ACTIVE_PROCS.clear()
        pattern_study_batch._reset_shutdown_state()

    def _valid_tile_spec(self) -> dict[str, object]:
        return {"pattern": "+[0,0]-[]", "to_play": "red", "cells": []}


    def test_run_one_reuses_existing_valid_tile_json(self):
        with TemporaryDirectory() as td:
            td_path = Path(td)
            out_dir = td_path / "batch"
            target = pattern_study_batch.StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,k11")
            output_json = pattern_study_batch._target_json_path(out_dir=out_dir, target=target)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(self._valid_tile_spec()), encoding="utf-8")
            with mock.patch("pattern_study_batch.subprocess.run") as run_mock:
                row = pattern_study_batch._run_one(index=1, target=target, out_dir=out_dir, repo_root=td_path)
        run_mock.assert_not_called()
        self.assertTrue(row["ok"])
        self.assertTrue(row["reused_existing"])
        self.assertEqual(row["pattern"], "+[0,0]-[]")

    def test_run_one_reruns_when_existing_tile_json_is_invalid(self):
        with TemporaryDirectory() as td:
            td_path = Path(td)
            out_dir = td_path / "batch"
            target = pattern_study_batch.StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,k11")
            output_json = pattern_study_batch._target_json_path(out_dir=out_dir, target=target)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text("{}", encoding="utf-8")
            valid_tile_spec = self._valid_tile_spec

            class FakeProc:
                def __init__(self):
                    self.pid = 43210
                    self.returncode = 0
                    self.stdout = iter(["[2026-03-15 14:00:00] child log\n"])

                def wait(self):
                    output_json.write_text(json.dumps(valid_tile_spec()), encoding="utf-8")
                    return self.returncode

                def poll(self):
                    return self.returncode

            def fake_popen(cmd, cwd, stdout, stderr, text, bufsize, start_new_session):
                self.assertTrue(start_new_session)
                self.assertEqual(stdout, pattern_study_batch.subprocess.PIPE)
                self.assertEqual(stderr, pattern_study_batch.subprocess.STDOUT)
                self.assertEqual(bufsize, 1)
                self.assertEqual(cmd[0:2], ["python3", "-u"])
                output_json.write_text(json.dumps(self._valid_tile_spec()), encoding="utf-8")
                return FakeProc()

            with mock.patch("pattern_study_batch.subprocess.Popen", side_effect=fake_popen) as run_mock:
                row = pattern_study_batch._run_one(index=1, target=target, out_dir=out_dir, repo_root=td_path)
        run_mock.assert_called_once()
        self.assertTrue(row["ok"])
        self.assertFalse(row["reused_existing"])
        self.assertEqual(row["pattern"], "+[0,0]-[]")
        self.assertNotIn("stdout", row)
        self.assertNotIn("stderr", row)
        self.assertNotIn("command", row)
        self.assertIsInstance(row["runtime_seconds"], float)
        self.assertTrue(row["completed_at"])

    def test_terminate_active_procs_kills_live_process_groups(self):
        live = mock.Mock()
        live.pid = 111
        live.poll.side_effect = [None, None]
        done = mock.Mock()
        done.pid = 222
        done.poll.return_value = 0
        pattern_study_batch._ACTIVE_PROCS[111] = live
        pattern_study_batch._ACTIVE_PROCS[222] = done
        with (
            mock.patch("pattern_study_batch.os.killpg") as killpg_mock,
            mock.patch.object(live, "wait") as wait_live,
            mock.patch.object(done, "wait") as wait_done,
        ):
            pattern_study_batch._terminate_active_procs()
        killpg_mock.assert_any_call(111, pattern_study_batch.signal.SIGTERM)
        killpg_mock.assert_any_call(111, pattern_study_batch.signal.SIGKILL)
        wait_live.assert_called_once_with(timeout=2.0)
        wait_done.assert_called_once_with(timeout=2.0)







if __name__ == "__main__":
    unittest.main()
