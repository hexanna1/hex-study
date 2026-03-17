import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import pattern_study_batch


class PatternStudyBatchTests(unittest.TestCase):
    def tearDown(self):
        pattern_study_batch._ACTIVE_PROCS.clear()
        pattern_study_batch._reset_shutdown_state()

    def test_resolve_page_grid_defaults_to_square_manual(self):
        self.assertEqual(
            pattern_study_batch._resolve_page_grid(count=13, catalog_mode=False),
            (4, 4),
        )

    def test_resolve_page_grid_defaults_to_square_catalog(self):
        self.assertEqual(
            pattern_study_batch._resolve_page_grid(count=244, catalog_mode=True),
            (10, 10),
        )

    def test_load_existing_tile_spec_accepts_valid_tile_json(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "tile.json"
            path.write_text(json.dumps({"pattern": "+[0,0]-[]", "to_play": "red", "cells": []}), encoding="utf-8")
            spec = pattern_study_batch._load_existing_tile_spec(path)
        self.assertEqual(spec, {"pattern": "+[0,0]-[]", "to_play": "red", "cells": []})

    def test_default_out_dir_uses_catalog_parent(self):
        path = Path("artifacts/interior_patterns_m4_d4_span12/catalog.json")
        out_dir = pattern_study_batch._default_out_dir(catalog_path=path)
        self.assertEqual(out_dir, Path("artifacts") / "interior_patterns_m4_d4_span12")

    def test_load_catalog_targets_uses_hexworld_and_catalog_delta(self):
        with TemporaryDirectory() as td:
            catalog = Path(td) / "catalog.json"
            catalog.write_text(
                json.dumps(
                    {
                        "patterns": [
                            {
                                "pattern": "+[0,0]-[]",
                                "candidate_Δ_max": 7,
                                "hexworld_21": "https://hexworld.org/board/#21c1,k11",
                            },
                            {
                                "pattern": "+[0,0]-[0,1]",
                                "candidate_Δ_max": 5,
                                "hexworld_21": "https://hexworld.org/board/#21c1,k10k11",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            payload = pattern_study_batch._load_catalog_payload(catalog)
            targets = pattern_study_batch._load_catalog_targets(payload, catalog_path=catalog)
        self.assertEqual(
            targets,
            (
                pattern_study_batch.StudyTarget(candidate_Δ_max=7, hexworld="https://hexworld.org/board/#21c1,k11"),
                pattern_study_batch.StudyTarget(candidate_Δ_max=5, hexworld="https://hexworld.org/board/#21c1,k10k11"),
            ),
        )

    def test_target_json_path_uses_stable_slug_only(self):
        out_dir = Path("/tmp/example-batch")
        target = pattern_study_batch.StudyTarget(candidate_Δ_max=12, hexworld="https://hexworld.org/board/#21c1,k11:p")
        path = pattern_study_batch._target_json_path(out_dir=out_dir, target=target)
        self.assertEqual(path, out_dir / "tiles" / "d12-k11_p.json")

    def test_run_one_reuses_existing_valid_tile_json(self):
        with TemporaryDirectory() as td:
            td_path = Path(td)
            out_dir = td_path / "batch"
            target = pattern_study_batch.StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,k11")
            output_json = pattern_study_batch._target_json_path(out_dir=out_dir, target=target)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps({"pattern": "+[0,0]-[]", "to_play": "red", "cells": []}), encoding="utf-8")
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

            class FakeProc:
                def __init__(self):
                    self.pid = 43210
                    self.returncode = 0
                    self.stdout = iter(["[2026-03-15 14:00:00] child log\n"])

                def wait(self):
                    output_json.write_text(json.dumps({"pattern": "+[0,0]-[]", "to_play": "red", "cells": []}), encoding="utf-8")
                    return self.returncode

                def poll(self):
                    return self.returncode

            def fake_popen(cmd, cwd, stdout, stderr, text, bufsize, start_new_session):
                self.assertTrue(start_new_session)
                self.assertEqual(stdout, pattern_study_batch.subprocess.PIPE)
                self.assertEqual(stderr, pattern_study_batch.subprocess.STDOUT)
                self.assertEqual(bufsize, 1)
                self.assertEqual(cmd[0:2], ["python3", "-u"])
                output_json.write_text(json.dumps({"pattern": "+[0,0]-[]", "to_play": "red", "cells": []}), encoding="utf-8")
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

    def test_stream_proc_output_prints_stdout(self):
        proc = mock.Mock()
        proc.stdout = iter(["one\n", "two\n"])
        with mock.patch("builtins.print") as print_mock:
            pattern_study_batch._stream_proc_output(proc)
        print_mock.assert_any_call("one\n", end="")
        print_mock.assert_any_call("two\n", end="")

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

    def test_interrupt_handler_sets_shutdown_state_and_raises(self):
        with self.assertRaises(pattern_study_batch.BatchInterrupted):
            pattern_study_batch._interrupt_handler(pattern_study_batch.signal.SIGINT, None)
        self.assertTrue(pattern_study_batch._SHUTDOWN_REQUESTED.is_set())
        self.assertEqual(pattern_study_batch._SHUTDOWN_SIGNAL, pattern_study_batch.signal.SIGINT)

    def test_interrupt_handler_is_noop_during_cleanup(self):
        pattern_study_batch._CLEANUP_IN_PROGRESS.set()
        try:
            self.assertIsNone(pattern_study_batch._interrupt_handler(pattern_study_batch.signal.SIGTERM, None))
        finally:
            pattern_study_batch._CLEANUP_IN_PROGRESS.clear()

    def test_build_contact_sheet_items_preserves_index_order(self):
        with TemporaryDirectory() as td:
            td_path = Path(td)
            spec_a = {"pattern": "+[0,0]-[]", "to_play": "red", "cells": []}
            spec_b = {"pattern": "+[0,0]-[0,1]", "to_play": "red", "cells": []}
            path_a = td_path / "b.json"
            path_b = td_path / "a.json"
            path_a.write_text(json.dumps(spec_a), encoding="utf-8")
            path_b.write_text(json.dumps(spec_b), encoding="utf-8")
            results = [
                {"index": 2, "candidate_Δ_max": 3, "hexworld": "https://hexworld.org/board/#21c1,k10k11", "output_json": str(path_a), "ok": True},
                {"index": 1, "candidate_Δ_max": 12, "hexworld": "https://hexworld.org/board/#21c1,k11", "output_json": str(path_b), "ok": True},
            ]
            items = pattern_study_batch._build_contact_sheet_items(results)
        self.assertEqual([item["title"] for item in items], ["+[0,0]-[0,1]", "+[0,0]-[]"])
        self.assertEqual(items[0].get("footer"), None)

    def test_main_runs_targets_and_writes_manifest_and_square_page(self):
        with TemporaryDirectory() as td:
            td_path = Path(td)
            args = SimpleNamespace(
                workers=2,
                catalog=None,
                delta=None,
                out_dir=str(td_path / "batch"),
                no_png=False,
            )
            targets = (
                pattern_study_batch.StudyTarget(candidate_Δ_max=12, hexworld="https://hexworld.org/board/#21c1,k11"),
                pattern_study_batch.StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,k10k11"),
            )

            def fake_run_one(*, index, target, out_dir, repo_root):
                output_json = pattern_study_batch._target_json_path(out_dir=out_dir, target=target)
                output_json.parent.mkdir(parents=True, exist_ok=True)
                if index == 1:
                    spec = {"pattern": "+[0,0]-[]", "to_play": "red", "cells": []}
                    output_json.write_text(json.dumps(spec), encoding="utf-8")
                    return {
                        "index": index,
                        "candidate_Δ_max": target.candidate_Δ_max,
                        "hexworld": target.hexworld,
                        "output_json": str(output_json),
                        "returncode": 0,
                        "ok": True,
                        "pattern": spec["pattern"],
                    }
                return {
                    "index": index,
                    "candidate_Δ_max": target.candidate_Δ_max,
                    "hexworld": target.hexworld,
                    "output_json": str(output_json),
                    "returncode": 1,
                    "ok": False,
                }

            write_pages_mock = mock.Mock(return_value=["001.png"])
            with (
                mock.patch("pattern_study_batch._parse_args", return_value=args),
                mock.patch("pattern_study_batch.STUDY_TARGETS", targets),
                mock.patch("pattern_study_batch._run_one", side_effect=fake_run_one),
                mock.patch("pattern_study_batch.sout.write_local_map_contact_sheet_pages", write_pages_mock),
                mock.patch("builtins.print"),
            ):
                rc = pattern_study_batch.main()

            self.assertEqual(rc, 1)
            manifest = json.loads((Path(args.out_dir) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["target_source"], {"kind": "manual", "count": 2})
            self.assertEqual(manifest["targets_total"], 2)
            self.assertEqual(manifest["targets_ok"], 1)
            self.assertEqual(manifest["columns"], 1)
            self.assertEqual(manifest["rows"], 1)
            self.assertEqual(manifest["png_page_count"], 1)
            self.assertEqual(manifest["png_pages"], ["eval_maps/001.png"])
            self.assertNotIn("results", manifest)
            self.assertNotIn("targets_reused_existing", manifest)
            write_pages_mock.assert_called_once()
            self.assertEqual(write_pages_mock.call_args.args[1], Path(args.out_dir) / "eval_maps")
            self.assertEqual(write_pages_mock.call_args.kwargs["columns"], 1)
            self.assertEqual(write_pages_mock.call_args.kwargs["rows_per_page"], 1)
            items = write_pages_mock.call_args.args[0]
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["title"], "+[0,0]-[]")

    def test_main_uses_catalog_targets_with_catalog_selected_delta(self):
        with TemporaryDirectory() as td:
            td_path = Path(td)
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "patterns": [
                            {
                                "pattern": "+[0,0]-[]",
                                "candidate_Δ_max": 7,
                                "hexworld_21": "https://hexworld.org/board/#21c1,k11",
                            },
                            {
                                "pattern": "+[0,0]-[0,1]",
                                "candidate_Δ_max": 5,
                                "hexworld_21": "https://hexworld.org/board/#21c1,k10k11",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                workers=2,
                catalog=str(catalog_path),
                delta=None,
                out_dir=str(td_path / "batch"),
                no_png=False,
            )

            seen_targets = []

            def fake_run_one(*, index, target, out_dir, repo_root):
                seen_targets.append(target)
                output_json = pattern_study_batch._target_json_path(out_dir=out_dir, target=target)
                output_json.parent.mkdir(parents=True, exist_ok=True)
                spec = {"pattern": "+[0,0]-[]", "to_play": "red", "cells": []}
                output_json.write_text(json.dumps(spec), encoding="utf-8")
                return {
                    "index": index,
                    "candidate_Δ_max": target.candidate_Δ_max,
                    "hexworld": target.hexworld,
                    "output_json": str(output_json),
                    "returncode": 0,
                    "ok": True,
                    "pattern": spec["pattern"],
                }

            write_pages_mock = mock.Mock(return_value=["001.png"])
            with (
                mock.patch("pattern_study_batch._parse_args", return_value=args),
                mock.patch("pattern_study_batch._run_one", side_effect=fake_run_one),
                mock.patch("pattern_study_batch.sout.write_local_map_contact_sheet_pages", write_pages_mock),
                mock.patch("builtins.print"),
            ):
                rc = pattern_study_batch.main()

            self.assertEqual(rc, 0)
            self.assertEqual(
                seen_targets,
                [
                    pattern_study_batch.StudyTarget(candidate_Δ_max=7, hexworld="https://hexworld.org/board/#21c1,k11"),
                    pattern_study_batch.StudyTarget(candidate_Δ_max=5, hexworld="https://hexworld.org/board/#21c1,k10k11"),
                ],
            )
            manifest = json.loads((Path(args.out_dir) / "manifest.json").read_text(encoding="utf-8"))
            copied_catalog = json.loads((Path(args.out_dir) / "catalog.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["target_source"],
                {"kind": "catalog", "catalog": str(Path(args.out_dir) / "catalog.json")},
            )
            self.assertEqual(copied_catalog["patterns"], json.loads(catalog_path.read_text(encoding="utf-8"))["patterns"])
            self.assertEqual(manifest["png_page_count"], 1)
            self.assertEqual(manifest["png_pages"], ["eval_maps/001.png"])
            self.assertNotIn("results", manifest)
            self.assertNotIn("targets_reused_existing", manifest)
            write_pages_mock.assert_called_once()
            self.assertEqual(write_pages_mock.call_args.args[1], Path(args.out_dir) / "eval_maps")

    def test_main_catalog_mode_uses_paginated_output(self):
        with TemporaryDirectory() as td:
            td_path = Path(td)
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "patterns": [
                            {
                                "pattern": "+[0,0]-[]",
                                "candidate_Δ_max": 7,
                                "hexworld_21": "https://hexworld.org/board/#21c1,k11",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                workers=2,
                catalog=str(catalog_path),
                delta=3,
                out_dir=str(td_path / "batch"),
                no_png=False,
            )

            def fake_run_one(*, index, target, out_dir, repo_root):
                output_json = pattern_study_batch._target_json_path(out_dir=out_dir, target=target)
                output_json.parent.mkdir(parents=True, exist_ok=True)
                spec = {"pattern": "+[0,0]-[]", "to_play": "red", "cells": []}
                output_json.write_text(json.dumps(spec), encoding="utf-8")
                return {
                    "index": index,
                    "candidate_Δ_max": target.candidate_Δ_max,
                    "hexworld": target.hexworld,
                    "output_json": str(output_json),
                    "returncode": 0,
                    "ok": True,
                    "pattern": spec["pattern"],
                }

            write_pages_mock = mock.Mock(return_value=["001.png", "002.png"])
            with (
                mock.patch("pattern_study_batch._parse_args", return_value=args),
                mock.patch("pattern_study_batch._run_one", side_effect=fake_run_one),
                mock.patch("pattern_study_batch.sout.write_local_map_contact_sheet_pages", write_pages_mock),
                mock.patch("builtins.print"),
            ):
                rc = pattern_study_batch.main()

            self.assertEqual(rc, 0)
            manifest = json.loads((Path(args.out_dir) / "manifest.json").read_text(encoding="utf-8"))
            copied_catalog = json.loads((Path(args.out_dir) / "catalog.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["target_source"],
                {"kind": "catalog", "catalog": str(Path(args.out_dir) / "catalog.json")},
            )
            self.assertEqual(
                copied_catalog,
                {
                    "patterns": [
                        {
                            "pattern": "+[0,0]-[]",
                            "candidate_Δ_max": 3,
                            "hexworld_21": "https://hexworld.org/board/#21c1,k11",
                        },
                    ]
                },
            )
            self.assertEqual(manifest["png_pages"], ["eval_maps/001.png", "eval_maps/002.png"])
            self.assertEqual(manifest["png_page_count"], 2)
            self.assertEqual(write_pages_mock.call_args.args[1], Path(args.out_dir) / "eval_maps")


if __name__ == "__main__":
    unittest.main()
