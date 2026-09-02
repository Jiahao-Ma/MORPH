from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import numpy as np

from DataLib.preprocess.filter_backward_motions import scan_motion, subset_g1


def _frame(move: tuple[float, float]) -> bytes:
    return (
        json.dumps(
            {
                "t": 0.0,
                "dt": 1.0 / 60.0,
                "cmd": {"move": list(move), "crouch": False},
                "root": {},
                "joints": [],
            }
        ).encode()
        + b"\n"
    )


class FilterBackwardMotionsTest(unittest.TestCase):
    def test_rejects_whole_motion_for_one_s_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "motion_frames.jsonl"
            path.write_bytes(
                _frame((0.0, 1.0))
                + _frame((0.0, 0.0))
                + _frame((0.0, -1.0))
                + _frame((0.0, 1.0))
            )

            result = scan_motion(path, eps_back=0.2)

        self.assertEqual(result["frame_count"], 4)
        self.assertEqual(result["backward_frames"], 1)
        self.assertEqual(result["first_backward_frame"], 2)
        self.assertEqual(result["last_backward_frame"], 2)
        self.assertEqual(result["min_forward_cmd"], -1.0)

    def test_ignores_small_forward_axis_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "motion_frames.jsonl"
            path.write_bytes(
                _frame((0.0, 1.0))
                + _frame((0.0, -0.01))
                + _frame((0.0, 0.0))
            )

            result = scan_motion(path, eps_back=0.2)

        self.assertEqual(result["frame_count"], 3)
        self.assertEqual(result["backward_frames"], 0)

    def test_subsets_existing_g1_without_retargeting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            input_root = root / "g1_input"
            input_root.mkdir()
            rows = []
            for seg_id in ("keep", "reject"):
                qpos_path = input_root / f"{seg_id}_frames.npy"
                np.save(qpos_path, np.zeros((2, 36), dtype=np.float32))
                rows.append(
                    {
                        "seg_id": seg_id,
                        "category": "stairs",
                        "n_frames": 2,
                        "qpos_npy": str(qpos_path),
                    }
                )
            input_manifest = input_root / "segments_manifest.jsonl"
            input_manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            stage2_jsonl = root / "keep_frames.jsonl"
            stage2_meta = root / "keep_meta.json"
            stage2_jsonl.touch()
            stage2_meta.write_text("{}", encoding="utf-8")
            accepted = [
                {
                    "seg_id": "keep",
                    "category": "stairs",
                    "n_frames": 2,
                    "washed_jsonl": str(stage2_jsonl),
                    "washed_meta": str(stage2_meta),
                }
            ]

            result = subset_g1(
                input_manifest=input_manifest,
                output_root=root / "g1_output",
                accepted_stage2=accepted,
                rejected_ids={"reject"},
            )

            output_rows = [
                json.loads(line)
                for line in pathlib.Path(result["manifest"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            output_qpos = np.load(output_rows[0]["qpos_npy"])

        self.assertEqual(result["output_motions"], 1)
        self.assertEqual(result["rejected_motions"], 1)
        self.assertEqual(output_qpos.shape, (2, 36))
        self.assertEqual(output_rows[0]["status"], "qa_ok_forward_only")


if __name__ == "__main__":
    unittest.main()
