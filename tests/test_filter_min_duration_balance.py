from __future__ import annotations

import unittest

import numpy as np

from DataLib.preprocess.filter_min_duration_balance import (
    is_stationary,
    root_motion_descriptor,
    select_diverse_stationary,
)


def _qpos(n_frames: int = 181) -> np.ndarray:
    qpos = np.zeros((n_frames, 36), dtype=np.float64)
    qpos[:, 3] = 1.0
    return qpos


class FilterMinDurationBalanceTest(unittest.TestCase):
    def test_stationary_descriptor(self):
        qpos = _qpos()
        qpos[:, 0] = np.linspace(0.0, 0.05, len(qpos))

        descriptor = root_motion_descriptor(
            qpos,
            fps=60.0,
            speed_window_s=0.5,
        )

        self.assertAlmostEqual(
            descriptor["max_horizontal_excursion_m"], 0.05
        )
        self.assertLess(descriptor["p95_window_speed_mps"], 0.02)
        self.assertEqual(descriptor["max_root_rotation_deg"], 0.0)
        self.assertTrue(
            is_stationary(
                descriptor,
                max_excursion_m=0.3,
                max_p95_speed_mps=0.2,
                max_rotation_deg=20.0,
            )
        )

    def test_out_and_back_motion_is_not_stationary(self):
        qpos = _qpos()
        half = len(qpos) // 2
        qpos[:half, 0] = np.linspace(0.0, 1.0, half)
        qpos[half:, 0] = np.linspace(1.0, 0.0, len(qpos) - half)

        descriptor = root_motion_descriptor(
            qpos,
            fps=60.0,
            speed_window_s=0.5,
        )

        self.assertGreater(descriptor["max_horizontal_excursion_m"], 0.9)
        self.assertFalse(
            is_stationary(
                descriptor,
                max_excursion_m=0.3,
                max_p95_speed_mps=0.2,
                max_rotation_deg=20.0,
            )
        )

    def test_in_place_turn_is_not_stationary(self):
        qpos = _qpos()
        angles = np.linspace(0.0, np.deg2rad(45.0), len(qpos))
        qpos[:, 3] = np.cos(angles / 2.0)
        qpos[:, 6] = np.sin(angles / 2.0)

        descriptor = root_motion_descriptor(
            qpos,
            fps=60.0,
            speed_window_s=0.5,
        )

        self.assertAlmostEqual(descriptor["max_root_rotation_deg"], 45.0)
        self.assertFalse(
            is_stationary(
                descriptor,
                max_excursion_m=0.3,
                max_p95_speed_mps=0.2,
                max_rotation_deg=20.0,
            )
        )

    def test_diverse_selection_is_reproducible_and_meets_quota(self):
        rows = []
        for index in range(8):
            rows.append(
                {
                    "seg_id": f"seg{index}",
                    "source_recording": f"source{index % 3}",
                    "duration_s": 3.5 + index,
                    "motion_metrics": {
                        "start_xy_m": [float(index % 2), float(index // 2)]
                    },
                }
            )

        first = select_diverse_stationary(
            rows,
            quota=4,
            seed=123,
            xy_cell_m=1.0,
        )
        second = select_diverse_stationary(
            rows,
            quota=4,
            seed=123,
            xy_cell_m=1.0,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(first <= {row["seg_id"] for row in rows})

    def test_zero_moving_quota_rejects_all_stationary(self):
        selected = select_diverse_stationary(
            [
                {
                    "seg_id": "stand",
                    "source_recording": "source",
                    "duration_s": 4.0,
                    "motion_metrics": {"start_xy_m": [0.0, 0.0]},
                }
            ],
            quota=0,
            seed=123,
            xy_cell_m=1.0,
        )
        self.assertEqual(selected, set())


if __name__ == "__main__":
    unittest.main()
