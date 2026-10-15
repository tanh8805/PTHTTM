"""Unit test cho class_map / level_of / frame_stats (không cần best.pt). Chạy: cd backend && python -m unittest tests.test_analyzer"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analyzer as A


def model(names):
    return SimpleNamespace(names=dict(enumerate(names)))


class ClassMapTest(unittest.TestCase):
    def test_three_depth_classes(self):
        cmap = A.class_map(model(["flood", "road", "car_dry", "car_shallow", "car_deep"]))
        self.assertEqual(cmap, {0: ("flood", None), 1: ("road", None), 2: ("car", 0), 3: ("car", 2), 4: ("car", 3)})
        self.assertTrue(A.DEPTH_MODEL)

    def test_order_independent(self):
        cmap = A.class_map(model(["car_deep", "flood", "car_dry", "road", "car_shallow"]))
        self.assertEqual(cmap[0], ("car", 3)); self.assertEqual(cmap[1], ("flood", None)); self.assertEqual(cmap[4], ("car", 2))

    def test_legacy_levels_still_work(self):
        cmap = A.class_map(model(["flood", "road"] + [f"car_L{i}" for i in range(5)]))
        self.assertEqual([cmap[i] for i in range(2, 7)], [("car", k) for k in range(5)])
        self.assertFalse(A.DEPTH_MODEL)

    def test_requires_flood(self):
        with self.assertRaises(ValueError):
            A.class_map(model(["road", "car_dry"]))


class LevelOfTest(unittest.TestCase):
    def test_coverage_caps_at_level_2(self):
        self.assertEqual(A.level_of(0.0, None), 0)
        self.assertEqual(A.level_of(0.04, None), 0)
        self.assertEqual(A.level_of(0.05, None), 1)
        self.assertEqual(A.level_of(0.29, None), 1)
        self.assertEqual(A.level_of(0.30, None), 2)
        self.assertEqual(A.level_of(1.00, None), 2)      # coverage alone never reaches Nguy hiểm
        self.assertIsNone(A.LEVELS[3][2])

    def test_cars_by_depth_class(self):
        self.assertEqual(A.level_of(0.0, A.CAR_DEPTH_LEVEL["car_dry"]), 0)     # car_dry alone is safe
        self.assertEqual(A.level_of(0.0, A.CAR_DEPTH_LEVEL["car_shallow"]), 2)
        self.assertEqual(A.level_of(0.0, A.CAR_DEPTH_LEVEL["car_deep"]), 3)    # only car_deep -> Nguy hiểm
        self.assertEqual(A.level_of(0.40, A.CAR_DEPTH_LEVEL["car_dry"]), 2)    # coverage still counts with a dry car
        self.assertEqual(A.level_of(1.00, A.CAR_DEPTH_LEVEL["car_shallow"]), 2)
        self.assertEqual(A.level_of(0.10, A.CAR_DEPTH_LEVEL["car_deep"]), 3)

    def test_legacy_levels(self):
        self.assertEqual([A.level_of(0.0, k) for k in range(5)], [0, 1, 2, 3, 3])


class RenderNamesTest(unittest.TestCase):
    def test_colors_and_text(self):
        for n in ("car_dry", "car_shallow", "car_deep"):
            self.assertIn(n, A.CLASS_COLORS)
        A.class_map(model(["flood", "road", "car_dry", "car_shallow", "car_deep"]))
        self.assertEqual([A.car_class(l) for l in (0, 2, 3)], ["car_dry", "car_shallow", "car_deep"])
        self.assertEqual([A.car_text(l) for l in (0, 2, 3)], ["Khô", "Nông", "Sâu"])
        self.assertEqual(list(A.legend_classes()), ["flood", "road", "car_dry", "car_shallow", "car_deep"])
        A.class_map(model(["flood", "road", "car_L0", "car_L3"]))
        self.assertEqual(A.car_text(3), "L3")
        self.assertEqual(len(A.legend_classes()), 7)


class NoCarNoteTest(unittest.TestCase):
    def test_banner_and_json_note(self):
        import numpy as np
        h = w = 100
        flood = np.ones((h, w), bool)
        st = A.frame_stats([{"kind": "flood", "level": None, "conf": 1, "mask": flood, "box": None}], h, w)
        self.assertEqual((st["level"], st["max_car_level"]), (2, None))    # fully covered road, no car: level 2, not 3
        ov = A.build_overlay([{"kind": "flood", "level": None, "conf": 1, "mask": flood, "box": None}], h, w)
        out = A.render_frame(np.zeros((h, w, 3), np.uint8), ov, ov, 0, 2, st, no_car=True)
        self.assertEqual(out.shape[1], w)
        self.assertEqual(A.NO_CAR_NOTE, "không có xe để ước lượng độ sâu")


class FrameStatsTest(unittest.TestCase):
    def test_deep_car_gives_danger(self):
        import numpy as np
        h = w = 100
        car = np.zeros((h, w), bool); car[10:20, 10:20] = True
        inst = [{"kind": "car", "level": 3, "conf": .9, "mask": car, "box": np.array([10, 10, 20, 20])}]
        st = A.frame_stats(inst, h, w)
        self.assertEqual(st["level"], 3); self.assertEqual(st["max_car_level"], 3)


if __name__ == "__main__":
    unittest.main()
