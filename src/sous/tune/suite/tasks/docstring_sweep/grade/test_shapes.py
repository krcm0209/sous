import math
import unittest


class BehaviourTests(unittest.TestCase):
    def test_rectangle(self):
        from shapes import area_rect, perimeter_rect

        self.assertEqual(area_rect(2, 3), 6)
        self.assertEqual(perimeter_rect(2, 3), 10)

    def test_circle(self):
        from shapes import area_circle, perimeter_circle

        self.assertAlmostEqual(area_circle(1), math.pi)
        self.assertAlmostEqual(perimeter_circle(1), 2 * math.pi)

    def test_triangle_and_scale(self):
        from shapes import area_triangle, scale

        self.assertEqual(area_triangle(4, 3), 6)
        self.assertEqual(scale(2, 1.5), 3)

    def test_non_positive_dimensions_are_refused(self):
        from shapes import area_rect, scale

        with self.assertRaises(ValueError):
            area_rect(0, 1)
        with self.assertRaises(ValueError):
            scale(1, -1)
