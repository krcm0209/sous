import unittest

from intervals import merge


class MergeTests(unittest.TestCase):
    def test_disjoint_intervals_stay_apart(self):
        self.assertEqual(merge([[1, 2], [4, 5]]), [[1, 2], [4, 5]])

    def test_overlapping_intervals_merge(self):
        self.assertEqual(merge([[1, 4], [2, 6]]), [[1, 6]])

    def test_touching_intervals_merge(self):
        self.assertEqual(merge([[1, 3], [3, 5]]), [[1, 5]])
