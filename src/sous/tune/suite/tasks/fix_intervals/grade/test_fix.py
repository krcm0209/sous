import unittest


class FixTests(unittest.TestCase):
    def test_a_touching_pair_merges(self):
        from intervals import merge

        self.assertEqual(merge([[1, 3], [3, 5]]), [[1, 5]])
        self.assertEqual(merge([[1, 2], [4, 5]]), [[1, 2], [4, 5]])

    def test_a_touching_chain_merges_whatever_the_input_order(self):
        from intervals import merge

        self.assertEqual(merge([[5, 7], [1, 3], [3, 5]]), [[1, 7]])
        self.assertEqual(merge([[1, 4], [2, 6]]), [[1, 6]])

    def test_touching_beside_disjoint(self):
        from intervals import merge

        self.assertEqual(merge([[1, 2], [2, 3], [5, 6]]), [[1, 3], [5, 6]])
