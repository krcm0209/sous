import unittest


class TokenizeTests(unittest.TestCase):
    def test_splits_on_whitespace_and_blank_is_empty(self):
        from rpn import tokenize

        self.assertEqual(tokenize("3  4 +\n"), ["3", "4", "+"])
        self.assertEqual(tokenize("   "), [])


class EvaluateTests(unittest.TestCase):
    def test_addition(self):
        from rpn import evaluate

        self.assertEqual(evaluate(["3", "4", "+"]), 7)

    def test_nested_expression(self):
        from rpn import evaluate

        self.assertEqual(evaluate(["5", "1", "2", "+", "4", "*", "+", "3", "-"]), 14)

    def test_operand_order_for_subtraction_and_division(self):
        from rpn import evaluate

        self.assertEqual(evaluate(["10", "3", "-"]), 7)
        self.assertEqual(evaluate(["7", "2", "/"]), 3)

    def test_floor_division_of_a_negative(self):
        from rpn import evaluate

        self.assertEqual(evaluate(["-7", "2", "/"]), -4)

    def test_errors(self):
        from rpn import evaluate

        with self.assertRaises(ValueError):
            evaluate(["1", "x", "+"])
        with self.assertRaises(ValueError):
            evaluate(["+"])
        with self.assertRaises(ValueError):
            evaluate(["1", "2"])
