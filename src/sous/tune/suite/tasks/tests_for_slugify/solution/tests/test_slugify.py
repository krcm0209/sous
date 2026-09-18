import unittest

from slugify import slugify


class SlugifyTests(unittest.TestCase):
    def test_letters_are_lower_cased(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_a_run_of_non_alphanumerics_becomes_one_dash(self):
        self.assertEqual(slugify("a  --  b!!c"), "a-b-c")

    def test_leading_and_trailing_dashes_are_removed(self):
        self.assertEqual(slugify("  hello! "), "hello")

    def test_the_slug_is_cut_to_max_length(self):
        self.assertEqual(slugify("abcdefghij", max_length=5), "abcde")

    def test_a_dash_left_by_the_cut_is_removed(self):
        self.assertEqual(slugify("abc def", max_length=4), "abc")
