import dataclasses
import unittest

FULL = {"id": 1, "name": "a", "active": True, "tags": ["x"], "score": 2.5}


class RecordTests(unittest.TestCase):
    def test_round_trip(self):
        from record import Record

        record = Record.from_dict(FULL)
        self.assertEqual(record.to_dict(), FULL)
        self.assertEqual(Record.from_dict(record.to_dict()), record)

    def test_optional_fields_default(self):
        from record import Record

        record = Record.from_dict({"id": 2, "name": "b", "active": False})
        self.assertEqual(record.tags, [])
        self.assertIsNone(record.score)

    def test_a_missing_required_key_is_a_valueerror(self):
        from record import Record

        with self.assertRaises(ValueError):
            Record.from_dict({"id": 3})

    def test_it_is_a_dataclass_with_the_schema_fields(self):
        from record import Record

        self.assertTrue(dataclasses.is_dataclass(Record))
        self.assertEqual(
            [f.name for f in dataclasses.fields(Record)], ["id", "name", "active", "tags", "score"]
        )
