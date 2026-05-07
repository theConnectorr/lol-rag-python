# tests/test_text_utils.py
import unittest
from src.core.text_utils import flatten_toc

class TestTextUtils(unittest.TestCase):
    def test_flatten_toc(self):
        sample_toc = [
            {"title": "Early Life", "textContent": "Born in Demacia."},
            {"title": "Abilities", "textContent": "", "children": [
                {"title": "Q", "textContent": "Strikes hard."}
            ]}
        ]
        result = flatten_toc(sample_toc)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["section"], "Early Life")
        self.assertEqual(result[1]["section"], "Q")
        self.assertEqual(result[1]["text"], "Strikes hard.")

if __name__ == "__main__":
    unittest.main()
