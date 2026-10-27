"""Label-free B0 parser and boundary contracts."""
import unittest
import run


class AuthorTopicContracts(unittest.TestCase):
    def test_full_names_are_normalized_without_surname_merges(self):
        first = run.canonical_name("  Émilie   Duval ")
        second = run.canonical_name("Émilie Duval")
        self.assertEqual(first, second)
        self.assertNotEqual(run.canonical_name("Smith John"), run.canonical_name("Smith Jane"))
        self.assertNotEqual(run.canonical_name("J Smith"), run.canonical_name("John Smith"))

    def test_missing_and_uncertain_authors_stay_separate(self):
        self.assertEqual(run.parse_authors("")["status"], "missing")
        row = run.parse_authors("J. Smith, Alex Morgan")
        self.assertEqual(row["status"], "known")
        self.assertEqual([a["display"] for a in row["authors"]], ["Alex Morgan"])
        self.assertEqual([a["display"] for a in row["uncertain"]], ["J. Smith"])

    def test_parser_only_splits_commas_and_deduplicates_exact_normalized_names(self):
        row = run.parse_authors("Alice Person,  Bob Person, alice person")
        self.assertEqual([a["display"] for a in row["authors"]], ["Alice Person", "Bob Person"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
