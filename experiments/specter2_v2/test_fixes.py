import importlib.util
import sys
import unittest

import numpy as np

from prepare_data import ROOT, clean_latex, validate_crossref
from run_v2 import assemble, evaluate, make_priorities

spec = importlib.util.spec_from_file_location("pdf_extractor", ROOT / "extract_openaire_pdf_abstracts.py")
pdf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdf)


class DataFixTest(unittest.TestCase):
    def test_remove_preambles_preserve_formula_and_prose(self):
        text = r"We solve \documentclass[12pt]{minimal}\usepackage{amsmath}\setlength{\oddsidemargin}{-69pt}\begin{document}$$x^2 + y$$\end{document} efficiently."
        cleaned = clean_latex(text)
        self.assertIn("x^2 + y", cleaned)
        self.assertIn("efficiently.", cleaned)
        self.assertNotIn("usepackage", cleaned)
        self.assertEqual(cleaned, clean_latex(cleaned))

    def test_stop_pdf_abstract_before_unnumbered_body_section(self):
        abstract = "We give an example showing that causal intervention does not preserve mutual exclusivity in an annotated disjunction."
        text = "Paper title\nAbstract\n" + abstract + "\nBug Report\nThe rest is body text, not an abstract.\n"
        self.assertEqual(pdf.extract_abstract(text), abstract.rstrip("."))

    def test_related_topic_does_not_validate_a_different_pdf_title(self):
        title="Certified MaxSAT Solving with Pseudo Boolean Proof Logging"
        self.assertFalse(pdf.title_ok(title,"Certified Model Checking and Verification of Distributed Systems"))
        self.assertTrue(pdf.title_ok(title,title+"\nAuthors\nAbstract"))

    def test_same_title_does_not_validate_a_dissertation(self):
        row = {"title":"A paper", "year":"2016", "dataset_doi":"https://www.semanticscholar.org/paper/a", "abstract_url":"https://doi.org/10.1234/thesis"}
        flags, exclude = validate_crossref(row, {"message":{"title":["A paper"], "type":"dissertation"}})
        self.assertTrue(exclude)
        self.assertIn("wrong_publication_type_dissertation", flags)

    def test_later_version_requires_review_when_source_identifier_differs(self):
        row = {"title":"A paper", "year":"2016", "dataset_doi":"https://www.semanticscholar.org/paper/a", "abstract_url":"https://doi.org/10.1234/journal"}
        _, exclude = validate_crossref(row, {"message":{"title":["A paper"], "type":"journal-article", "published":{"date-parts":[[2020]]}}})
        self.assertTrue(exclude)


class SelectionModelTest(unittest.TestCase):
    def test_two_stage_selection_and_rank_are_separate(self):
        config = {"kind":"two_stage", "space":"clean", "C":10.0}
        gain, probability = assemble(config, {"clean_prob_10.0":np.array([.8,.4,.7]), "clean_conditional_gain":np.array([.5,1,.9])})
        priority = make_priorities(gain, probability, .5)
        self.assertEqual(priority[1], 0)
        self.assertGreater(priority[2], priority[0])
        result = evaluate([4,1,5],priority,["a","b","c"])
        self.assertEqual(result["FP"],0)
        self.assertEqual(result["FN"],0)

    def test_zero_priority_excluded_and_text_id_tie_break(self):
        result = evaluate([5,1,4],[1,1,0],["2","10","3"])
        self.assertEqual(result["selected"],2)
        self.assertEqual(result["coverage"],.5)
        self.assertAlmostEqual(result["relevance"],.6309297535714575/1.6309297535714575)


if __name__=="__main__": unittest.main()
