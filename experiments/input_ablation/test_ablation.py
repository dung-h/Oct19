import unittest
import numpy as np
import pandas as pd
from scipy import sparse

from run_ablation import space_for
from audit_metadata import ceur_entries, family_tokens
from run_metadata import meta_spaces,linear_kernel_ridge_grid
from sklearn.linear_model import Ridge


class Checks(unittest.TestCase):
    def test_kernel_linear_ridge_matches_primal_with_intercept(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(20,9));z=rng.normal(size=(7,9));y=rng.normal(size=20)+3
        pred,train=linear_kernel_ridge_grid(x@x.T,z@x.T,y,[.01,1.])
        for alpha in [.01,1.]:
            model=Ridge(alpha=alpha,solver='cholesky').fit(x,y)
            np.testing.assert_allclose(pred[alpha],model.predict(z),atol=1e-10)
            np.testing.assert_allclose(train[alpha],model.predict(x),atol=1e-10)

    def test_missing_abstract_does_not_create_fake_vector(self):
        spaces={('specter','title'):(np.array([[1.,0.]]),np.array([[0.,1.]])),
                ('specter','abstract'):(np.zeros((1,2)),np.zeros((1,2)))}
        a,b=space_for(dict(representation='specter',mode='separate',weight=2),spaces)
        np.testing.assert_array_equal(a,[[1,0,0,0]])
        np.testing.assert_array_equal(b,[[0,1,0,0]])

    def test_weight_changes_relative_information_not_row_norm(self):
        spaces={('tfidf','title'):(sparse.csr_matrix([[1.,0.]]),sparse.csr_matrix([[1.,0.]])),
                ('tfidf','abstract'):(sparse.csr_matrix([[0.,1.]]),sparse.csr_matrix([[0.,1.]]))}
        a,_=space_for(dict(representation='tfidf',mode='separate',weight=.5),spaces)
        self.assertAlmostEqual(np.linalg.norm(a.toarray()),1.)
        self.assertAlmostEqual(a[0,3]/a[0,0],.5)

    def test_publisher_authors_not_adjacent_paper_authors(self):
        html='<li><span class="CEURTITLE">First</span><span class="CEURAUTHOR">A. One</span></li><li><span class="CEURTITLE">Second</span><span class="CEURAUTHOR">B. Two</span></li>'
        rows=ceur_entries(html,'https://ceur-ws.org/Vol-1/')
        self.assertEqual(rows[0]['authors'],'A. One');self.assertEqual(rows[1]['authors'],'B. Two')
        self.assertEqual(family_tokens('Stefania Costantini, Andrea Formisano'),{'costantini','formisano'})

    def test_metadata_transform_does_not_learn_validation_authors_or_year(self):
        data=pd.DataFrame(dict(venue=['cav','cav','new']))
        audit=pd.DataFrame(dict(author_tokens_original=['one','two','unseen'],author_tokens_clean=['one','two','unseen'],
                                year_original=[2000,2002,2050],year_clean=[2000,2002,2050]))
        blocks=meta_spaces(data,audit,np.array([0,1]),np.array([2]))
        self.assertEqual(blocks['clean_authors'][1].nnz,0)
        self.assertEqual(blocks['venue'][1].nnz,0)
        self.assertAlmostEqual(blocks['clean_year'][1].toarray()[0,0],12.25)


if __name__=='__main__':unittest.main()
