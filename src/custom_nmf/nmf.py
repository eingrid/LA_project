import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from scipy.sparse.linalg import svds
# from update_rules import update_H, update_W

class OptimizedNMF(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, max_iter=200, init='random', random_seed=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.init = init  # 'random' or 'svd'
        self.random_seed = random_seed

    def _initialize_nmf(self, X):
        n_samples, n_features = X.shape
        np.random.seed(self.random_seed)

        if self.init == 'svd':
            if sp.issparse(X):
                U, S, Vt = svds(X, k=self.n_components)
            else:
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
            W = np.maximum(U[:, :self.n_components], 0)
            H = np.maximum(Vt[:self.n_components, :], 0)
        elif self.init == 'random':
            W = np.abs(np.random.randn(n_samples, self.n_components))
            H = np.abs(np.random.randn(self.n_components, n_features))
        else:
            raise ValueError("Invalid init parameter")
        return W, H

    def _update_H(self, X, W, H):
        # update_H(W, H, X) 
        numerator = safe_sparse_dot(W.T, X)
        denominator = safe_sparse_dot(safe_sparse_dot(W.T, W), H) + 1e-4
        H *= numerator / denominator

    def _update_W(self, X, W, H):
        # update_W(W, H, X)
        numerator = safe_sparse_dot(X, H.T)
        denominator = safe_sparse_dot(W, safe_sparse_dot(H, H.T)) + 1e-4
        W *= numerator / denominator

    def fit_transform(self, X, y=None):
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float)
        W, H = self._initialize_nmf(X)

        for n_iter in range(self.max_iter):
            self._update_H(X, W, H)
            self._update_W(X, W, H)

        self.components_ = H
        return W

    def transform(self, X):
        check_is_fitted(self, 'components_')
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float)

        W = np.abs(np.random.randn(X.shape[0], self.n_components))
        for n_iter in range(self.max_iter):
            self._update_W(X, W, self.components_)
        return W
