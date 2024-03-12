import numpy as np

class SimpleNMF:
    def __init__(self, n_components=2, max_iter=1000, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.W = None
        self.components_ = None

    def fit(self, X):
        m, n = X.shape
        self.W = np.abs(np.random.randn(m, self.n_components))
        self.components_ = np.abs(np.random.randn(self.n_components, n))

        for i in range(self.max_iter):
            WH = np.dot(self.W, self.components_)
            loss = np.linalg.norm(X - WH, 'fro')

            if loss < self.tol:
                break

            H_update = self.components_ * (np.dot(self.W.T, X) / np.dot(self.W.T, WH))
            self.components_ = np.where(H_update > 0, H_update, 0)

            W_update = self.W * (np.dot(X, self.components_.T) / np.dot(WH, self.components_.T))
            self.W = np.where(W_update > 0, W_update, 0)

        return self

    def transform(self, X):
        return np.dot(np.linalg.pinv(self.W), X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
