from sklearn.neighbors import KNeighborsClassifier

class NearestNeighbors:
    def __init__(self, k):
        self.k = k
        self.nn = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X):
        self.X = X

    def predict(self, l, Yl):
        scores = self.nn.fit(self.X[l,:], Yl).predict(self.X)
        return scores
