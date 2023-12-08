from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

### ------------------------------------------------------------------ ###
#                                 Scaling                                #
### ------------------------------------------------------------------ ###
class scaler:
    def __init__(self, x_in):
        self.scaler = MinMaxScaler().fit(x_in)

    def scale_data(self, data):
        return self.scaler.transform(data)

    def scale(self, x):
        return self.scaler.transform(x.reshape(1, -1))[0]

    def scale_back(self, x):
        return self.scaler.inverse_transform(x.reshape(1, -1))[0]


### ------------------------------------------------------------------ ###
#                        Dimensionality Reduction                        #
### ------------------------------------------------------------------ ###
class pca:
    def __init__(self, x_in, explained_variance_ratio: float = 0.95):
        i = 1
        ratio = [0]
        while i <= len(x_in) and sum(ratio) <= explained_variance_ratio:
            self.pca = PCA(n_components=i)
            self.pca.fit(x_in)
            ratio = self.pca.explained_variance_ratio_
            i += 1

    def reduce_data(self, data):
        return self.pca.transform(data)

    def reduce_dimensions(self, x):
        return self.pca.transform(x.reshape(1, -1))[0]