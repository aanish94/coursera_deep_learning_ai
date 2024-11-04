import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = np.array([[ 99,  -1],
       [ 98,  -1],
       [ 97,  -2],
       [101,   1],
       [102,   1],
       [103,   2]])

plt.plot(X[:,0], X[:,1], 'ro')

# Loading the PCA algorithm
pca_2 = PCA(n_components=2)

# Let's fit the data. We do not need to scale it, since sklearn's implementation already handles it.
pca_2.fit(X)

pca_2.explained_variance_ratio_

X_trans_2 = pca_2.transform(X)

pca_1 = PCA(n_components=1)
pca_1.fit(X)
pca_1.explained_variance_ratio_
X_trans_1 = pca_1.transform(X)

X_reduced_2 = pca_2.inverse_transform(X_trans_2)

plt.plot(X_reduced_2[:,0], X_reduced_2[:,1], 'ro')
X_reduced_1 = pca_1.inverse_transform(X_trans_1)
plt.plot(X_reduced_1[:,0], X_reduced_1[:,1], 'ro')
plt.show()
