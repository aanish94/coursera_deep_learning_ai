import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("utils/data/toy_dataset.csv")

# Check correlation of columns
# This may take 1 minute to run
corr = df.corr()

## This will show all the features that have correlation > 0.5 in absolute value. We remove the features 
## with correlation == 1 to remove the correlation of a feature with itself

mask = (abs(corr) > 0.5) & (abs(corr) != 1)
corr.where(mask).stack().sort_values()

# This doesn't show much. Use PCA now
# Loading the PCA object
pca = PCA(n_components = 2) # Here we choose the number of components that we will keep.
X_pca = pca.fit_transform(df)
df_pca = pd.DataFrame(X_pca, columns = ['principal_component_1','principal_component_2'])

plt.scatter(df_pca['principal_component_1'],df_pca['principal_component_2'], color = "#C00000")
plt.xlabel('principal_component_1')
plt.ylabel('principal_component_2')
plt.title('PCA decomposition')
plt.show()

# pca.explained_variance_ration_ returns a list where it shows the amount of variance explained by each principal component.
print(sum(pca.explained_variance_ratio_))

# Can see well defined clusters and per variance ratio - only kept 14.6% of the variance