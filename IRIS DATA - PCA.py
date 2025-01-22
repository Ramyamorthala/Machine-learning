import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target

pca = PCA(n_components=2)

x_pca = pca.fit_transform(x)

sns.scatterplot(x=x_pca[:,0],y=x_pca[:,1],
                hue=y,palette='viridis', s=50)
plt.title('PCA:IRIS DATASET')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.legend()
plt.show()

