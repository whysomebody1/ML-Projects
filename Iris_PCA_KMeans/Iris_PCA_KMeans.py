import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Importing dataset and examining it
dataset = pd.read_csv("iris.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# Converting Categorical features into Numerical features
dataset['Species'] = dataset['Species'].map({'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3})
print(dataset.info())

# Dividing dataset into label and feature sets
X = dataset.drop('Species', axis = 1) # Features
Y = dataset['Species'] # Labels
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# # Implementing PCA to visualize dataset
pca = PCA(n_components = 2)
pca.fit(X_scaled)
x_pca = pca.transform(X_scaled)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))

digits=list(dataset['Species'])
data = [go.Scatter(x=x_pca[:,0], y=x_pca[:,1], mode='markers',
                    marker = dict(color=Y, colorscale='Rainbow', opacity=0.5),
                                text=[f'digit: {a}' for a in digits],
                                hoverinfo='text')]

layout = go.Layout(title = 'Data with Original Labels', width = 500, height = 500,
                    xaxis = dict(title='First Principal Component'),
                    yaxis = dict(title='Second Principal Component'))
fig = go.Figure(data=data, layout=layout)
fig.show()

# Implementing K-Means CLustering on dataset and visualizing clusters
kmeans = KMeans(n_clusters = 3)
kmeans.fit(x_pca)
labels=list(kmeans.labels_)
data = [go.Scatter(x=x_pca[:,0], y=x_pca[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'label: {a}' for a in labels],
                                hoverinfo='text')]

layout = go.Layout(title = 'Data with KMeans Labels', width = 500, height = 500,
                    xaxis = dict(title='First Principal Component'),
                    yaxis = dict(title='Second Principal Component'))
fig = go.Figure(data=data, layout=layout)
fig.show()


# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(x_pca)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()