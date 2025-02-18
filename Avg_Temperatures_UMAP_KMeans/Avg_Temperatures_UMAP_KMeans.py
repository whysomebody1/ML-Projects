import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import plotly.figure_factory as ff
import umap # use 'pip install umap-learn' or 'conda install -c conda-forge umap-learn'


# Importing dataset and examining it
dataset = pd.read_csv("Avg_Temperatures.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# Defining feature set
X = dataset.drop(['Regions'], axis = 1) # Features
print(type(X))
print(X.shape)


# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)


# Implementing UMAP to visualize dataset
u = umap.UMAP(n_components = 2, n_neighbors=15, min_dist=0.4)
x_umap = u.fit_transform(X_scaled)

regions=list(dataset['Regions'])
data = [go.Scatter(x=x_umap[:,0], y=x_umap[:,1], mode='markers',
                    marker = dict(color=None, colorscale='Rainbow', opacity=0.5),
                                text=[f'Region: {a}' for a in regions],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
fig.show()

# Labelling clusters using KMeans
kmeans = KMeans(n_clusters = 5)
kmeans.fit(x_umap)

labels = list(kmeans.labels_)
data = [go.Scatter(x=x_umap[:,0], y=x_umap[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'Region: {a}<br>Label: {b}' for a,b in list(zip(regions,labels))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
fig.show()

dataset['Label'] = kmeans.labels_
dataset.to_csv("ClusteredRegions.csv", index=False)
print(dataset.Label.value_counts())