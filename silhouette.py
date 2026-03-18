import prep_matrix
import os
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data=np.load('weights.npy')

def do_silhouette(data):
    matrix=distance_matrix(data)