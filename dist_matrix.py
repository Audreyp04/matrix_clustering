import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import mdtraj
from mdtraj.core.trajectory import Trajectory
from mdtraj.core.topology import Topology
import numpy as np
import os
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')