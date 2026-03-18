import prep_matrix
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def do_silhouette(X):
    matrix=distance_matrix(X)
    pca = PCA(n_components=2)
    kmeans_X = pca.fit_transform(matrix)

    for n_clusters in range(2,20):
        model = AgglomerativeClustering(n_clusters=n_clusters, metric = 'euclidean', linkage='ward')
        y = model.fit_predict(kmeans_X)
        silhouette = silhouette_score(kmeans_X, y)
        #diverging from kelsie a bit here, i am going to have the script automatically pick the best N value based on the silhouette score rather than having to manually check
        best_n=n_clusters(np.argmax(silhouette))
        print(f'{best_n}')
    return kmeans_X, best_n

def do_scatterplot(n_clusters, X):
    model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    y = model.fit_predict(X)
    df = pd.DataFrame(X, columns=['pc1','pc2'])
    df['label'] = y
    times = [i for i in range(0,X.shape[0])]
    df['frame'] = times
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    for label in df['label'].unique():
        data=df[df['label'] == label]
        ax.scatter(data['pc1'], data['pc2'], label='{} ({} frames0)'.format(label, len(data)))
    ax.legend()
    plt.savefig('cluster_plot.png')
    return df

def write_files(out_path, df, traj):
    for label in df['label'].unique():
        data = df[df['label'] == label]
        x = data['pc1'].sum() / len(data)
        y = data['pc2'].sum() / len(data)
        xy = data[['pc1', 'pc2']].to_numpy()
        dist = np.sqrt((xy[:,0]-x)**2 + (xy[:,1]-y)**2)
        d = data.reset_index(drop=True)
        ndx = int(d.loc[np.argmin(dist), :]['frame'])
        frame = traj[ndx]
        out = os.path.join(out_path, 'cluster{}.pdb'.format(str(label).zfill(3)))
        frame.save_pdb(out)
        print('Wrote {}'.format(out))

    out = os.path.join(out_path, 'cluster-size.dat')
    with open(out, 'w') as f:
        f.write('Cluster          N      Percent\n')
        for label in sorted(df['label'].unique()):
            data = df[df['label'] == label]
            label_zfill = str(label).zfill(3)
            n = len(data)
            percentage=(n/len(df))*100
            line = '{:.3s}\t{:>10d}\t{:>3.2f}\n'.format(label_zfill,n,percentage)
            f.write(line)
    print('Wrote {}'.format(out))