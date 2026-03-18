import numpy as np
import silhouette
import prep_matrix

X = np.load('weights.npy')
S = silhouette.do_silhouette(X)
df = silhouette.do_scatterplot(silhouette.best_n, X)
traj = prep_matrix.traj
out = prep_matrix.out
silhouette.write_files(out, df, traj)