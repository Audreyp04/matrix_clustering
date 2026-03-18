import numpy as np
import silhouette
import prep_matrix

prep_matrix.flags2variables()
prep_matrix.get_index2seq_dic()
prep_matrix.recolumn()
prep_matrix.read_pairdata()
prep_matrix.get_blank_matrix()
prep_matrix.write_weights_for_clustering()
prep_matrix.weight_pair_data()



X = np.load('weights.npy')
S = silhouette.do_silhouette(X)
df = silhouette.do_scatterplot(silhouette.best_n, X)
traj = prep_matrix.traj
out = prep_matrix.out
silhouette.write_files(out, df, traj)