import os
import pandas as pd
import numpy as np
from tqdm import trange
from time import sleep

#generates a dictionary based on the residues in the topology file to map residues based on their position in the system
def get_index2seq_dic(top):
    dic = {}
    for residue in top.residues:
        dic[residue.index] = residue.resSeq
    return dic

#renames columns in the df for creating the distance/interaction matrix
def recolumn(df, top):
    i2seq = get_index2seq_dic(top)
    new = []
    for column in df.columns:
        i, j = tuple(map(int, column.split('_')))
        new.append('{}_{}'.format(i2seq[i], i2seq[j]))
    df.columns = new
    return df

#builds a dictionary of dfs containing the interaction distances indexed by the column names
def read_pairdata(dist_folder, n_chains, top, skip=None):
    done = []
    pairs = {}
    for m in range(0, n_chains):
        for n in range(0, n_chains):
            if m == n:
                continue
            if ((m, n) in done) or ((n, m) in done):
                continue
            chpair = '{}_{}'.format(m,n)
            filename = os.path.join(dist_folder, f'dist_chain{m}_chain{n}.csv')
            data = pd.read_csv(filename, index_col=0)
            data = recolumn(data, top)
            if (skip is not None) and (isinstance(skip, int)):
                data = data[::skip]
            for column in data.columns:
                if column not in pairs.keys():
                    pairs[column] = pd.DataFrame()
                pairs[column][chpair] = data[column]
            done.append((m,n))
            done.append((n,m))
    return pairs

#builds an empty NxN interaction matrix with N being the number of residues
def get_blank_matrix(top):
    mat = {}
    for ri in top.residues:
        if ri.resSeq not in mat.keys():
            mat[ri.resSeq] = {}
        for rj in top.residues:
            if rj.resSeq not in mat[ri.resSeq].keys():
                mat[ri.resSeq][rj.resSeq] = np.nan
    return mat

#exactly what it says, writes the weights with their pairs and times to a single csv file
def write_weights_for_clustering(cluster_path):
    weights = []
    times = []
    pairs = []
    for b in range(0,20000,2000):
        time_pairs = []
        e = b +2000
        inp = os.path.join(cluster_path, 'weights_raw.{}.{}.npy'.format(b,e))
        mat = np.load(inp)
        ndx = [i for i in range(len(mat))]
        t = '{}_{}'.format(b,e)
        for i in ndx:
            for j in ndx:
                sort_pair = sorted([i,j])
                pair = '{}_{}'.format(*sort_pair)
                if pair in time_pairs:
                    continue
                time_pairs.append(pair)
                times.append(t)
                weights.append(mat[i][j])
        pairs.extend(time_pairs)
    df = pd.DataFrame([pairs, times, weights])
    df = df.T
    df.columns = columns = ['pair','times', 'weight']
    out = os.path.join(cluster_path, 'all_weights_raw.csv')
    df.to_csv(out)
    print('Wrote {}\n'.format(out))

#changes weights above the cutoff to be zero, excluding them from later calculations. Includes a progress bar
def weight_pair_data(pairs, top, cutoff=0.6, start=0, stop=-1, progress=True):
    weights = []
    for i in trange(min=0, max=len(pairs.keys())): ##NOT SURE IF THIS IS CORRECT MAY HAVE TO CHANGE (progress bar piece)
        sleep(1)
    for pair, df in pairs.items():
        data = df.iloc[start:stop,:].to_numpy()[0]
        data = data[data <= cutoff]
        if len(data) == 0:
            weights.append(0)
        else:
            weights.append(len(data)/np.mean(data))
    return np.array(weights)