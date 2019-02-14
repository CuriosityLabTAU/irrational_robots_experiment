import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
psi0 = np.kron(plus, plus).reshape(16,1)

def load_quant_pred(use_U=True, with_mixing=True, use_neutral=False, h_mix_type = 0):
    control_str = '_U_%s_mixing_%s_neutral_%s_mix_type_%d' % (use_U, with_mixing, use_neutral, h_mix_type)
    all_data = pickle.load(open('/home/torr/PycharmProjects/quantum_prediction/data/all_data%s.pkl' % control_str, 'rb'), encoding='latin1')
    q_info = pickle.load(open('/home/torr/PycharmProjects/quantum_prediction/data/q_info%s.pkl' % control_str, 'rb'), encoding='latin1')
    df = pd.read_csv('/home/torr/PycharmProjects/quantum_prediction/data/new_dataframe.csv', index_col=0)
    return all_data, q_info, df

all_data, q_info, df = load_quant_pred()
# all_data, q_info, df = load_quant_pred(False, False, True,0)

participants_list = all_data.keys()

qn, qirr = [], []
for q in list(all_data[list(all_data.keys())[0]].keys())[:-1]:
    for p_id in participants_list:
        try:
            irr = all_data[p_id][q]['p_ab'][0] - np.min((all_data[p_id][q]['p_a'][0], all_data[p_id][q]['p_b'][0]))
        except:
            irr = all_data[p_id][q]['p_ab'] - np.min((all_data[p_id][q]['p_a'], all_data[p_id][q]['p_b']))
        if irr > 0:
            qirr += [1]
        else:
            qirr += [0]
        if 'q_psies' not in locals():
            q_psies = all_data[p_id][q]['psi'].reshape(16, 1)
            # q_psies = psi0
        else:
            q_psies = np.concatenate((q_psies, all_data[p_id][q]['psi'].reshape(16, 1)), axis=1)
            # q_psies = np.concatenate((q_psies, psi0), axis=1)
        qn += [q]

tsne = TSNE(n_components=2, verbose=1, perplexity=80, n_iter=300)
tsne_results = tsne.fit_transform(q_psies.T)

# pca = PCA(n_components=10)
# tsne_results = pca.fit_transform(q_psies.T)

tsne_df = pd.DataFrame()

tsne_df['xt'] = tsne_results[:,0]
tsne_df['yt'] = tsne_results[:,1]

tsne_df['irr'] = qirr
tsne_df['qn'] = qn

fg = sns.FacetGrid(data=tsne_df, hue='qn', hue_order=tsne_df['qn'].unique(), aspect=1.61)
# fg = sns.FacetGrid(data=tsne_df, hue='irr', hue_order=tsne_df['irr'].unique(), aspect=1.61)
fg.map(plt.scatter, 'xt', 'yt').add_legend()

plt.show()
print()