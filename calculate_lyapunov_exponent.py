import os
import mne
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score

data_folder = '/mnt/s3-data2/amiftakhova/OCD/OCD/'
folders = [x for x in os.listdir(data_folder) if os.path.isdir(data_folder + x)]
inside_folders = []
for fldr in folders:
    ins = [fldr + '/' + x for x in os.listdir(data_folder + fldr) if os.path.isdir(data_folder + fldr + '/' + x)]
    if len(ins) > 0:
        inside_folders.extend(ins)
    else:
        inside_folders.append(fldr)
fldr2label = {inside_folders[i]: i for i in range(len(inside_folders))}
label2class = {
    4: 0, 5: 0, 6: 0, # anxiety
    7: 1, 8: 1, 9: 1, # bipolar
    10: 2, # control
    11: 3, 12: 3, 13: 3, # depression
    14: 4, # personality disorder,
    0: 5, 1: 5, 2: 5, 3: 5# stress
}

class2name = {0: 'anxiety', 1: 'bipolar', 2: 'control', 3: 'depression', 4: 'personality disorder', 5: 'stress'}
name2class = {v: k for k, v in class2name.items()}

og_files = []
zg_files = []
og_labels = []
zg_labels = []
for fldr in inside_folders:
    pth =  data_folder + fldr
    og_pths = [x for x in os.listdir(pth) if x.lower().endswith('.edf') and ('og.' in x.lower() or '_ог.' in x.lower() or ' ог.' in x.lower() or 'eo.' in x.lower() or '_eo' in x.lower() or '_of' in x.lower())]
    zg_pths = [x for x in os.listdir(pth) if x.lower().endswith('.edf') and ('zg.' in x.lower() or 'зог.' in x.lower() or 'зг.' in x.lower() or 'ec.' in x.lower() or 'fon.' in x.lower() or '_ec' in x.lower() or 'eс.' in x.lower())]
    left = [x for x in os.listdir(pth) if x not in og_pths and x not in zg_pths]
    if len(left) > 0:
        print(pth, left)
    for f in og_pths:
        og_files.append(pth + '/' + f)
        og_labels.append(fldr2label[fldr])
    for f in zg_pths:
        zg_files.append(pth + '/' + f)
        zg_labels.append(fldr2label[fldr])

print(f'Number of files for open eyes: {len(og_files)}')
print(f'Number of files for closed eyes: {len(zg_files)}')

# ensured that og_files[i] is the pair for zg_files[i]

lengths_og = []
lengths_zg = []
for i in tqdm(range(len(og_files))):
    og_file = og_files[i]
    zg_file = zg_files[i]
    try:
        og_sample = mne.io.read_raw_edf(og_file, verbose=False)
        og_len = 1.0 * len(og_sample) / og_sample.info['sfreq']
        zg_sample = mne.io.read_raw_edf(zg_file, verbose=False)
        zg_len = 1.0 * len(zg_sample) / zg_sample.info['sfreq']
        lengths_og.append(og_len)
        lengths_zg.append(zg_len)
    except Exception as e:
        print(f, e)
        lengths_og.append(0)
        lengths_zg.append(0)

ids = np.argwhere(np.array(lengths_zg) >= 20.0).flatten()

from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed

def lyapunov_exponent(data, tau=1, embedding_dim=2):
    N = len(data)
    max_index = N - (embedding_dim - 1) * tau
    embedded = np.array([data[i: max_index + i : tau] for i in range(embedding_dim)]).T
    dist_matrix = squareform(pdist(embedded))
    np.fill_diagonal(dist_matrix, np.inf)
    min_indices = np.argmin(dist_matrix, axis=1)
    divergence = np.mean(np.abs(embedded - embedded[min_indices]), axis=0)
    divergence = divergence[divergence > 0]
    log_divergence = np.log(divergence)
    time = np.arange(len(log_divergence))
    slope, _ = np.polyfit(time, log_divergence, 1)
    return slope

s_freq = 120
duration = 20.0
overlap = 15.0
n_jobs = -1
channels2use = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'Cz', 'C4', 'T5', 'T6', 'P3', 'Pz', 'P4', 'O1', 'O2']
to_skip = ['BORUTTO_JANNA_VLADIMIROVNA', 'Kutuz_f23_contr', 'MANUILOVA_ELENA_55', 'Martinenko_m45', 'Skopincev_20', 'FiAV_m50', 'LOMTEV_30']

spectrograms = []
lyapunovs = []
labels = []

for i in tqdm(ids):
    path = zg_files[i]
    if any([x in path for x in to_skip]):
        continue
    sample = mne.io.read_raw_edf(path, verbose=False, preload=True)
    sample = sample.resample(s_freq, verbose=False)

    sample = sample.filter(l_freq=1, h_freq=30, method='iir', verbose=False)
    channels = sample.ch_names
    to_drop = channels[19:]

    new_idx = []
    skip = False
    for ch in channels2use:
        found = False
        for k in range(19):
            if ch in channels[k]:
                new_idx.append(k)
                found = True
                break
        if not found:
            skip = True
            break
    sample = sample.pick(np.array(channels)[new_idx])
    if len(sample) / s_freq > 60:
        sample = sample.crop(tmin=0.0, tmax=60.0)
    events = mne.make_fixed_length_events(sample, duration=duration, overlap=overlap)
    epochs = mne.Epochs(sample, events, tmin=0.0, tmax=duration, baseline=None, preload=True, verbose=False)


    cur_l = []
    for epoch in epochs:
        features = Parallel(n_jobs=n_jobs)(delayed(lyapunov_exponent)(channel) for channel in epoch)
        cur_l.append(np.array(features))
    lyapunovs.append(cur_l)

with open('/mnt/s3-data2/amiftakhova/OCD/lyapunovs_zg.pkl', 'wb') as f:
    pickle.dump(lyapunovs, f)