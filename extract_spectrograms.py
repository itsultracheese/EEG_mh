import os
import json
from collections import Counter

import pickle
import numpy as np
from tqdm import tqdm

import mne
from scipy.signal import stft

def load_data(use_og=True):
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

    class2name = {0: 'anxiety', 1: 'bipolar', 2: 'control', 3: 'depression', 4: 'personality disorder', 5: 'stress', 6: 'ocd'}
    name2class = {v: k for k, v in class2name.items()}

    og_files = []
    og_labels = []
    for fldr in inside_folders:
        pth =  data_folder + fldr
        og_pths = [x for x in os.listdir(pth) if x.lower().endswith('.edf') and ('og.' in x.lower() or '_ог.' in x.lower() or ' ог.' in x.lower() or 'eo.' in x.lower() or '_eo' in x.lower() or '_of' in x.lower()) or '_оg' in x.lower()]
        for f in og_pths:
            og_files.append(pth + '/' + f)
            og_labels.append(fldr2label[fldr])

    print(f'Number of files for open eyes: {len(og_files)}')

    # lengths_og = []
    # for i in tqdm(range(len(og_files)), desc='Calculating lengths'):
    #     og_file = og_files[i]
    #     try:
    #         og_sample = mne.io.read_raw_edf(og_file, verbose=False)
    #         og_len = 1.0 * len(og_sample) / og_sample.info['sfreq']
    #         lengths_og.append(og_len)
    #     except Exception as e:
    #         print(f, e)
    #         lengths_og.append(0)
    # ids = np.argwhere((np.array(lengths_og) >= 20.0) & (np.array(lengths_zg) >= 20.0)).flatten()

    with open('split_multi_classes.json', encoding='utf-8') as f:
        split = json.load(f)

    og_train = []
    og_test = []
    og_val = []

    for i, fname in enumerate(og_files):
        fname_clr = fname.split('/')[-1][:-4]
        if 'DlE_f13_f32-0_f92-8_og' in fname:
            continue
        if fname_clr in split:
            if split[fname_clr]['split'] == 'train':
                og_train.append((fname, name2class[split[fname_clr]['target']]))
            elif split[fname_clr]['split'] == 'test':
                og_test.append((fname, name2class[split[fname_clr]['target']]))
            else:
                og_val.append((fname, name2class[split[fname_clr]['target']]))
    return og_train, og_test, og_val
    
def extract_spectrograms(paths, type='morlet'):
    s_freq = 120
    duration = 10.0
    overlap = 5.0
    channels2use = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'Cz', 'C4', 'T5', 'T6', 'P3', 'Pz', 'P4', 'O1', 'O2']

    spectrograms = []
    labels = []

    for i in tqdm(range(len(paths)), desc='Calculating spectrograms'):
        path = paths[i][0]

        sample = mne.io.read_raw_edf(path, verbose=False, preload=True)
        sample = sample.resample(s_freq, verbose=False)
        sample = sample.filter(l_freq=1, h_freq=30, method='iir', verbose=False)
        channels = sample.ch_names

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
        if skip:
            print(f'skipped {path}')
            continue

        sample = sample.pick(np.array(channels)[new_idx])
        if len(sample) / s_freq > 60:
            sample = sample.crop(tmin=0.0, tmax=60.0)
        events = mne.make_fixed_length_events(sample, duration=duration, overlap=overlap)
        epochs = mne.Epochs(sample, events, tmin=0.0, tmax=duration, baseline=None, preload=True, verbose=False)
        if len(epochs) == 0:
            print(path)
            break
        
        # FOR MORLET
        if type == 'morlet':
            freqs = np.arange(1, 30, 0.5)
            n_cycles = freqs / 2.
            tfr = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                                                return_itc=False, average=False, n_jobs=8, verbose=False)
            power = tfr.data
            # power = (power - power.min(axis=(0, 1, 3), keepdims=True)) / \
            #         (power.max(axis=(0, 1, 3), keepdims=True) - power.min(axis=(0, 1, 3), keepdims=True))
        # FOR MORLET

        # FOR STFT
        else:
            data = epochs.get_data()

            powers = []
            for e_id in range(data.shape[0]):
                powers_epoch = []
                for ch_id in range(data.shape[1]):
                    ch_data = data[e_id, ch_id]
                    fs = s_freq
                    window = 'hann'
                    nperseg = int(fs * 0.5)
                    noverlap = int(nperseg * 0.75)
                    f, t, Zxx = stft(ch_data, fs=fs, window=window, 
                                            nperseg=nperseg, noverlap=noverlap)
                    power = np.abs(Zxx)
                    powers_epoch.append(power)
                powers.append(np.array(powers_epoch))
            power = np.array(powers)
            # power = (power - power.min(axis=(0, 1, 3), keepdims=True)) / \
            #         (power.max(axis=(0, 1, 3), keepdims=True) - power.min(axis=(0, 1, 3), keepdims=True))
        # FOR STFT

        spectrograms.append(power)
        label = paths[i][1]
        labels.append(label * np.ones(len(epochs)))
    return spectrograms, labels
    
def prepare_data(use_og=True, type='morlet'):
    og_train, og_test, og_val = load_data(use_og)
    train_spectrograms, train_labels = extract_spectrograms(og_train, type)
    test_spectrograms, test_labels = extract_spectrograms(og_test, type)
    val_spectrograms, val_labels = extract_spectrograms(og_val, type)

    train_cnt = Counter(np.concatenate(train_labels).tolist())
    test_cnt = Counter(np.concatenate(test_labels).tolist())
    val_cnt = Counter(np.concatenate(val_labels).tolist())
    print(train_cnt, test_cnt, val_cnt)

    print(f'Train size: {len(np.concatenate(train_spectrograms))}, val size: {len(np.concatenate(val_spectrograms))}, test size: {len(np.concatenate(test_spectrograms))}')

    with open('/media/ssd-3t/amiftakhova/spectrograms/train_10s_5o_norm.pkl', 'wb') as f:
        pickle.dump((train_spectrograms, train_labels), f)
    with open('/media/ssd-3t/amiftakhova/spectrograms/test_10s_5o_norm.pkl', 'wb') as f:
        pickle.dump((test_spectrograms, test_labels), f)
    with open('/media/ssd-3t/amiftakhova/spectrograms/val_10s_5o_norm.pkl', 'wb') as f:
        pickle.dump((val_spectrograms, val_labels), f)

    return train_spectrograms, train_labels, test_spectrograms, test_labels, val_spectrograms, val_labels

if __name__ == '__main__':
    prepare_data()