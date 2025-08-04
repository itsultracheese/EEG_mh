import os
import mne
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score

label2class = {
        4: 0, 5: 0, 6: 0, # anxiety
        7: 1, 8: 1, 9: 1, # bipolar
        10: 2, # control
        11: 3, 12: 3, 13: 3, # depression
        14: 4, # personality disorder,
        0: 5, 1: 5, 2: 5, 3: 5# stress
    }

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        return x, self.labels[idx]
    
class EEG2DCNN(nn.Module):
    def __init__(self, num_classes):
        super(EEG2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=19, out_channels=32, kernel_size=(5, 10), stride=(2, 5))
        # self.bn1 = nn.BatchNorm2d(32)
        # self.dropout1 = nn.Dropout(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 10), stride=(1, 5))
        # self.bn2 = nn.BatchNorm2d(64)
        # self.dropout2 = nn.Dropout(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2))
        self.dropout = nn.Dropout(0.2)
        # self.bn3 = nn.BatchNorm1d(2112)
        self.fc1 = nn.Linear(1408, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, 19, 58, 1201)
        x = F.relu(self.conv1(x))
        # x = self.pool1(self.dropout1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        # x = self.pool2(self.dropout2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        # x = self.bn3(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
def prepare_data(use_zg=True):
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

    lengths_og = []
    lengths_zg = []
    for i in tqdm(range(len(og_files)), desc='Calculating lengths'):
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

    # ids = np.argwhere((np.array(lengths_og) >= 20.0) & (np.array(lengths_zg) >= 20.0)).flatten()
    ids = np.argwhere(np.array(lengths_zg) >= 20.0).flatten()

    s_freq = 120
    duration = 20.0
    overlap = 15.0
    channels2use = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'Cz', 'C4', 'T5', 'T6', 'P3', 'Pz', 'P4', 'O1', 'O2']
    to_skip = ['BORUTTO_JANNA_VLADIMIROVNA', 'Kutuz_f23_contr', 'MANUILOVA_ELENA_55', 'Martinenko_m45', 'Skopincev_20', 'FiAV_m50', 'LOMTEV_30']

    spectrograms = []
    labels = []

    for i in tqdm(ids, desc='Calculating spectrograms'):
        if use_zg:
            path = zg_files[i]
        else:
            path = og_files[i]
        # print(path)
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
        
        # FOR MORLET
        freqs = np.arange(1, 30, 0.5)
        n_cycles = freqs / 2.
        tfr = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                                            return_itc=False, average=False, n_jobs=8, verbose=False)
        power = tfr.data
        power = (power - power.min(axis=(0, 1, 3), keepdims=True)) / \
                (power.max(axis=(0, 1, 3), keepdims=True) - power.min(axis=(0, 1, 3), keepdims=True))
        # FOR MORLET

        # FOR STFT
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
        power = (power - power.min(axis=(0, 1, 3), keepdims=True)) / \
                (power.max(axis=(0, 1, 3), keepdims=True) - power.min(axis=(0, 1, 3), keepdims=True))
        # FOR STFT

        spectrograms.append(power)
        if use_zg:
            labels.append(zg_labels[i] * np.ones(len(epochs)))
        else:
            labels.append(og_labels[i] * np.ones(len(epochs)))
    return spectrograms, labels

def main():
    spectrograms, labels = prepare_data(use_zg=True)
    class_labels = [[label2class[j] for j in labels[i]] for i in range(len(labels))]
    train_ids, test_ids = train_test_split([i for i in range(len(spectrograms))], test_size=0.2, stratify=[i[0] for i in class_labels], random_state=92)

    train_dataset = EEGDataset(np.concatenate([spectrograms[i] for i in train_ids]), np.concatenate([class_labels[i] for i in train_ids]))
    test_dataset = EEGDataset(np.concatenate([spectrograms[i] for i in test_ids]), np.concatenate([class_labels[i] for i in test_ids]))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(np.concatenate([class_labels[i] for i in train_ids])), y=np.concatenate([class_labels[i] for i in train_ids]))
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    device = 'cuda:0'
    num_epochs = 100

    # Training loop
    train_losses = []
    test_losses = []
    lrs = [5e-5]
    metrics = {}
    # print('5e-5, only zg, class weight')

    for j, lr in enumerate(lrs):
        model = EEG2DCNN(num_classes=6).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-9)

        for epoch in tqdm(range(num_epochs), desc='Training'):
            model.train()
            losses = []
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {np.mean(losses)}')
            train_losses.append(np.mean(losses))
            model.eval()
            losses = []
            accs = []
            for inputs, targets in test_loader:
                with torch.no_grad():
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    # print(torch.argmax(outputs, dim=1), targets)
                    loss = criterion(outputs, targets)
                    accuracy = torch.sum(torch.argmax(outputs, dim=1) == targets).item() / len(targets)
                    losses.append(loss.item())
                    accs.append(accuracy)
            print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {np.mean(losses)}, Test Accuracy: {np.mean(accs)}')
            test_losses.append(np.mean(losses))

            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'loss_zg_{j}.png')

        X = [spectrograms[i] for i in test_ids]
        Y = [class_labels[i] for i in test_ids]

        model.eval()
        subject_preds = []
        subject_preds_majority = []
        subject_true = []
        with torch.no_grad():
            for x, y in zip(X, Y):
                x = torch.tensor(x, dtype=torch.float32)
                # x = x.unsqueeze(0)
                x = x.to(device)
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)
                # print(probs)
                avg_probs = probs.mean(dim=0)  # (num_classes,)
                pred_class = avg_probs.argmax().item()

                subject_preds.append(pred_class)
                pred_classes = outputs.argmax(dim=1)
                pred_class = torch.mode(pred_classes).values.item()
                subject_preds_majority.append(pred_class)

                subject_true.append(y[0])

        subject_preds = np.array(subject_preds)
        subject_true = np.array(subject_true)
        accuracy = np.mean(subject_preds == subject_true)
        print(f'F1-score, macro: {f1_score(subject_true, subject_preds, average="macro")}')
        print(f'F1-score, micro: {f1_score(subject_true, subject_preds, average="micro")}')
        print(f'Accuracy: {accuracy}')

        metrics[j] = f1_score(subject_true, subject_preds, average="macro")
        with open('metrics_zg.json', 'w') as f:
            json.dump(metrics, f)

        subject_preds_majority = np.array(subject_preds_majority)
        accuracy = np.mean(subject_preds_majority == subject_true)
        print(f'only zg, Majority Accuracy: {accuracy}')
        # torch.save(model, '/mnt/s3-data2/amiftakhova/eeg_cnn/eeg_cnn_zg_40.pth')

if __name__ == '__main__':
    main()