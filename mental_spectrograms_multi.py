import os
import json
import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

import mne
from scipy.signal import stft

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torch.utils.tensorboard import SummaryWriter

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
    17: 4, # personality disorder,
    0: 5, 1: 5, 2: 5, 3: 5, # stress,
    14: 6, 15: 6, 16: 6 #ocd
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
        self.conv1 = nn.Conv2d(in_channels=19, out_channels=32, kernel_size=(5, 10), stride=(1, 3))
        # self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 8), stride=(1, 3))
        # self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2))
        # self.bn3 = nn.BatchNorm2d(128)
       
        self.gap = nn.AdaptiveAvgPool2d((2, 10))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2560, 128) # for 19s
        # self.fc1 = nn.Linear(640, 128) # for 10s
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        # print(x.shape)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

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
    duration = 19.0
    overlap = 15.0
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
        
        # FOR MORLET
        if type == 'morlet':
            freqs = np.arange(1, 30, 0.5)
            n_cycles = freqs / 2.
            tfr = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                                                return_itc=False, average=False, n_jobs=8, verbose=False)
            power = tfr.data
            power = (power - power.min(axis=(0, 1, 3), keepdims=True)) / \
                    (power.max(axis=(0, 1, 3), keepdims=True) - power.min(axis=(0, 1, 3), keepdims=True))
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
            power = (power - power.min(axis=(0, 1, 3), keepdims=True)) / \
                    (power.max(axis=(0, 1, 3), keepdims=True) - power.min(axis=(0, 1, 3), keepdims=True))
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
    print(train_cnt, test_cnt)

    return train_spectrograms, train_labels, test_spectrograms, test_labels, val_spectrograms, val_labels

def load_spectrograms(path, type='20s_15o_norm'):
    with open(os.path.join(path, f'train_{type}.pkl'), 'rb') as f:
        train_spectrograms, train_labels = pickle.load(f)
    with open(os.path.join(path, f'test_{type}.pkl'), 'rb') as f:
        test_spectrograms, test_labels = pickle.load(f)
    with open(os.path.join(path, f'val_{type}.pkl'), 'rb') as f:
        val_spectrograms, val_labels = pickle.load(f)
    return train_spectrograms, train_labels, test_spectrograms, test_labels, val_spectrograms, val_labels

def stand_spect(data):
    mean = np.mean(data, axis=(-2, -1), keepdims=True)
    std = np.std(data, axis=(-2, -1), keepdims=True)
    std = np.clip(std, a_min=1e-8, a_max=None)
    data_normalized = (data - mean) / std
    return data_normalized

def test_model(test_spectrograms, test_labels, model, writer, device = 'cuda:0'):
    X, Y = test_spectrograms, test_labels

    model.eval()
    subject_preds = []
    subject_preds_majority = []
    subject_true = []
    with torch.no_grad():
        for x, y in zip(X, Y):
            x = stand_spect(x)
            x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
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
    macro_f1 = f1_score(subject_true, subject_preds, average="macro")
    micro_f1 = f1_score(subject_true, subject_preds, average="micro")
    print(subject_preds[:30])
    print(f'F1-score, macro: {macro_f1}')
    print(f'F1-score, micro: {micro_f1}')
    print(f'Accuracy: {accuracy}')

    writer.add_scalar("F1-macro/test", macro_f1, 0)
    writer.add_scalar("F1-micro/test", micro_f1, 0)
    writer.add_scalar("Accuracy/test", accuracy, 0)

    return macro_f1, micro_f1, accuracy

class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        """
        patience: how many epochs to wait after last improvement
        delta: minimum change to count as an improvement
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.delta:  # no significant improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # improvement
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

def train_model(train_loader, val_loader, test_spectrograms, test_labels, weight_decay=1e-5,
                num_epochs=100, weighted=False, weights=None, device='cuda:0', save=False, run_name='exp'):

    # Training loop
    lrs = [1e-5, 5e-5, 1e-4]
    metrics = {}

    if weighted:
        run_name += '_weighted'
    else:
        run_name += '_no-weights'

    for j, lr in enumerate(lrs):
        writer = SummaryWriter(log_dir=f"runs_unnormed/{run_name}_lr_{lr}")

        # log hyperparameters
        writer.add_hparams(
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": 64,
                "num_epochs": num_epochs,
                "optimizer": "AdamW",
                "loss_fn": "CrossEntropy(weighted)"
            },
            {}
        )

        model = EEG2DCNN(num_classes=7).to(device)

        if weighted:
            criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-9)

        early_stopping = EarlyStopping(patience=10, delta=1e-4)

        train_losses = []
        val_losses = []
        val_accs = []
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
            # scheduler.step()

            train_loss = np.mean(losses)
            train_losses.append(np.mean(losses))
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {np.mean(train_loss)}')

            model.eval()
            losses = []
            accs = []
            predictions = []
            gts = []
            for inputs, targets in val_loader:
                with torch.no_grad():
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    
                    loss = criterion(outputs, targets)
                    # preds = torch.sigmoid(outputs)
                    accuracy = torch.sum(torch.argmax(outputs, dim=1) == targets).item() / len(targets)
                    
                    pred_classes = outputs.argmax(dim=1)
                    predictions.append(pred_classes.cpu().numpy())
                    gts.append(targets.cpu().numpy())

                    losses.append(loss.item())
                    accs.append(accuracy)
            gts = np.concatenate(gts).reshape(-1)
            predictions = np.concatenate(predictions).reshape(-1)
            f1 = f1_score(gts, predictions, average='macro')

            val_loss = np.mean(losses)
            val_acc = np.mean(accs)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss}, Val Accuracy: {val_acc}, Val F1: {f1}')

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("F1-macro/val", f1, epoch)

            early_stopping(f1, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(early_stopping.best_state)  # restore best
                break

        test_macro_f1, test_micro_f1, test_accuracy = test_model(test_spectrograms, test_labels, model, writer, device)
        metrics[j] = {
            'macro-f1': test_macro_f1,
            'micro-f1': test_micro_f1,
            'acc': test_accuracy,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accs': val_accs
        }
        with open(f'results/metrics_og_{run_name}.json', 'w') as f:
            json.dump(metrics, f)

        if save:
            torch.save(model, f'/media/ssd-3t/amiftakhova/eeg_cnn/eeg_cnn_og_{run_name}_{j}.pth')

    return model, metrics

def main():
    # train_spectrograms, train_labels, test_spectrograms, test_labels, val_spectrograms, val_labels = prepare_data(use_og=True, type='morlet')
    data_fldr = '/media/ssd-3t/amiftakhova/spectrograms/'
    run_name = '20s_15o_bs64_no_bn'

    train_spectrograms, train_labels, test_spectrograms, test_labels, val_spectrograms, val_labels = load_spectrograms(data_fldr, type='20s_15o')
    print('loaded data')

    # norm by mean, std
    train_spectrograms = np.concatenate(train_spectrograms)
    train_spectrograms = stand_spect(train_spectrograms)
    val_spectrograms = np.concatenate(val_spectrograms)
    val_spectrograms = stand_spect(val_spectrograms)

    train_dataset = EEGDataset(train_spectrograms, np.concatenate(train_labels))
    # test_dataset = EEGDataset(test_spectrograms, np.concatenate(test_labels))
    val_dataset = EEGDataset(val_spectrograms, np.concatenate(val_labels))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(np.concatenate(train_labels)), y=np.concatenate(train_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    model, metrics = train_model(train_loader, val_loader, test_spectrograms, test_labels, weight_decay=1e-5,
        num_epochs=100, weighted=True, weights=class_weights, device='cuda:0', save=True, run_name=run_name)
    
    for k, v in metrics.items():
        print(k)
        ms = ['macro-f1', 'micro-f1', 'acc']
        for m in ms:
            print(f'\t{m}: {v[m]}')
        print()

if __name__ == '__main__':
    main()