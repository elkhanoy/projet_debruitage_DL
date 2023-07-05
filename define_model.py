import torch
import torch.cuda
import h5py
from torch.utils.data import DataLoader, Dataset, BatchSampler, Sampler
from Outils import mix
import soundfile as sf
import librosa
import librosa.display
import math
import soundfile as sf
import pandas as pd
import os
import numpy as np
from scipy.io.wavfile import write
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pylab as plt
from torchsummary import summary


num_workers = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_plot_one_file(model, model_file):

    model.load_state_dict(torch.load(model_file, map_location = torch.device(device)))
    # model.load_state_dict(torch.load('models/trained_model_fc_500_2_error_0.060_permutation_epoch_100_batch_1000'), map_location = torch.device(device))

    model.eval()
    SNR_dB= 5
    path_clean= 'audio_files/dev-clean/8842/304647/8842-304647-0008.flac'
    path_noise = 'audio_test/babble.wav'
    # summary(model, (2, 125, 257))


    clean, sr = librosa.load(path_clean, sr = 16000) # Load the clean speech and the noise
    noise, sr_noise = librosa.load(path_noise, sr = 16000)
    mixed, sr_mixed = librosa.load('/home/yelkhanoussi/Projet_DL/debruitage_samer_yassine/audio_files/mixture/84-121123-0003__babble.wav', sr = 16000)


    # mixed = mix.mix_noise(clean, noise, SNR_dB) # Create the mixture
    mix.save_wav16k(mixed, 'sorties/noised.wav') # Save the noisy audio file
    mix.save_wav16k(clean, 'sorties/clean.wav') # Save the noisy audio file
    
    stft_clean = mix.compute_stft(clean)
    clean_mag, clean_phase = librosa.magphase(stft_clean)
    clean_mag = mix.create_batches(clean_mag)
    clean_mag = np.reshape(clean_mag, (clean_mag.shape[0]*clean_mag.shape[1], clean_mag.shape[2])).T
    clean_normalizer = MinMaxScaler()
    clean_mag_norm = clean_normalizer.fit_transform(clean_mag.reshape(-1, clean_mag.shape[-1])).reshape(clean_mag.shape)

    stft_mixed = mix.compute_stft(mixed) # compute spectrogram
    mixed_mag, mixed_phase = librosa.magphase(stft_mixed) # extract magnitude M and phase P from spectrogram D, such as D = M*P
    mixed_normalizer = MinMaxScaler()

    batch_mixed_phase = mix.create_batches(mixed_phase) # Convert phase spec to batches
    batch_mixed_mag = mix.create_batches(mixed_mag)
    batch_mixed_mag = mixed_normalizer.fit_transform(batch_mixed_mag.reshape(-1, batch_mixed_mag.shape[-1])).reshape(batch_mixed_mag.shape)
    mixed_mag_plot = np.reshape(batch_mixed_mag, (batch_mixed_mag.shape[0]*batch_mixed_mag.shape[1], batch_mixed_mag.shape[2])).T
    batch_mixed_mag = torch.from_numpy(batch_mixed_mag) # Conver mag spec to batches + convert to tensor
    data = batch_mixed_mag.to(device) # to CPU or GPU

    output = model(data) # Predict
    output = torch.squeeze(output)
    np_arr_norm = output.cpu().detach().numpy() # Convert tensor to ndarray
    np_arr = mixed_normalizer.inverse_transform(np_arr_norm.reshape(-1, np_arr_norm.shape[-1])).reshape(np_arr_norm.shape)
    out_phase = np.reshape(batch_mixed_phase, (batch_mixed_phase.shape[0]*batch_mixed_phase.shape[1], batch_mixed_phase.shape[2])).T # resphape to original shape
    out_mag = np.reshape(np_arr, (np_arr.shape[0]*np_arr.shape[1], np_arr.shape[2])).T # reshape to original shape
    out_mag_norm = np.reshape(np_arr_norm, (np_arr_norm.shape[0]*np_arr_norm.shape[1], np_arr_norm.shape[2])).T # reshape to original shape

    ### PLOT

    fig, ax = plt.subplots(nrows= 2, ncols= 2)
    plt.plasma()
    librosa.display.specshow(clean_mag_norm, sr = 16000, x_axis = 's', y_axis = 'hz', n_fft= 512, hop_length= 128, ax= ax[0,0], cmap = None)
    librosa.display.specshow(mixed_mag_plot, sr = 16000, x_axis = 's', y_axis = 'hz', n_fft= 512, hop_length= 128, ax= ax[0,1], cmap = None)
    librosa.display.specshow(out_mag_norm, sr = 16000, x_axis = 's', y_axis = 'hz', n_fft= 512, hop_length= 128, ax= ax[1,0], cmap = None)
    librosa.display.specshow(clean_mag, sr = 16000, x_axis = 's', y_axis = 'hz', n_fft= 512, hop_length= 128, ax= ax[1,1], cmap = None)
    ax[0][0].set(title='Target normalized spectrogram')
    ax[0][1].set(title='Noised normalized spectrogram')
    ax[1][0].set(title='Denoised normalized spectrogram')
    ax[1][1].set(title='Target spectrogram (non-normalized)')

    plt.show()
    recons_stft = out_mag * out_phase # reconstruct spectogram
    recons_audio = mix.compute_istft(recons_stft) # reconstruct audio file
    snr = mix.compute_SNR(clean, mixed)
    sdr = mix.compute_SNR(clean, recons_audio)
    print(f'SNRin : {snr} \n SDRout : {sdr} \n Gain : {sdr-snr} \n')

    mix.save_wav16k(recons_audio, 'sorties/denoised.wav')
    # write('sorties/652-130737-0000_denoised.wav', 16000, recons_audio) # save the audio file

def test_on_files(model, model_file):
    # summary(model, (2, 125, 257))
    path_to_save = 'denoised_test/' + model_file.split('.')[0] + '/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    model.load_state_dict(torch.load(model_file, map_location = torch.device(device)))
    # model.load_state_dict(torch.load('models/trained_model_fc_500_2_error_0.060_permutation_epoch_100_batch_1000'), map_location = torch.device(device))

    model.eval()

    path_csv= 'mixtures_1h_babble_v2_test_list.csv'
    df = pd.read_csv(path_csv)

    list_samples = df.reset_index()[['mixture', 'path_clean', 'SNR']].values.astype(str).tolist()
    SDRout = []
    SNRin = []
    gain = []
    for it, (x, y, snr) in enumerate(list_samples):
        print(it)
        clean, sr = librosa.load(y, sr = 16000) # Load the clean speech and the mixture
        mixed, sr_mixed = librosa.load(x, sr = 16000)
        stft_clean = mix.compute_stft(clean)
        clean_mag, clean_phase = librosa.magphase(stft_clean)
        clean_mag = mix.create_batches(clean_mag)
        clean_mag = np.reshape(clean_mag, (clean_mag.shape[0]*clean_mag.shape[1], clean_mag.shape[2])).T

        stft_mixed = mix.compute_stft(mixed) # compute spectrogram
        mixed_mag, mixed_phase = librosa.magphase(stft_mixed) # extract magnitude M and phase P from spectrogram D, such as D = M*P
        mixed_normalizer = MinMaxScaler()

        batch_mixed_phase = mix.create_batches(mixed_phase) # Convert phase spec to batches
        batch_mixed_mag = mix.create_batches(mixed_mag)
        batch_mixed_mag = mixed_normalizer.fit_transform(batch_mixed_mag.reshape(-1, batch_mixed_mag.shape[-1])).reshape(batch_mixed_mag.shape)
        batch_mixed_mag = torch.from_numpy(batch_mixed_mag) # Conver mag spec to batches + convert to tensor
        data = batch_mixed_mag.to(device) # to CPU or GPU

        output = model(data) # Predict
        # output = torch.squeeze(output)
        np_arr_norm = output.cpu().detach().numpy() # Convert tensor to ndarray
        np_arr = mixed_normalizer.inverse_transform(np_arr_norm.reshape(-1, np_arr_norm.shape[-1])).reshape(np_arr_norm.shape)
        out_phase = np.reshape(batch_mixed_phase, (batch_mixed_phase.shape[0]*batch_mixed_phase.shape[1], batch_mixed_phase.shape[2])).T # resphape to original shape
        out_mag = np.reshape(np_arr, (np_arr.shape[0]*np_arr.shape[1], np_arr.shape[2])).T # reshape to original shape
        out_mag_norm = np.reshape(np_arr_norm, (np_arr_norm.shape[0]*np_arr_norm.shape[1], np_arr_norm.shape[2])).T # reshape to original shape

        recons_stft = out_mag * out_phase # reconstruct spectogram
        recons_audio = mix.compute_istft(recons_stft) # reconstruct audio file
        SNRin_value = mix.compute_SNR(clean, mixed)
        SDRout_value = mix.compute_SNR(clean, recons_audio)
        clean = (clean * 32767).astype(np.int16)
        recons_audio = (recons_audio * 32767).astype(np.int16)
        mix.save_wav16k(recons_audio, path_to_save +'denoised_' + os.path.basename(y))
        gain_value = SDRout_value - SNRin_value
        SNRin.append(SNRin_value)
        SDRout.append(SDRout_value)
        gain.append(gain_value)
    
    print(f'Mean SDRout = {np.mean(SDRout)} \n Mean gain = {np.mean(gain)}')

    new_df = pd.DataFrame(list_samples, columns = ['mixture', 'target', 'SNR_mixture'])
    new_df['SNRin'] = SNRin
    new_df['SDRout'] = SDRout
    new_df['gain'] = gain
    new_df.to_csv(path_to_save + (model_file.split('.')[0]).split('/')[1] + '_test_results.csv')

def test_from_h5():
    hf = h5py.File('dataset_mixture_1_noise.hdf5', 'r')
    batch_mag_stft = hf['x_train'][0:10]
    mag_stft = np.reshape(batch_mag_stft, (-1, 513)).T
    audio = librosa.griffinlim(mag_stft, hop_length= 128, n_fft= 256)
    mix.save_wav16k(audio, 'test_hdf5.wav')

def train(model, loader, criterion, optimizer, epoch, log=None):
    # Set model to training mode
    model.train(True)
    epoch_loss = 0.
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # Loop over each batch from the training set
    for data, target in loader:
        # Copy data to GPU if needed

        # data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
        # target = scaler.transform(target.reshape(-1, target.shape[-1])).reshape(target.shape)

        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)
        # output = torch.squeeze(output)

        # Calculate loss
        loss = criterion(output, target.to(torch.float32))
        epoch_loss += loss.item()

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

    epoch_loss /= len(loader.dataset)
    print('Train Epoch: {}, Loss: {:.4f}'.format(epoch, epoch_loss))


def evaluate(model, loader, criterion=None, epoch=None, log=None):
    model.eval()
    loss = 0
    for data, target in loader:
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
        # target = scaler.transform(target.reshape(-1, target.shape[-1])).reshape(target.shape)
        data = data.to(device)
        target = target.to(device)
        
        output = torch.squeeze(model(data))

        if criterion is not None:
            loss += criterion(output, target.to(torch.float32)).item()

    if criterion is not None:
        loss /= len(loader.dataset)

    print('Average loss: {:.4f}\n'.format(loss))


class RandomBatchSampler(Sampler):
    """Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    This is useful for cases when you are interested in 'weak shuffling'
    :param dataset: dataset you want to batch
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch size
    :type batch_size: int
    :returns: generator object of shuffled batch indices
    """
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = torch.randperm(int(self.n_batches))

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)


def fast_loader(dataset, batch_size=32, drop_last=False, transforms=None):
    """Implements fast loading by taking advantage of .h5 dataset
    The .h5 dataset has a speed bottleneck that scales (roughly) linearly with the number
    of calls made to it. This is because when queries are made to it, a search is made to find
    the data item at that index. However, once the start index has been found, taking the next items
    does not require any more significant computation. So indexing data[start_index: start_index+batch_size]
    is almost the same as just data[start_index]. The fast loading scheme takes advantage of this. However,
    because the goal is NOT to load the entirety of the data in memory at once, weak shuffling is used instead of
    strong shuffling.
    :param dataset: a dataset that loads data from .h5 files
    :type dataset: torch.utils.data.Dataset
    :param batch_size: size of data to batch
    :type batch_size: int
    :param drop_last: flag to indicate if last batch will be dropped (if size < batch_size)
    :type drop_last: bool
    :returns: dataloading that queries from data using shuffled batches
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(
        dataset, batch_size=None,  # must be disabled when using samplers
        sampler=BatchSampler(RandomBatchSampler(dataset, batch_size), batch_size=batch_size, drop_last=drop_last), 
        num_workers=num_workers
    )


class HDF5Dataset(Dataset):
    def __init__(self, file_path, operation = 'train'):
        self.file_path = file_path
        self.x_dataset_name = 'x_' + operation
        self.y_dataset_name = 'y_' + operation
        self.length = None

        with h5py.File(self.file_path, 'r') as hf:
            
            self.length = len(hf.get(self.x_dataset_name))

    def __len__(self):
        return self.length

    def _open_hdf5(self):
        self._hf = h5py.File(self.file_path, 'r')

    def __getitem__(self, index):
        if not hasattr(self, '_hf'):
            self._open_hdf5()

        x = self._hf[self.x_dataset_name][index]
        y = self._hf[self.y_dataset_name][index]

        x = (torch.from_numpy(x)).to(torch.float32)
        y = (torch.from_numpy(y)).to(torch.float32)
        return (x, y)

def get_train_loader_hdf5(batch_size, dataset_name):
    print('Train: ', end="")
    train_dataset = HDF5Dataset('datasets/' + dataset_name + '.hdf5', operation = 'train')
    # train_loader = DataLoader(train_dataset, batch_size=batch_size,
    #                           shuffle=True, num_workers=num_workers, drop_last= False)
    train_loader = fast_loader(train_dataset, batch_size=batch_size, drop_last=False)
    print('Found', len(train_dataset), ' train samples')
    return train_loader


def get_test_loader_hdf5(batch_size, dataset_name):
    print('Test: ', end="")
    test_dataset = HDF5Dataset('datasets/' + dataset_name + '.hdf5', operation = 'test')
    # test_loader = DataLoader(test_dataset, batch_size=batch_size,
    #                           shuffle=True, num_workers=num_workers, drop_last= False)
    test_loader = fast_loader(test_dataset, batch_size=batch_size, drop_last=False)

    print('Found', len(test_dataset), ' test samples')
    return test_loader
