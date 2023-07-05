from Outils import mix
import soundfile as sf
import librosa as lb
import pandas as pd
import h5py
import numpy as np
import multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

hdf5_path = 'datasets/dataset_babble_v2.hdf5'
csv_path = 'mixtures_1h_babble_v2.csv'

df = pd.read_csv(csv_path)
y = np.empty((0, 125, 257))
X = np.empty((0, 125, 257))

with h5py.File(hdf5_path, 'w') as hf:
    dset_x_train = hf.create_dataset('x_train', data=X, shape=X.shape, maxshape = (None, 125, 257), chunks=True)
    dset_y_train = hf.create_dataset('y_train', data=X, shape=X.shape, maxshape = (None, 125, 257), chunks=True)
    dset_x_test = hf.create_dataset('x_test', data=X, shape=X.shape, maxshape = (None, 125, 257), chunks=True)
    dset_y_test = hf.create_dataset('y_test', data=X, shape=X.shape, maxshape = (None, 125, 257), chunks=True)

df_train, df_test = train_test_split(df, test_size=0.10, random_state=1998, shuffle = True)

df_train.to_csv(csv_path.split('.')[0] + '_train_list.csv')
df_test.to_csv(csv_path.split('.')[0] + '_test_list.csv')

train_list = df_train.reset_index()[['mixture', 'path_clean']].values.astype(str).tolist()
test_list = df_test.reset_index()[['mixture', 'path_clean']].values.astype(str).tolist()

size_train = len(train_list)
size_test = len(test_list)
print(f'Train size: {size_train}, Test size = {size_test}')
print(train_list[0])
input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

for it, (mixed_path, clean_path) in enumerate(train_list):
    print(it/size_train)
    clean_data, sr = lb.load(clean_path, sr = 16000)
    mixed_data, sr = lb.load(mixed_path, sr = 16000)
    clean_STFT = mix.compute_stft(clean_data)
    mixed_STFT = mix.compute_stft(mixed_data)
    clean_mag = np.abs(clean_STFT)
    mixed_mag = np.abs(mixed_STFT)
    batched_clean_mag = mix.create_batches(clean_mag)
    batched_mixed_mag = mix.create_batches(mixed_mag)
    if (batched_clean_mag.shape[0] == 0):
        continue
    try:
        batched_mixed_mag = input_scaler.fit_transform(batched_mixed_mag.reshape(-1, batched_mixed_mag.shape[-1])).reshape(batched_mixed_mag.shape)
        batched_clean_mag = target_scaler.fit_transform(batched_clean_mag.reshape(-1, batched_clean_mag.shape[-1])).reshape(batched_clean_mag.shape)
    except:
        print(mixed_path)
        print(clean_path)
        quit()
    y = np.append(y, batched_clean_mag, axis = 0)
    X = np.append(X, batched_mixed_mag, axis = 0)
    if(len(X) > 500 or it == size_train - 1):
        
        with h5py.File(hdf5_path, 'a') as hf:
            hf["x_train"].resize((hf["x_train"].shape[0] + X.shape[0]), axis = 0)
            hf["x_train"][-X.shape[0]:] = X

            hf["y_train"].resize((hf["y_train"].shape[0] + y.shape[0]), axis = 0)
            hf["y_train"][-y.shape[0]:] = y
        
        y = np.empty((0, 125, 257))
        X = np.empty((0, 125, 257))



for it, (mixed_path, clean_path) in enumerate(test_list):
    print(it/size_test)
    clean_data, sr = lb.load(clean_path, sr = 16000)
    mixed_data, sr = lb.load(mixed_path, sr = 16000)
    clean_STFT = mix.compute_stft(clean_data)
    mixed_STFT = mix.compute_stft(mixed_data)
    clean_mag = np.abs(clean_STFT)
    mixed_mag = np.abs(mixed_STFT)
    batched_clean_mag = mix.create_batches(clean_mag)
    batched_mixed_mag = mix.create_batches(mixed_mag)

    if (batched_clean_mag.shape[0] == 0):
        continue
    try:
        batched_mixed_mag = input_scaler.fit_transform(batched_mixed_mag.reshape(-1, batched_mixed_mag.shape[-1])).reshape(batched_mixed_mag.shape)
        batched_clean_mag = target_scaler.fit_transform(batched_clean_mag.reshape(-1, batched_clean_mag.shape[-1])).reshape(batched_clean_mag.shape)
    except:
        print(mixed_path)
        print(clean_path)
        quit()

    y = np.append(y, batched_clean_mag, axis = 0)
    X = np.append(X, batched_mixed_mag, axis = 0)


    if(len(X) > 500 or it == size_test - 1):

        with h5py.File(hdf5_path, 'a') as hf:
            hf["x_test"].resize((hf["x_test"].shape[0] + X.shape[0]), axis = 0)
            hf["x_test"][-X.shape[0]:] = X

            hf["y_test"].resize((hf["y_test"].shape[0] + y.shape[0]), axis = 0)
            hf["y_test"][-y.shape[0]:] = y

        y = np.empty((0, 125, 257))
        X = np.empty((0, 125, 257))

