from Outils import mix
import soundfile as sf
import librosa as lb
import random
from numpy.random import default_rng
import pandas as pd
import os

file_clean_path = r'/home/yelkhanoussi/Projet_DL/debruitage_samer_yassine/audio_files/dev-clean/clean_files.txt'
file_noise_path = r'/home/yelkhanoussi/Projet_DL/debruitage_samer_yassine/audio_files/stationary_noise/noise_files.txt'
mixture_path = r'/home/yelkhanoussi/Projet_DL/debruitage_samer_yassine/audio_files/mixture/'
SNR_LOW = 0
SNR_HIGH = 10


with open(file_clean_path) as f:
    list_clean = f.read().splitlines()

with open(file_noise_path) as f:
    list_noise = f.read().splitlines()

n_list = len(list_clean)

while(n_list > len(list_noise)):
    list_noise = 2*list_noise

list_noise = list_noise[:n_list]

n_list = len(list_clean)
noise_size = len(list_noise)
print(f'clean size: {n_list}, noise size : {noise_size}')

random.shuffle(list_noise)
random.shuffle(list_clean)

rng = default_rng()
list_SNR = rng.uniform(SNR_LOW, SNR_HIGH, n_list)
list_mixture = list(zip(list_clean, list_noise, list_SNR))

list_path_mix = []

noise, sr_noise = lb.load(os.fspath('/home/yelkhanoussi/Projet_DL/debruitage_samer_yassine/audio_test/babble.wav'))
if(sr_noise != 16000):
    noise = mix.resample_to_16k(noise, sr_noise)

for i, (clean_path, noise_path, snr) in enumerate(list_mixture):
    print(i)
    clean_name = (clean_path.split('/')[-1]).split('.')[0]
    noise_name = 'babble' #noise_path.split('/')[-1].split('.')[0]
    clean, sr = lb.load(clean_path)
    if(sr != 16000):
        clean = mix.resample_to_16k(clean, sr)
    # noise, sr_noise = lb.load(os.fspath(noise_path))
    # if(sr_noise != 16000):
    #     noise = mix.resample_to_16k(noise, sr_noise)

    mixed = mix.mix_noise(clean, noise, snr)
    mixed_name = clean_name + '__' + noise_name
    mixed_path = mixture_path + mixed_name + '.wav'
    # sf.write(path, 16000, mixed)
    mix.save_wav16k(mixed, mixed_path)
    list_path_mix.append(mixed_path)

mixture_df = pd.DataFrame(list_mixture, columns=['path_clean', 'path_noise', 'SNR'])
mixture_df['mixture'] = list_path_mix
mixture_df.to_csv('mixtures_1h_babble_v2.csv', sep=',')