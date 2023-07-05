import numpy as np
import pandas as pd
import soundfile as sf
import librosa as lb
import random
import math

hop_length = 128 # taille du saut, ici on utilise 1/4 de win length
win_length = 512 # taille en samples du signal avant padding
n_fft = win_length # taille de la fenetre apres padding, puissance de 2.
batch_sample_size = 16000//hop_length # taille en frames STFT d'un sample du batch , ici on prends 1 seconde.

def resample_to_16k(data, orig_sr):
    return lb.resample(data, orig_sr = orig_sr, target_sr = 16000)

def compute_rms(audio):
    return np.sqrt(np.mean(np.square(audio)))

def compute_SNR(signal, mixture):
    signal = signal[0:min(len(signal), len(mixture))]
    mixture = mixture[0:min(len(signal), len(mixture))]
    signal_power = np.mean(np.square(signal))
    residual = mixture - signal
    max_float32 = 1.
    min_float32 = -1.
    # Considerer le cas ou il y a depassement et le corriger le cas echeant
    if residual.max(axis=0) > max_float32 or residual.min(axis=0) < min_float32: 
        if residual.max(axis=0) >= abs(residual.min(axis=0)): 
            reduction_rate = max_float32 / residual.max(axis=0)
        else :
            reduction_rate = min_float32 / residual.min(axis=0)
        residual = residual * (reduction_rate)

    noise_power = np.mean(np.square(residual))
    ratio = signal_power / noise_power
    return 10*(math.log10( ratio ))

def compute_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a) 
    return noise_rms


def mix_noise(clean_audio, noise_audio, SNR_dB):

    
    while(clean_audio.shape[0] > noise_audio.shape[0]):
        noise_audio = np.concatenate((noise_audio, noise_audio))
    
    start = random.randint(0, len(noise_audio)-len(clean_audio))
    divided_noise = noise_audio[start: start + len(clean_audio)]
    
    clean_rms = compute_rms(clean_audio)
    noise_rms = compute_rms(divided_noise)
    adjusted_noise_rms = compute_adjusted_rms(clean_rms, SNR_dB)
    adjusted_noise = divided_noise * (adjusted_noise_rms / noise_rms) #On ajuste la puissance du bruit pour ce SNR choisi
    mixed_audio = clean_audio + adjusted_noise # Il y a peut etre depassement de -1 / +1
    max_float32 = 1.
    min_float32 = -1.
    # Considerer le cas ou il y a depassement et le corriger le cas echeant
    if mixed_audio.max(axis=0) > max_float32 or mixed_audio.min(axis=0) < min_float32: 
        if mixed_audio.max(axis=0) >= abs(mixed_audio.min(axis=0)): 
            reduction_rate = max_float32 / mixed_audio.max(axis=0)
        else :
            reduction_rate = min_float32 / mixed_audio.min(axis=0)
        mixed_audio = mixed_audio * (reduction_rate)
    
    return mixed_audio

def save_wav16k(data, path):
    if(data.dtype != 'int16'):
        data = (data * 32767).astype(np.int16)
    sf.write(path, data, 16000)

def compute_stft(data):
    return lb.stft(data, n_fft=n_fft, hop_length = hop_length, win_length=win_length)

def compute_istft(stft):
    return lb.istft(stft, n_fft=n_fft, hop_length = hop_length, win_length=win_length)

def create_batches(stft_shape):
    end = stft_shape.shape[1]//batch_sample_size
    stft_trimmed = stft_shape[:,:end*batch_sample_size].T
    batch_stft = np.resize(stft_trimmed, (end, batch_sample_size, stft_trimmed.shape[1]))
    return batch_stft

if __name__ == '__main__':

    signal = np.zeros(64000)
    stft = compute_stft(signal)
    b_stft = create_batches(stft)
    print(b_stft.shape)