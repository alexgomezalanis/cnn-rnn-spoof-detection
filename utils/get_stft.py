import numpy as np
import soundfile as sf
import math
from scipy import signal

def get_stft(file_path, n_filts, n_frames, nperseg, noverlap, nfft):
  sig, fs = sf.read(file_path, dtype='float32')

  f, t, Sxx =  signal.stft(sig, fs, window='blackman', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
  Sxx = np.abs(Sxx)
  features = np.where(Sxx > 1e-10, np.log10(Sxx), -10)

  data = np.reshape(features, (1, -1, n_filts)) # channel, frames, filters

  return data