from scipy.io import wavfile


def read_wav_data_by_scipy(filename):
    sampling_freq, audio = wavfile.read(filename)
    return audio, sampling_freq


