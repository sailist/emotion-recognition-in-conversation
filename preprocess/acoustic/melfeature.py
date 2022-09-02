import sys
import numpy as np


def win_func_hamming(x):
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(0, x, dtype=np.float32) / (x - 1))


def win_func_blackman(x):
    return 0.42 - 0.5 * np.cos(2.0 * np.arange(0, x, dtype=np.float32) * np.pi / x) + 0.08 * np.cos(
        4.0 * np.arange(0, x, dtype=np.float32) * np.pi / x)


def win_func_povey(x):
    return np.power((0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(0, x, dtype=np.float32) / (x - 1))), 0.85)


def gcd(a, b):
    if a % b == 0:
        return b
    else:
        return gcd(b, a % b)


def rolling_window(wav_data, frm_size=400, frm_sft=160):
    cn, wav_len = wav_data.shape

    sub_frm_size = gcd(frm_size, frm_sft)
    frm_size_1 = frm_size // sub_frm_size
    frm_sft_1 = frm_sft // sub_frm_size

    frm_num = (wav_len - frm_size) // frm_sft + 1
    sub_frame_selector_row = np.reshape(np.arange(0, frm_num) * frm_sft_1, [frm_num, 1])
    sub_frame_selector_col = np.reshape(np.arange(0, frm_size_1), [1, frm_size_1])
    sub_frame_selector = sub_frame_selector_row + sub_frame_selector_col

    wav_len_1 = wav_len // sub_frm_size
    sub_frm_data = wav_data[:, :wav_len_1 * sub_frm_size]
    sub_frm_data = np.reshape(sub_frm_data, [cn, wav_len_1, sub_frm_size])

    # frame select
    frm_data = np.take(sub_frm_data, sub_frame_selector, axis=-2)
    frm_data = np.reshape(frm_data, [cn, frm_num, frm_size])

    return frm_data


def hz2mel(hz):
    return 1127.0 * np.log(1 + hz / 700.0)


def mel2hz(mel):
    return (np.exp(mel / 1127.0) - 1) * 700.0


def get_mel_filter(fb_dim=80, fft_size=512, sample_rate=16000, low_freq=0.0, high_freq=8000.0):
    high_freq = high_freq or sample_rate / 2
    assert high_freq <= sample_rate / 2, "highfreq is greater than samplerate/2"

    low_mel = hz2mel(low_freq)
    high_mel = hz2mel(high_freq)
    filtfreq = np.linspace(low_mel, high_mel, fb_dim + 2)

    fbank = np.zeros([fft_size // 2 + 1, fb_dim])
    for i in range(0, fft_size // 2 + 1):
        freq = hz2mel(sample_rate * i * 1.0 / fft_size)
        for j in range(0, fb_dim):
            if filtfreq[j] <= freq <= filtfreq[j + 1]:
                fbank[i, j] = (freq - filtfreq[j]) / (filtfreq[j + 1] - filtfreq[j])
            if filtfreq[j + 1] < freq <= filtfreq[j + 2]:
                fbank[i, j] = (filtfreq[j + 2] - freq) / (filtfreq[j + 2] - filtfreq[j + 1])
    return fbank[1:, :].astype(np.float32)


# -------------------------------------------------------------------------------------------------------------
def get_mfcc_coef(mfcc_dim=23, fb_dim=40):
    Q = 22.0
    normalizer_1 = np.sqrt(1.0 / fb_dim)
    normalizer_2 = np.sqrt(2.0 / fb_dim)
    mfcc_coef = np.zeros(shape=[fb_dim, mfcc_dim])
    for i in range(mfcc_dim):
        cc = 1.0 + 0.5 * Q * np.sin(np.pi * i / Q)
        for j in range(fb_dim):
            if i == 0:
                mfcc_coef[j, i] = normalizer_1 * cc
            else:
                mfcc_coef[j, i] = normalizer_2 * np.cos(np.pi * (j + 0.5) * i / fb_dim) * cc

    return mfcc_coef.astype(np.float32)


# -------------------------------------------------------------------------------------------------------------
def wav_to_stft(sig, frm_size=400, frm_sft=160, fft_size=400, win_func=win_func_hamming):
    cn, sig_len = sig.shape
    frm_num = (sig_len - frm_size) // frm_sft + 1

    # split
    frames = rolling_window(sig, frm_size=frm_size, frm_sft=frm_sft)
    frames = np.reshape(frames, [-1, frames.shape[-1]])

    # remove dc
    frames = frames - np.mean(frames, axis=-1, keepdims=True)

    # PreEm
    coeff = 0.97
    frames = np.append(frames[:, 0:1] * (1 - coeff), frames[:, 1:] - coeff * frames[:, :-1], axis=1)

    # add win
    frames = frames * win_func(frm_size)

    # fft
    stft = np.fft.rfft(frames).astype(np.complex64)

    # out
    stft = np.reshape(stft, [cn, frm_num, fft_size // 2 + 1])

    return stft


def wav_to_fb(sig, frm_size=400, frm_sft=160, fft_size=512, win_func=win_func_hamming, low_freq=0.0, high_freq=8000.0,
              fb_dim=80):
    cn, sig_len = sig.shape
    frm_num = (sig_len - frm_size) // frm_sft + 1

    # split
    frames = rolling_window(sig, frm_size=frm_size, frm_sft=frm_sft)
    frames = np.reshape(frames, [-1, frames.shape[-1]])

    # remove dc
    frames = frames - np.mean(frames, axis=-1, keepdims=True)
    # PreEm
    coeff = 0.97
    frames = np.append(frames[:, 0:1] * (1 - coeff), frames[:, 1:] - coeff * frames[:, :-1], axis=1)

    # add win
    frames = frames * win_func(frm_size)

    # sfft
    frames = np.pad(frames, [[0, 0], [0, fft_size - frm_size]], 'constant')
    stft = np.fft.rfft(frames).astype(np.complex64)
    stft = stft[:, 1:fft_size // 2 + 1]

    # fb
    mel_filter = get_mel_filter(fb_dim=fb_dim, fft_size=fft_size, sample_rate=16000, low_freq=low_freq,
                                high_freq=high_freq)

    psd = stft.real * stft.real + stft.imag * stft.imag + 0.000001
    fb = np.matmul(psd, mel_filter)
    fb = np.log(fb)
    fb = np.reshape(fb, newshape=[cn, frm_num, -1])

    return fb


def wav_to_mfcc(sig, frm_size=400, frm_sft=160, fft_size=512, win_func=win_func_povey, low_freq=0.0, high_freq=8000.0,
                fb_dim=40, mfcc_dim=23):
    cn, sig_len = sig.shape
    frm_num = (sig_len - frm_size) // frm_sft + 1

    # split
    frames = rolling_window(sig, frm_size=frm_size, frm_sft=frm_sft)
    frames = np.reshape(frames, [-1, frames.shape[-1]])

    # remove dc
    frames = frames - np.mean(frames, axis=-1, keepdims=True)

    # en
    energy = np.log(np.sum(frames * frames, axis=-1, keepdims=True) + 0.000001)

    # PreEm
    coeff = 0.97
    frames = np.append(frames[:, 0:1] * (1 - coeff), frames[:, 1:] - coeff * frames[:, :-1], axis=1)

    # add win
    frames = frames * win_func(frm_size)

    # stft
    frames = np.pad(frames, [[0, 0], [0, fft_size - frm_size]], 'constant')
    stft = np.fft.rfft(frames).astype(np.complex64)
    stft = stft[:, 1:fft_size // 2 + 1]

    # fb
    mel_filter = get_mel_filter(fb_dim=fb_dim, fft_size=fft_size, sample_rate=16000, low_freq=low_freq,
                                high_freq=high_freq)
    psd = stft.real * stft.real + stft.imag * stft.imag + 0.000001
    fb = np.matmul(psd, mel_filter)
    fb = np.log(fb)

    # mfcc
    mfcc_coef = get_mfcc_coef(mfcc_dim=mfcc_dim, fb_dim=fb_dim)
    mfcc = np.matmul(fb, mfcc_coef)
    mfcc = np.concatenate((energy, mfcc[:, 1:]), axis=-1)
    mfcc = np.reshape(mfcc, newshape=[cn, frm_num, -1])

    return mfcc


if __name__ == '__main__':
    print(wav_to_fb(np.zeros((1, 512))).shape)
