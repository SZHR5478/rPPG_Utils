import numpy as np
import scipy
import pandas as pd
import glob
import os

from tqdm import tqdm
from scipy.signal import butter
from scipy.sparse import spdiags


def read_ppg_hr_csv(data_path, wave_fs=60):
    ppg_files = glob.glob(data_path + os.sep + "*_wave.csv")
    if not ppg_files:
        raise ValueError(data_path + "  is empty!")
    dirs = list()
    for i, ppg_file in enumerate(ppg_files):
        ppg = pd.read_csv(ppg_file)['Wave'].to_numpy()

        hr = pd.read_csv(os.path.splitext(ppg_file)[0][:-5] + ".csv")['PULSE'].to_numpy()

        if len(ppg) // len(hr) >= wave_fs:
            ppg = ppg[:wave_fs * len(hr)]
            hr = hr[:len(ppg) // wave_fs]
        else:
            hr = hr[:len(ppg) // wave_fs]
            ppg = ppg[:wave_fs * len(hr)]

        assert len(ppg) // len(hr) * len(hr) == len(ppg)

        dirs.append({"ppg": ppg, "hr": hr})
    return dirs


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal


def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak


def calculate_metric_per_ppg(predictions, fs=30, use_bandpass=True, hr_method='FFT'):
    """Calculate ppg-level HR"""
    predictions = _detrend(predictions, 100)
    if use_bandpass:
        # bandpass filter between [0.75, 2.5] Hz
        # equals [45, 150] beats per min
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))

    if hr_method == 'FFT':
        hr_pred = _calculate_fft_hr(predictions, fs=fs)
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')
    return hr_pred


def calculate_metrics(ppg_hrs, fs=60, window_size=300):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE)."""
    assert window_size // fs * fs == window_size
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    print("Calculating metrics!")
    for ppg_hr in tqdm(ppg_hrs):
        ppg = ppg_hr["ppg"]
        hr = ppg_hr["hr"]

        for i in range(0, len(ppg)):
            if i >= window_size - 1:
                ppg_window = ppg[i - window_size + 1:i + 1]
                pred_hr_peak = calculate_metric_per_ppg(ppg_window, fs=fs, hr_method='Peak')
                gt_hr_peak_all.append(hr[(i + 1) // fs - 1])
                predict_hr_peak_all.append(pred_hr_peak)

                pred_hr_fft = calculate_metric_per_ppg(ppg_window, fs=fs, hr_method='FFT')
                gt_hr_fft_all.append(hr[(i + 1) // fs - 1])
                predict_hr_fft_all.append(pred_hr_fft)

    gt_hr_fft_all = np.array(gt_hr_fft_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)

    gt_hr_peak_all = np.array(gt_hr_peak_all)
    predict_hr_peak_all = np.array(predict_hr_peak_all)

    res_df = pd.DataFrame({'gt_hr': gt_hr_fft_all, 'fft_hr': predict_hr_fft_all, 'peak_hr': predict_hr_peak_all})
    res_df.to_csv('gt_fft_peak.csv', index=False)

    num_test_samples = len(predict_hr_fft_all)

    MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
    standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
    print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))

    RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
    standard_error = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
    print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))

    MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
    standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(
        num_test_samples) * 100
    print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))

    MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
    standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
    print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))

    RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
    standard_error = np.std(np.square(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
    print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))

    MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
    standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(
        num_test_samples) * 100
    print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
