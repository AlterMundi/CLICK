#!/usr/bin/env python3
"""
Extract cardiac artifact from EEG recording.

This is often more accurate than using separate Polar HR timestamps
because the cardiac artifact is ALREADY SYNCHRONIZED with the EEG!

Usage: python extract_cardiac_from_eeg.py <eeg_csv_file>
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def extract_cardiac_artifact(eeg_data, fs=256, channel_idx=0):
    """
    Extract cardiac artifact from EEG channel.

    Best results typically from:
    - Temporal channels (TP9, TP10) - close to major blood vessels
    - Occipital channels - posterior circulation

    Args:
        eeg_data: 2D array (samples x channels)
        fs: Sampling rate (Hz)
        channel_idx: Which channel to analyze (0=TP9, 3=TP10 for Muse)

    Returns:
        dict with cardiac_signal, r_peaks, rr_intervals, etc.
    """

    print(f"Extracting cardiac artifact from channel {channel_idx}...")

    # 1. Bandpass filter for cardiac frequencies
    # Heart rate 40-120 BPM = 0.67-2.0 Hz
    cardiac_signal = bandpass_filter(eeg_data[:, channel_idx],
                                     lowcut=0.6,
                                     highcut=2.5,
                                     fs=fs)

    # 2. Adaptive peak detection
    # Use median absolute deviation for robust threshold
    mad = np.median(np.abs(cardiac_signal - np.median(cardiac_signal)))
    threshold = mad * 3

    # Minimum distance between peaks = 400ms (max 150 BPM)
    min_distance = int(fs * 0.4)

    peaks, properties = find_peaks(cardiac_signal,
                                   height=threshold,
                                   distance=min_distance,
                                   prominence=threshold * 0.5)

    if len(peaks) < 2:
        print(f"WARNING: Only {len(peaks)} peaks detected. Try different channel or check data quality.")
        return None

    # 3. Calculate metrics
    peak_times = peaks / fs
    rr_intervals = np.diff(peak_times)

    # Remove outliers (RR intervals outside 0.3-2.0 seconds = 30-200 BPM)
    valid_rr = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]

    if len(valid_rr) < len(rr_intervals) * 0.8:
        print(f"WARNING: {len(rr_intervals) - len(valid_rr)} outlier RR intervals removed")

    mean_hr = 60 / np.mean(valid_rr) if len(valid_rr) > 0 else 0

    # HRV metrics
    rr_ms = valid_rr * 1000
    sdnn = np.std(rr_ms)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2)) if len(rr_ms) > 1 else 0

    return {
        'cardiac_signal': cardiac_signal,
        'raw_eeg': eeg_data[:, channel_idx],
        'r_peaks': peaks,
        'peak_times': peak_times,
        'rr_intervals': valid_rr,
        'mean_hr': mean_hr,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'channel': channel_idx
    }

def compare_all_channels(eeg_data, timestamps, fs=256):
    """
    Compare cardiac artifact detection across all channels.
    Returns the best channel.
    """
    print("\nComparing cardiac artifact across all channels...")
    results = []

    for ch in range(eeg_data.shape[1]):
        result = extract_cardiac_artifact(eeg_data, fs, ch)
        if result is not None:
            results.append((ch, len(result['r_peaks']), result['mean_hr']))
            print(f"  Channel {ch}: {len(result['r_peaks'])} peaks, HR={result['mean_hr']:.1f} BPM")

    if not results:
        return None

    # Best channel = most peaks detected (indicates strongest artifact)
    best_ch = max(results, key=lambda x: x[1])[0]
    print(f"\n→ Best channel: {best_ch} (most consistent peaks)")

    return best_ch

def plot_results(result, timestamps, duration_plot=10):
    """Plot cardiac artifact extraction results."""

    fs = len(result['cardiac_signal']) / timestamps[-1]
    n_samples_plot = int(duration_plot * fs)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    # 1. Raw EEG
    ax = axes[0]
    t = timestamps[:n_samples_plot]
    ax.plot(t, result['raw_eeg'][:n_samples_plot], 'k-', linewidth=0.5, alpha=0.7)
    ax.set_ylabel('Raw EEG (µV)')
    ax.set_title(f'Channel {result["channel"]}: Raw EEG with Cardiac Artifact')
    ax.grid(True, alpha=0.3)

    # 2. Filtered cardiac signal
    ax = axes[1]
    ax.plot(t, result['cardiac_signal'][:n_samples_plot], 'r-', linewidth=1)

    # Mark detected R-peaks
    peaks_in_window = result['r_peaks'][result['r_peaks'] < n_samples_plot]
    ax.plot(peaks_in_window / fs,
            result['cardiac_signal'][peaks_in_window],
            'go', markersize=8, label='Detected R-peaks')

    ax.set_ylabel('Cardiac Signal (filtered)')
    ax.set_title('Bandpass Filtered (0.6-2.5 Hz) - Cardiac Artifact')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. RR intervals (tachogram)
    ax = axes[2]
    rr_times = result['peak_times'][1:]  # RR[i] is time between peak[i] and peak[i+1]
    ax.plot(rr_times, result['rr_intervals'] * 1000, 'b.-', linewidth=1.5, markersize=4)
    ax.axhline(np.mean(result['rr_intervals']) * 1000, color='r',
               linestyle='--', label=f'Mean: {np.mean(result["rr_intervals"])*1000:.0f} ms')
    ax.set_ylabel('RR Interval (ms)')
    ax.set_title('RR Intervals (Tachogram)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Heart rate
    ax = axes[3]
    hr = 60 / result['rr_intervals']
    ax.plot(rr_times, hr, 'g.-', linewidth=1.5, markersize=4)
    ax.axhline(result['mean_hr'], color='r',
               linestyle='--', label=f'Mean: {result["mean_hr"]:.1f} BPM')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heart Rate (BPM)')
    ax.set_title('Instantaneous Heart Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def save_cardiac_data(result, timestamps, output_file):
    """Save extracted cardiac data to CSV."""

    # Create DataFrame with R-peak times and RR intervals
    peak_times = result['peak_times']
    r_peaks = result['r_peaks']
    rr_intervals = result['rr_intervals']

    # RR intervals are between consecutive peaks, so there's one less than peaks
    # Pad with NaN at the beginning
    rr_padded = np.concatenate([[np.nan], rr_intervals])

    # Ensure lengths match (in case outliers were removed)
    if len(rr_padded) < len(peak_times):
        # Pad with NaN to match
        rr_padded = np.concatenate([rr_padded, [np.nan] * (len(peak_times) - len(rr_padded))])
    elif len(rr_padded) > len(peak_times):
        # Trim to match
        rr_padded = rr_padded[:len(peak_times)]

    df = pd.DataFrame({
        'r_peak_time': peak_times,
        'r_peak_sample': r_peaks,
        'rr_interval_s': rr_padded,
    })

    df['hr_bpm'] = 60 / df['rr_interval_s']

    df.to_csv(output_file, index=False)
    print(f"\n✓ Cardiac data saved to: {output_file}")
    print(f"  {len(result['r_peaks'])} R-peaks")
    print(f"  Mean HR: {result['mean_hr']:.1f} BPM")
    print(f"  SDNN: {result['sdnn']:.1f} ms")
    print(f"  RMSSD: {result['rmssd']:.1f} ms")

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_cardiac_from_eeg.py <eeg_csv_file>")
        print("Example: python extract_cardiac_from_eeg.py recording_20251104_022518_eeg.csv")
        sys.exit(1)

    eeg_file = sys.argv[1]

    print(f"Loading EEG data: {eeg_file}")
    df = pd.read_csv(eeg_file)

    # Extract EEG channels (assuming standard Muse format)
    channel_cols = [col for col in df.columns if col.startswith('ch') or
                    col in ['TP9', 'AF7', 'AF8', 'TP10']]

    if not channel_cols:
        print("ERROR: No EEG channels found in CSV")
        sys.exit(1)

    eeg_data = df[channel_cols].values
    timestamps = (df['timestamp'] - df['timestamp'].iloc[0]).values

    print(f"Loaded {len(eeg_data)} samples, {len(channel_cols)} channels")
    print(f"Duration: {timestamps[-1]:.1f} seconds")
    print(f"Channels: {channel_cols}")

    # Find best channel
    best_channel = compare_all_channels(eeg_data, timestamps)

    if best_channel is None:
        print("\nERROR: No cardiac artifact detected in any channel")
        print("Possible reasons:")
        print("  - Signal quality too poor")
        print("  - Recording too short")
        print("  - Cardiac artifact too weak in these channels")
        sys.exit(1)

    # Extract from best channel
    result = extract_cardiac_artifact(eeg_data, fs=256, channel_idx=best_channel)

    # Save results
    output_file = eeg_file.replace('.csv', '_cardiac.csv')
    save_cardiac_data(result, timestamps, output_file)

    # Plot
    print("\nGenerating plots...")
    plot_results(result, timestamps, duration_plot=30)

if __name__ == "__main__":
    main()
