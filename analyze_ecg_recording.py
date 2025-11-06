#!/usr/bin/env python3
"""
Analyze recorded ECG data from CSV file.

Performs:
- R-peak detection
- HRV analysis (SDNN, RMSSD, pNN50)
- ECG quality metrics
- Generates plots

Usage: python analyze_ecg_recording.py <ecg_csv_file>
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Bandpass filter for ECG."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def detect_r_peaks(ecg_data, fs=130):
    """
    Detect R-peaks in ECG signal.

    Uses Pan-Tompkins-inspired approach:
    1. Bandpass filter (5-15 Hz for QRS)
    2. Differentiation
    3. Squaring
    4. Moving average
    5. Adaptive thresholding
    """

    # 1. Bandpass filter (5-15 Hz for QRS complex)
    filtered = bandpass_filter(ecg_data, 5, 15, fs, order=2)

    # 2. Differentiation (approximation)
    diff = np.diff(filtered)
    diff = np.append(diff, 0)

    # 3. Squaring
    squared = diff ** 2

    # 4. Moving average (integration)
    window = int(0.150 * fs)  # 150ms window
    integrated = np.convolve(squared, np.ones(window)/window, mode='same')

    # 5. Peak detection with adaptive threshold
    # Use median for robustness
    threshold = np.median(integrated) + 1.5 * np.std(integrated)

    # Find peaks
    peaks, properties = find_peaks(integrated,
                                   height=threshold,
                                   distance=int(fs * 0.4),  # Min 400ms between beats
                                   prominence=threshold * 0.3)

    # Refine peak locations to actual R-peak in original signal
    refined_peaks = []
    search_window = int(0.05 * fs)  # 50ms search window

    for peak in peaks:
        start = max(0, peak - search_window)
        end = min(len(ecg_data), peak + search_window)
        local_max = np.argmax(ecg_data[start:end]) + start
        refined_peaks.append(local_max)

    return np.array(refined_peaks), integrated

def calculate_hrv_metrics(rr_intervals):
    """Calculate time-domain HRV metrics."""
    rr_ms = rr_intervals * 1000

    # Remove outliers (0.3-2.0 seconds = 30-200 BPM)
    valid_rr = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]

    if len(valid_rr) < 2:
        return None

    # Time domain metrics
    sdnn = np.std(valid_rr)
    mean_rr = np.mean(valid_rr)
    mean_hr = 60000 / mean_rr

    # RMSSD: Root mean square of successive differences
    diff_rr = np.diff(valid_rr)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))

    # pNN50: Percentage of successive intervals differing by >50ms
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100

    return {
        'mean_hr': mean_hr,
        'mean_rr': mean_rr,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'pnn50': pnn50,
        'valid_beats': len(valid_rr),
        'rejected_beats': len(rr_ms) - len(valid_rr)
    }

def plot_analysis(ecg_data, timestamps, r_peaks, rr_intervals, hrv_metrics):
    """Generate comprehensive analysis plots."""

    fs = len(ecg_data) / (timestamps[-1] - timestamps[0])

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # 1. Full ECG with R-peaks
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timestamps, ecg_data, 'b-', linewidth=0.5, alpha=0.7, label='ECG')
    ax1.plot(timestamps[r_peaks], ecg_data[r_peaks], 'ro',
             markersize=6, label=f'R-peaks (n={len(r_peaks)})')
    ax1.set_ylabel('ECG (µV)')
    ax1.set_title('ECG with Detected R-peaks')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Zoomed ECG (first 10 seconds)
    ax2 = fig.add_subplot(gs[1, 0])
    zoom_samples = int(10 * fs)
    ax2.plot(timestamps[:zoom_samples], ecg_data[:zoom_samples], 'b-', linewidth=1)
    zoom_peaks = r_peaks[r_peaks < zoom_samples]
    ax2.plot(timestamps[zoom_peaks], ecg_data[zoom_peaks], 'ro', markersize=8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('ECG (µV)')
    ax2.set_title('ECG Detail (first 10s)')
    ax2.grid(True, alpha=0.3)

    # 3. RR interval histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(rr_intervals * 1000, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax3.axvline(hrv_metrics['mean_rr'], color='red', linestyle='--',
                linewidth=2, label=f"Mean: {hrv_metrics['mean_rr']:.0f} ms")
    ax3.set_xlabel('RR Interval (ms)')
    ax3.set_ylabel('Count')
    ax3.set_title('RR Interval Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Tachogram (RR intervals over time)
    ax4 = fig.add_subplot(gs[2, :])
    rr_times = timestamps[r_peaks[1:]]
    ax4.plot(rr_times, rr_intervals * 1000, 'g.-', linewidth=1, markersize=3)
    ax4.axhline(hrv_metrics['mean_rr'], color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('RR Interval (ms)')
    ax4.set_title('Tachogram (RR Intervals Over Time)')
    ax4.grid(True, alpha=0.3)

    # 5. Heart rate over time
    ax5 = fig.add_subplot(gs[3, :])
    hr = 60 / rr_intervals
    ax5.plot(rr_times, hr, 'r.-', linewidth=1, markersize=3)
    ax5.axhline(hrv_metrics['mean_hr'], color='blue', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Heart Rate (BPM)')
    ax5.set_title('Heart Rate Over Time')
    ax5.grid(True, alpha=0.3)

    # Add text box with HRV metrics
    metrics_text = (
        f"HRV Metrics:\n"
        f"Mean HR: {hrv_metrics['mean_hr']:.1f} BPM\n"
        f"Mean RR: {hrv_metrics['mean_rr']:.0f} ms\n"
        f"SDNN: {hrv_metrics['sdnn']:.1f} ms\n"
        f"RMSSD: {hrv_metrics['rmssd']:.1f} ms\n"
        f"pNN50: {hrv_metrics['pnn50']:.1f} %\n"
        f"Valid beats: {hrv_metrics['valid_beats']}\n"
        f"Rejected: {hrv_metrics['rejected_beats']}"
    )

    fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontfamily='monospace')

    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_ecg_recording.py <ecg_csv_file>")
        print("Example: python analyze_ecg_recording.py recording_20251104_123456_ecg.csv")
        sys.exit(1)

    ecg_file = sys.argv[1]

    print(f"Loading ECG data: {ecg_file}")
    df = pd.read_csv(ecg_file)

    if 'ecg_uv' not in df.columns or 'timestamp' not in df.columns:
        print("ERROR: CSV must have 'timestamp' and 'ecg_uv' columns")
        sys.exit(1)

    ecg_data = df['ecg_uv'].values
    timestamps = (df['timestamp'] - df['timestamp'].iloc[0]).values

    print(f"Loaded {len(ecg_data)} samples")
    print(f"Duration: {timestamps[-1]:.1f} seconds")

    fs = len(ecg_data) / timestamps[-1]
    print(f"Sample rate: {fs:.1f} Hz")

    # Detect R-peaks
    print("\nDetecting R-peaks...")
    r_peaks, _ = detect_r_peaks(ecg_data, fs=fs)
    print(f"Found {len(r_peaks)} R-peaks")

    if len(r_peaks) < 2:
        print("ERROR: Not enough R-peaks detected")
        print("Check signal quality or adjust detection parameters")
        sys.exit(1)

    # Calculate RR intervals
    rr_intervals = np.diff(timestamps[r_peaks])
    print(f"Mean RR: {np.mean(rr_intervals)*1000:.0f} ms")

    # HRV analysis
    print("\nCalculating HRV metrics...")
    hrv_metrics = calculate_hrv_metrics(rr_intervals)

    if hrv_metrics is None:
        print("ERROR: Unable to calculate HRV metrics")
        sys.exit(1)

    print("\nHRV Metrics:")
    print(f"  Mean HR:     {hrv_metrics['mean_hr']:.1f} BPM")
    print(f"  Mean RR:     {hrv_metrics['mean_rr']:.0f} ms")
    print(f"  SDNN:        {hrv_metrics['sdnn']:.1f} ms")
    print(f"  RMSSD:       {hrv_metrics['rmssd']:.1f} ms")
    print(f"  pNN50:       {hrv_metrics['pnn50']:.1f} %")
    print(f"  Valid beats: {hrv_metrics['valid_beats']}")

    if hrv_metrics['rejected_beats'] > 0:
        print(f"  Rejected:    {hrv_metrics['rejected_beats']} outliers")

    # Save R-peak data
    output_file = ecg_file.replace('.csv', '_analysis.csv')
    peak_df = pd.DataFrame({
        'r_peak_sample': r_peaks,
        'r_peak_time': timestamps[r_peaks],
    })
    peak_df['rr_interval_s'] = np.concatenate([[np.nan], rr_intervals])
    peak_df['hr_bpm'] = 60 / peak_df['rr_interval_s']

    peak_df.to_csv(output_file, index=False)
    print(f"\n✓ R-peak data saved to: {output_file}")

    # Generate plots
    print("\nGenerating plots...")
    plot_analysis(ecg_data, timestamps, r_peaks, rr_intervals, hrv_metrics)

if __name__ == "__main__":
    main()
