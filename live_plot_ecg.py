#!/usr/bin/env python3
"""
Real-time ECG plotter for Polar H10 raw ECG stream.

This plots the 130 Hz ECG waveform with R-peak detection.

Usage: python live_plot_ecg.py
(Requires polar_ecg_to_lsl.py running in another terminal)
"""

import collections
import queue
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylsl import resolve_byprop, StreamInlet
from scipy.signal import find_peaks

def inlet_thread(out_q, stop_event):
    """Background thread to read ECG stream."""
    print("Looking for PolarH10_ECG stream...")
    streams = resolve_byprop('type', 'ECG', timeout=10)

    if not streams:
        print("ERROR: ECG stream not found!")
        print("Make sure polar_ecg_to_lsl.py is running.")
        return

    print(f"Found ECG stream: {streams[0].name()} @ {streams[0].nominal_srate()} Hz")
    inlet = StreamInlet(streams[0], max_buflen=360)

    while not stop_event.is_set():
        sample, ts = inlet.pull_sample(timeout=0.1)
        if sample is not None:
            out_q.put((ts, sample[0]))

def detect_r_peaks_online(ecg_buffer, fs=130):
    """Simple online R-peak detection."""
    if len(ecg_buffer) < fs:  # Need at least 1 second
        return []

    # Use recent data for threshold
    recent = list(ecg_buffer)[-int(fs * 3):]  # Last 3 seconds
    threshold = np.mean(recent) + 2 * np.std(recent)

    # Find peaks in last 0.5 seconds
    search_window = list(ecg_buffer)[-int(fs * 0.5):]
    peaks, _ = find_peaks(search_window,
                         height=threshold,
                         distance=int(fs * 0.4))  # Min 400ms between peaks

    # Convert to absolute indices
    offset = len(ecg_buffer) - len(search_window)
    return peaks + offset

def main():
    # Buffers
    win_sec = 5  # Show 5 seconds of ECG
    fs = 130

    ecg_q = queue.Queue()
    stop_event = threading.Event()

    times = collections.deque(maxlen=fs * win_sec)
    ecg_data = collections.deque(maxlen=fs * win_sec)

    # R-peak tracking
    r_peak_times = collections.deque(maxlen=50)  # Keep last 50 R-peaks
    rr_intervals = collections.deque(maxlen=49)
    heart_rates = collections.deque(maxlen=49)

    # Start reader thread
    t0 = time.time()
    reader = threading.Thread(target=inlet_thread, args=(ecg_q, stop_event), daemon=True)
    reader.start()

    # Give thread time to connect
    time.sleep(2)

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

    # ECG waveform
    ln_ecg, = ax1.plot([], [], 'b-', linewidth=1)
    r_peak_scatter = ax1.scatter([], [], c='red', s=100, marker='o',
                                 label='R-peaks', zorder=5)
    ax1.set_ylabel('ECG (µV)')
    ax1.set_title('Raw ECG Waveform (130 Hz)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # RR intervals (tachogram)
    ln_rr, = ax2.plot([], [], 'g.-', linewidth=1.5, markersize=4)
    ax2.set_ylabel('RR Interval (ms)')
    ax2.set_title('RR Intervals (Tachogram)')
    ax2.grid(True, alpha=0.3)

    # Heart rate
    ln_hr, = ax3.plot([], [], 'r.-', linewidth=1.5, markersize=4)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Heart Rate (BPM)')
    ax3.set_title('Instantaneous Heart Rate')
    ax3.grid(True, alpha=0.3)

    # Status text
    status_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    last_r_peak_idx = -1

    def update(frame):
        nonlocal last_r_peak_idx

        # Drain queue
        sample_count = 0
        while True:
            try:
                ts, ecg = ecg_q.get_nowait()
                times.append(ts - t0)
                ecg_data.append(ecg)
                sample_count += 1
            except queue.Empty:
                break

        if len(times) < 2:
            return ln_ecg, r_peak_scatter, ln_rr, ln_hr, status_text

        # Detect R-peaks
        r_peaks = detect_r_peaks_online(ecg_data, fs)

        # Update R-peak list if new peak detected
        if len(r_peaks) > 0:
            latest_peak_idx = r_peaks[-1]
            if latest_peak_idx != last_r_peak_idx:
                last_r_peak_idx = latest_peak_idx

                # Add to history
                peak_time = list(times)[latest_peak_idx]
                r_peak_times.append(peak_time)

                # Calculate RR interval
                if len(r_peak_times) >= 2:
                    rr = r_peak_times[-1] - r_peak_times[-2]
                    rr_intervals.append(rr)
                    hr = 60 / rr
                    heart_rates.append(hr)

        # Update ECG plot
        t_array = np.array(times)
        ecg_array = np.array(ecg_data)

        tmin = max(0, t_array[-1] - win_sec)
        mask = t_array >= tmin

        ln_ecg.set_data(t_array[mask], ecg_array[mask])
        ax1.set_xlim(tmin, t_array[-1])

        if len(ecg_array[mask]) > 0:
            ymin, ymax = np.percentile(ecg_array[mask], [1, 99])
            yrange = ymax - ymin
            ax1.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)

        # Update R-peak scatter
        if len(r_peak_times) > 0:
            recent_peaks = [t for t in r_peak_times if t >= tmin]
            if recent_peaks:
                peak_indices = [np.argmin(np.abs(t_array - t)) for t in recent_peaks]
                r_peak_scatter.set_offsets(np.c_[
                    [t_array[i] for i in peak_indices],
                    [ecg_array[i] for i in peak_indices]
                ])

        # Update RR intervals plot
        if len(rr_intervals) > 0:
            rr_times = list(r_peak_times)[1:]  # Times of second beat in each pair
            ln_rr.set_data(rr_times, np.array(rr_intervals) * 1000)
            ax2.relim()
            ax2.autoscale_view()

        # Update heart rate plot
        if len(heart_rates) > 0:
            ln_hr.set_data(rr_times, heart_rates)
            ax3.relim()
            ax3.autoscale_view()

        # Update status text
        current_hr = heart_rates[-1] if len(heart_rates) > 0 else 0
        current_rr = rr_intervals[-1] * 1000 if len(rr_intervals) > 0 else 0
        mean_hr = np.mean(heart_rates) if len(heart_rates) > 0 else 0

        status_text.set_text(
            f'Samples: {len(times)} | R-peaks: {len(r_peak_times)}\n'
            f'Current HR: {current_hr:.0f} BPM | RR: {current_rr:.0f} ms\n'
            f'Mean HR: {mean_hr:.0f} BPM'
        )

        return ln_ecg, r_peak_scatter, ln_rr, ln_hr, status_text

    print("\nStarting ECG visualization...")
    print("Close window or press Ctrl+C to stop.\n")

    ani = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    plt.tight_layout()

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()

if __name__ == "__main__":
    main()
