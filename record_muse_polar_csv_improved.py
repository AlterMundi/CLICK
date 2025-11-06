#!/usr/bin/env python3
"""
Improved version of record_muse_polar_csv.py with:
1. Inlet buffer flushing for cleaner start
2. Metadata logging (flush time, connection info)
3. Better timestamp analysis

Usage: python record_muse_polar_csv_improved.py [duration_seconds]
"""

import csv
import time
import threading
import queue
from datetime import datetime
from pylsl import resolve_byprop, StreamInlet, local_clock

def record_streams(duration_sec=60, prefix=None):
    """Record both EEG and Polar HR streams to separate CSV files."""
    if prefix is None:
        prefix = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    eeg_file = f"{prefix}_eeg.csv"
    hr_file = f"{prefix}_hr.csv"
    meta_file = f"{prefix}_metadata.txt"

    print(f"Looking for streams...")
    eeg_streams = resolve_byprop('type', 'EEG', timeout=5)
    hr_streams = resolve_byprop('name', 'PolarH10_HR', timeout=5)

    if not eeg_streams:
        print("ERROR: EEG stream not found!")
        return
    if not hr_streams:
        print("ERROR: PolarH10_HR stream not found!")
        return

    eeg_info = eeg_streams[0]
    hr_info = hr_streams[0]

    print(f"Found EEG stream: {eeg_info.name()} ({eeg_info.channel_count()} channels @ {eeg_info.nominal_srate()} Hz)")
    print(f"Found HR stream: {hr_info.name()} ({hr_info.channel_count()} channels)")

    # Create inlets
    eeg_inlet = StreamInlet(eeg_info, max_buflen=360)
    hr_inlet = StreamInlet(hr_info, max_buflen=60)

    num_eeg_ch = eeg_info.channel_count()

    # IMPROVEMENT 1: Flush buffers for cleaner start
    print("\nWaiting for buffers to fill (1 second)...")
    time.sleep(1)

    print("Flushing inlet buffers to remove stale data...")
    t_flush = local_clock()
    eeg_inlet.flush()
    hr_inlet.flush()

    print(f"Buffers flushed at LSL time: {t_flush:.6f}")
    print(f"\nRecording for {duration_sec} seconds...")
    print(f"Output files: {eeg_file}, {hr_file}")
    print("Press Ctrl+C to stop early\n")

    # Queues for thread-safe writing
    eeg_q = queue.Queue()
    hr_q = queue.Queue()
    stop_flag = threading.Event()

    def reader(inlet, q, name):
        while not stop_flag.is_set():
            sample, timestamp = inlet.pull_sample(timeout=0.1)
            if sample is not None:
                q.put((timestamp, sample))

    # Start reader threads
    t1 = threading.Thread(target=reader, args=(eeg_inlet, eeg_q, "EEG"), daemon=True)
    t2 = threading.Thread(target=reader, args=(hr_inlet, hr_q, "HR"), daemon=True)
    t_record_start = local_clock()
    t1.start()
    t2.start()

    # Write to CSV
    start_time = time.time()
    eeg_count = 0
    hr_count = 0
    eeg_timestamps = []
    hr_timestamps = []

    try:
        with open(eeg_file, 'w', newline='') as feeg, open(hr_file, 'w', newline='') as fhr:
            eeg_writer = csv.writer(feeg)
            hr_writer = csv.writer(fhr)

            # Headers
            eeg_writer.writerow(['timestamp'] + [f'ch{i}' for i in range(num_eeg_ch)])
            hr_writer.writerow(['timestamp', 'hr_bpm', 'rr_s'])

            while time.time() - start_time < duration_sec:
                # Write EEG samples
                while True:
                    try:
                        ts, sample = eeg_q.get_nowait()
                        eeg_writer.writerow([ts] + list(sample))
                        eeg_count += 1
                        eeg_timestamps.append(ts)
                    except queue.Empty:
                        break

                # Write HR samples
                while True:
                    try:
                        ts, sample = hr_q.get_nowait()
                        hr_writer.writerow([ts, sample[0], sample[1]])
                        hr_count += 1
                        hr_timestamps.append(ts)
                    except queue.Empty:
                        break

                time.sleep(0.01)  # Small delay to avoid busy-waiting

    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    finally:
        stop_flag.set()
        t_record_end = local_clock()

    # IMPROVEMENT 2: Write metadata file
    with open(meta_file, 'w') as f:
        f.write("RECORDING METADATA\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Recording timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration requested: {duration_sec} seconds\n")
        f.write(f"Duration actual: {t_record_end - t_record_start:.3f} seconds\n\n")

        f.write("LSL TIMING:\n")
        f.write(f"  Flush time:      {t_flush:.6f}\n")
        f.write(f"  Recording start: {t_record_start:.6f}\n")
        f.write(f"  Recording end:   {t_record_end:.6f}\n\n")

        f.write("STREAM INFORMATION:\n")
        f.write(f"  EEG:\n")
        f.write(f"    Name:          {eeg_info.name()}\n")
        f.write(f"    Type:          {eeg_info.type()}\n")
        f.write(f"    Channels:      {eeg_info.channel_count()}\n")
        f.write(f"    Sample rate:   {eeg_info.nominal_srate()} Hz\n")
        f.write(f"    Source ID:     {eeg_info.source_id()}\n\n")

        f.write(f"  HR:\n")
        f.write(f"    Name:          {hr_info.name()}\n")
        f.write(f"    Type:          {hr_info.type()}\n")
        f.write(f"    Channels:      {hr_info.channel_count()}\n")
        f.write(f"    Source ID:     {hr_info.source_id()}\n\n")

        f.write("SAMPLES COLLECTED:\n")
        f.write(f"  EEG: {eeg_count} samples\n")
        f.write(f"  HR:  {hr_count} samples\n\n")

        if eeg_timestamps and hr_timestamps:
            eeg_first = eeg_timestamps[0]
            eeg_last = eeg_timestamps[-1]
            hr_first = hr_timestamps[0]
            hr_last = hr_timestamps[-1]

            f.write("TIMESTAMP ANALYSIS:\n")
            f.write(f"  EEG first sample:\n")
            f.write(f"    Timestamp:     {eeg_first:.6f}\n")
            f.write(f"    Delay after flush: {(eeg_first - t_flush)*1000:.1f} ms\n\n")

            f.write(f"  HR first sample:\n")
            f.write(f"    Timestamp:     {hr_first:.6f}\n")
            f.write(f"    Delay after flush: {(hr_first - t_flush)*1000:.1f} ms\n\n")

            f.write(f"  First sample offset:\n")
            f.write(f"    |EEG - HR|:     {abs(eeg_first - hr_first)*1000:.1f} ms\n")
            f.write(f"    (This reflects BLE latency differences)\n\n")

            f.write(f"  Data span:\n")
            f.write(f"    EEG: {eeg_first:.3f} to {eeg_last:.3f} ({eeg_last - eeg_first:.3f}s)\n")
            f.write(f"    HR:  {hr_first:.3f} to {hr_last:.3f} ({hr_last - hr_first:.3f}s)\n\n")

        f.write("SYNCHRONIZATION NOTES:\n")
        f.write("  - Timestamps represent LSL local clock when samples arrived\n")
        f.write("  - NOT when physiological events occurred\n")
        f.write("  - BLE latency: ~50-100ms variable delay\n")
        f.write("  - Inter-device sync accuracy: ±50-100ms\n")
        f.write("  - See SYNCHRONIZATION_ANALYSIS.md for details\n")

    # IMPROVEMENT 3: Console summary with timing analysis
    print(f"\n✓ Recording complete!")
    print(f"  Files written:")
    print(f"    - {eeg_file} ({eeg_count} samples)")
    print(f"    - {hr_file} ({hr_count} samples)")
    print(f"    - {meta_file} (metadata)")

    if eeg_timestamps and hr_timestamps:
        print(f"\n  Timing analysis:")
        print(f"    EEG first sample: +{(eeg_timestamps[0] - t_flush)*1000:5.1f}ms after flush")
        print(f"    HR first sample:  +{(hr_timestamps[0] - t_flush)*1000:5.1f}ms after flush")
        print(f"    Offset between first samples: {abs(eeg_timestamps[0] - hr_timestamps[0])*1000:5.1f}ms")
        print(f"    (Due to BLE latency - see {meta_file} for details)")

if __name__ == "__main__":
    import sys
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    record_streams(duration_sec=duration)
