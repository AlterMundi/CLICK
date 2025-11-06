#!/usr/bin/env python3
"""
Record raw ECG stream from Polar H10 to CSV file.

Usage: python record_ecg_csv.py [duration_seconds]

Output: recording_YYYYMMDD_HHMMSS_ecg.csv
Format: timestamp, ecg_uv
"""

import csv
import time
import sys
from datetime import datetime
from pylsl import resolve_byprop, StreamInlet, local_clock

def record_ecg(duration_sec=60, prefix=None):
    """Record ECG stream to CSV file."""

    if prefix is None:
        prefix = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    ecg_file = f"{prefix}_ecg.csv"

    print(f"Looking for ECG stream...")
    ecg_streams = resolve_byprop('type', 'ECG', timeout=10)

    if not ecg_streams:
        print("ERROR: ECG stream not found!")
        print("Make sure polar_ecg_to_lsl.py is running.")
        return

    ecg_info = ecg_streams[0]
    print(f"Found ECG stream: {ecg_info.name()} @ {ecg_info.nominal_srate()} Hz")

    # Create inlet
    inlet = StreamInlet(ecg_info, max_buflen=360)

    # Flush old data
    print("Flushing inlet buffer...")
    time.sleep(1)
    t_flush = local_clock()
    inlet.flush()

    print(f"Flushed at LSL time: {t_flush:.6f}")
    print(f"\nRecording for {duration_sec} seconds...")
    print(f"Output file: {ecg_file}")
    print("Press Ctrl+C to stop early\n")

    # Record
    sample_count = 0
    t_start = local_clock()
    first_ts = None
    last_ts = None

    try:
        with open(ecg_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'ecg_uv'])

            while local_clock() - t_start < duration_sec:
                sample, timestamp = inlet.pull_sample(timeout=0.1)

                if sample is not None:
                    writer.writerow([timestamp, sample[0]])
                    sample_count += 1

                    if first_ts is None:
                        first_ts = timestamp
                    last_ts = timestamp

                    # Progress indicator
                    if sample_count % 130 == 0:  # Every second
                        elapsed = local_clock() - t_start
                        print(f"\rRecording... {sample_count} samples ({elapsed:.1f}s / {duration_sec}s)",
                              end='', flush=True)

    except KeyboardInterrupt:
        print("\n\nRecording stopped by user")

    print(f"\n\n✓ Recording complete!")
    print(f"  File: {ecg_file}")
    print(f"  Samples: {sample_count}")
    print(f"  Duration: {last_ts - first_ts:.1f} seconds" if first_ts else "")
    print(f"  Sample rate: {sample_count / (last_ts - first_ts):.1f} Hz" if first_ts and last_ts else "")

    if first_ts:
        print(f"\n  Timing:")
        print(f"    Flush time:       {t_flush:.6f}")
        print(f"    First sample:     {first_ts:.6f}")
        print(f"    Delay after flush: {(first_ts - t_flush)*1000:.1f} ms")

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    record_ecg(duration_sec=duration)
