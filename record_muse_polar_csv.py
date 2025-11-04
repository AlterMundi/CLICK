#!/usr/bin/env python3
# record_muse_polar_csv.py - Records both streams to CSV files
import csv
import time
import threading
import queue
from datetime import datetime
from pylsl import resolve_byprop, StreamInlet

def record_streams(duration_sec=60, prefix=None):
    """Record both EEG and Polar HR streams to separate CSV files."""
    if prefix is None:
        prefix = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    eeg_file = f"{prefix}_eeg.csv"
    hr_file = f"{prefix}_hr.csv"
    
    print(f"Looking for streams...")
    eeg_streams = resolve_byprop('type', 'EEG', timeout=5)
    hr_streams = resolve_byprop('name', 'PolarH10_HR', timeout=5)
    
    if not eeg_streams:
        print("ERROR: EEG stream not found!")
        return
    if not hr_streams:
        print("ERROR: PolarH10_HR stream not found!")
        return
    
    print(f"Found EEG stream: {eeg_streams[0].name()} ({eeg_streams[0].channel_count()} channels)")
    print(f"Found HR stream: {hr_streams[0].name()} ({hr_streams[0].channel_count()} channels)")
    
    # Create inlets
    eeg_inlet = StreamInlet(eeg_streams[0], max_buflen=360)
    hr_inlet = StreamInlet(hr_streams[0], max_buflen=60)
    
    num_eeg_ch = eeg_streams[0].channel_count()
    
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
    t1.start()
    t2.start()
    
    # Write to CSV
    start_time = time.time()
    eeg_count = 0
    hr_count = 0
    
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
                    except queue.Empty:
                        break
                
                # Write HR samples
                while True:
                    try:
                        ts, sample = hr_q.get_nowait()
                        hr_writer.writerow([ts, sample[0], sample[1]])
                        hr_count += 1
                    except queue.Empty:
                        break
                
                time.sleep(0.01)  # Small delay to avoid busy-waiting
                
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    finally:
        stop_flag.set()
    
    print(f"\n✓ Recording complete!")
    print(f"  - EEG: {eeg_count} samples → {eeg_file}")
    print(f"  - HR:  {hr_count} samples → {hr_file}")

if __name__ == "__main__":
    import sys
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    record_streams(duration_sec=duration)

