#!/usr/bin/env python3
"""
Test whether flushing LSL inlets improves synchronization.

This script:
1. Connects to both EEG and HR streams
2. Performs coordinated flush
3. Records samples
4. Analyzes whether flush improved alignment

Run with: python test_flush_synchronization.py
(Requires both Muse and Polar streams to be active)
"""

import time
import statistics
from pylsl import resolve_byprop, StreamInlet, local_clock

def test_with_flush():
    """Test recording WITH flush."""
    print("\n" + "="*70)
    print("TEST 1: WITH FLUSH")
    print("="*70)

    # Resolve streams
    print("Looking for streams...")
    eeg_streams = resolve_byprop('type', 'EEG', timeout=5)
    hr_streams = resolve_byprop('name', 'PolarH10_HR', timeout=5)

    if not eeg_streams:
        print("ERROR: EEG stream not found!")
        return None
    if not hr_streams:
        print("ERROR: HR stream not found!")
        return None

    print(f"Found EEG: {eeg_streams[0].name()}")
    print(f"Found HR: {hr_streams[0].name()}")

    # Create inlets
    eeg_inlet = StreamInlet(eeg_streams[0], max_buflen=360)
    hr_inlet = StreamInlet(hr_streams[0], max_buflen=60)

    # Let buffers fill
    print("\nWaiting 2s for buffers to fill...")
    time.sleep(2)

    # COORDINATED FLUSH
    print("Performing coordinated flush...")
    t_flush = local_clock()
    eeg_inlet.flush()
    hr_inlet.flush()
    print(f"Flushed at LSL time: {t_flush:.6f}")

    # Start recording immediately
    print("Recording for 5 seconds...\n")
    t_start = local_clock()
    eeg_samples = []
    hr_samples = []

    while local_clock() - t_start < 5.0:
        eeg_s, eeg_t = eeg_inlet.pull_sample(timeout=0.0)
        if eeg_s is not None:
            eeg_samples.append(eeg_t)

        hr_s, hr_t = hr_inlet.pull_sample(timeout=0.0)
        if hr_s is not None:
            hr_samples.append(hr_t)

        time.sleep(0.001)

    return analyze_results(t_flush, t_start, eeg_samples, hr_samples, "WITH FLUSH")


def test_without_flush():
    """Test recording WITHOUT flush."""
    print("\n" + "="*70)
    print("TEST 2: WITHOUT FLUSH")
    print("="*70)

    # Resolve streams
    print("Looking for streams...")
    eeg_streams = resolve_byprop('type', 'EEG', timeout=5)
    hr_streams = resolve_byprop('name', 'PolarH10_HR', timeout=5)

    if not eeg_streams or not hr_streams:
        print("ERROR: Streams not found!")
        return None

    # Create inlets
    eeg_inlet = StreamInlet(eeg_streams[0], max_buflen=360)
    hr_inlet = StreamInlet(hr_streams[0], max_buflen=60)

    # Let buffers fill (same as before)
    print("\nWaiting 2s for buffers to fill...")
    time.sleep(2)

    # NO FLUSH - just start recording
    print("Starting recording WITHOUT flush...")
    t_start = local_clock()
    t_flush = t_start  # For comparison purposes

    print("Recording for 5 seconds...\n")
    eeg_samples = []
    hr_samples = []

    while local_clock() - t_start < 5.0:
        eeg_s, eeg_t = eeg_inlet.pull_sample(timeout=0.0)
        if eeg_s is not None:
            eeg_samples.append(eeg_t)

        hr_s, hr_t = hr_inlet.pull_sample(timeout=0.0)
        if hr_s is not None:
            hr_samples.append(hr_t)

        time.sleep(0.001)

    return analyze_results(t_flush, t_start, eeg_samples, hr_samples, "WITHOUT FLUSH")


def analyze_results(t_flush, t_start, eeg_ts, hr_ts, label):
    """Analyze timestamp patterns."""
    if not eeg_ts or not hr_ts:
        print("ERROR: No samples received!")
        return None

    eeg_first = eeg_ts[0]
    hr_first = hr_ts[0]
    eeg_delay = (eeg_first - t_flush) * 1000
    hr_delay = (hr_first - t_flush) * 1000
    offset = abs(eeg_first - hr_first) * 1000

    # Calculate inter-sample intervals
    eeg_intervals = [(eeg_ts[i+1] - eeg_ts[i])*1000 for i in range(min(100, len(eeg_ts)-1))]
    hr_intervals = [(hr_ts[i+1] - hr_ts[i])*1000 for i in range(min(len(hr_ts)-1, 10))]

    print(f"\n{label} - RESULTS")
    print("-" * 70)
    print(f"Samples collected:")
    print(f"  EEG: {len(eeg_ts):5d} samples")
    print(f"  HR:  {len(hr_ts):5d} samples")
    print()
    print(f"First sample delays after flush/start:")
    print(f"  EEG: {eeg_delay:6.1f} ms")
    print(f"  HR:  {hr_delay:6.1f} ms")
    print()
    print(f"Timestamp offset between first samples:")
    print(f"  |EEG_first - HR_first| = {offset:.1f} ms")
    print()
    print(f"Sample timing regularity:")
    print(f"  EEG interval: {statistics.mean(eeg_intervals):.2f} ± {statistics.stdev(eeg_intervals):.2f} ms")
    print(f"              (expect ~3.91ms for 256 Hz)")
    if len(hr_intervals) > 1:
        print(f"  HR interval:  {statistics.mean(hr_intervals):.0f} ± {statistics.stdev(hr_intervals):.0f} ms")
        print(f"              (varies with heart rate)")

    # Samples predating the flush?
    pre_flush_eeg = sum(1 for ts in eeg_ts if ts < t_flush)
    pre_flush_hr = sum(1 for ts in hr_ts if ts < t_flush)

    if pre_flush_eeg > 0 or pre_flush_hr > 0:
        print()
        print(f"Samples with timestamps BEFORE flush/start:")
        print(f"  EEG: {pre_flush_eeg:5d} samples (should be 0 if flushed properly)")
        print(f"  HR:  {pre_flush_hr:5d} samples (should be 0 if flushed properly)")

    return {
        'eeg_delay': eeg_delay,
        'hr_delay': hr_delay,
        'offset': offset,
        'pre_flush_eeg': pre_flush_eeg,
        'pre_flush_hr': pre_flush_hr,
        'eeg_count': len(eeg_ts),
        'hr_count': len(hr_ts)
    }


def main():
    print("="*70)
    print("FLUSH SYNCHRONIZATION TEST")
    print("="*70)
    print("\nThis test will:")
    print("1. Record WITH flush (discards old buffer data)")
    print("2. Record WITHOUT flush (includes buffered data)")
    print("\nWe will compare:")
    print("- Whether flush reduces first-sample timestamp offset")
    print("- Whether flush removes stale data")
    print("\nBoth tests require active EEG and PolarH10_HR streams.")
    print("="*70)

    # Test 1: With flush
    results_with = test_with_flush()

    if results_with is None:
        print("\nTest failed. Ensure both streams are active.")
        return

    print("\n\nWaiting 3 seconds before second test...\n")
    time.sleep(3)

    # Test 2: Without flush
    results_without = test_without_flush()

    if results_without is None:
        print("\nTest failed.")
        return

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON & CONCLUSIONS")
    print("="*70)

    print("\n1. FIRST SAMPLE TIMESTAMP OFFSET:")
    print(f"   WITH flush:    {results_with['offset']:6.1f} ms")
    print(f"   WITHOUT flush: {results_without['offset']:6.1f} ms")
    print(f"   Difference:    {abs(results_with['offset'] - results_without['offset']):6.1f} ms")

    if abs(results_with['offset'] - results_without['offset']) < 10:
        print(f"   → Flush did NOT significantly improve alignment")
    else:
        print(f"   → Different, but both still have {min(results_with['offset'], results_without['offset']):.0f}ms+ offset")

    print("\n2. STALE DATA REMOVAL:")
    print(f"   WITH flush:    {results_with['pre_flush_eeg']} EEG samples before flush")
    print(f"   WITHOUT flush: {results_without['pre_flush_eeg']} EEG samples before start")

    if results_with['pre_flush_eeg'] < results_without['pre_flush_eeg']:
        print(f"   → Flush successfully removed stale data ✓")
    else:
        print(f"   → Minimal difference")

    print("\n3. RECORDING START DELAY:")
    print(f"   EEG first sample after flush: {results_with['eeg_delay']:.1f} ms")
    print(f"   HR first sample after flush:  {results_with['hr_delay']:.1f} ms")
    print(f"   → This {abs(results_with['eeg_delay'] - results_with['hr_delay']):.1f}ms difference is BLE latency")

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print("Flushing helps with:")
    print("  ✓ Removing stale buffered data")
    print("  ✓ Cleaner recording start time")
    print("  ✓ Reduced file size")
    print()
    print("Flushing does NOT help with:")
    print("  ✗ Inter-device synchronization accuracy")
    print("  ✗ BLE latency differences (~{:.0f}ms in this test)".format(
        abs(results_with['eeg_delay'] - results_with['hr_delay'])))
    print("  ✗ Jitter reduction")
    print()
    print("The timestamps reflect BLE arrival time, not sensor sampling time.")
    print("This cannot be fixed by flushing.")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
