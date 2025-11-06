#!/usr/bin/env python3
"""
Measure actual Bluetooth latency and jitter for Polar H10.

This script connects to Polar H10 and measures:
1. Time between consecutive HR notifications (connection interval)
2. Latency estimation via notification timing patterns
3. Jitter (variability in notification intervals)

Usage: python test_bluetooth_latency.py <MAC_ADDRESS>
Run for 60+ seconds for meaningful statistics.
"""

import sys
import time
import struct
import asyncio
import statistics
from collections import deque
from bleak import BleakClient

HR_SERVICE = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEAS = "00002a37-0000-1000-8000-00805f9b34fb"

class LatencyTester:
    def __init__(self):
        self.timestamps = deque(maxlen=1000)
        self.intervals = deque(maxlen=1000)
        self.last_ts = None
        self.count = 0

    def callback(self, sender, data):
        now = time.time()
        self.count += 1

        if self.last_ts is not None:
            interval_ms = (now - self.last_ts) * 1000
            self.intervals.append(interval_ms)

            # Parse HR to show we're getting data
            flags = data[0]
            hr = struct.unpack_from("<H", data, 1)[0] if flags & 0x01 else data[1]

            if self.count % 10 == 0:
                if len(self.intervals) >= 10:
                    mean = statistics.mean(self.intervals)
                    stdev = statistics.stdev(self.intervals) if len(self.intervals) > 1 else 0
                    min_int = min(self.intervals)
                    max_int = max(self.intervals)

                    print(f"\rCount: {self.count:4d} | "
                          f"HR: {hr:3d} bpm | "
                          f"Interval: {interval_ms:6.1f}ms | "
                          f"Mean: {mean:6.1f}ms ± {stdev:5.1f}ms | "
                          f"Range: [{min_int:5.1f}, {max_int:5.1f}]ms",
                          end='', flush=True)

        self.last_ts = now
        self.timestamps.append(now)

    def print_summary(self):
        print("\n\n" + "="*70)
        print("LATENCY TEST RESULTS")
        print("="*70)

        if len(self.intervals) < 2:
            print("Not enough data collected!")
            return

        mean_interval = statistics.mean(self.intervals)
        stdev_interval = statistics.stdev(self.intervals)
        min_interval = min(self.intervals)
        max_interval = max(self.intervals)

        print(f"\nNotifications received: {self.count}")
        print(f"Measurement duration: {(self.timestamps[-1] - self.timestamps[0]):.1f} seconds")
        print(f"\n{'NOTIFICATION INTERVAL (approximates BLE connection interval)':-^70}")
        print(f"  Mean:        {mean_interval:6.1f} ms")
        print(f"  Std Dev:     {stdev_interval:6.1f} ms  (jitter)")
        print(f"  Min:         {min_interval:6.1f} ms")
        print(f"  Max:         {max_interval:6.1f} ms")
        print(f"  Range:       {max_interval - min_interval:6.1f} ms  (total jitter)")

        # Estimate BLE connection interval (should be multiple of 1.25ms)
        estimated_conn_interval = round(mean_interval / 1.25) * 1.25
        print(f"\n  Estimated BLE connection interval: {estimated_conn_interval:.2f} ms")

        # Histogram
        print(f"\n{'INTERVAL DISTRIBUTION':-^70}")
        buckets = {}
        for interval in self.intervals:
            bucket = int(interval / 5) * 5  # 5ms buckets
            buckets[bucket] = buckets.get(bucket, 0) + 1

        for bucket in sorted(buckets.keys()):
            bar_length = int(buckets[bucket] / max(buckets.values()) * 40)
            print(f"  {bucket:3d}-{bucket+5:3d}ms: {'█' * bar_length} ({buckets[bucket]})")

        # Analysis
        print(f"\n{'SYNCHRONIZATION ACCURACY IMPLICATIONS':-^70}")
        print(f"  Minimum possible jitter: ±{stdev_interval:.1f} ms")
        print(f"  Maximum timestamp error: ±{max_interval:.1f} ms")
        print(f"  \n  NOTE: This only measures notification timing jitter.")
        print(f"  Actual sensor-to-timestamp latency includes:")
        print(f"    - Polar internal processing: ~10-20ms")
        print(f"    - BLE transmission: ~1-5ms")
        print(f"    - OS + Python overhead: ~2-10ms")
        print(f"    - Total estimated latency: 25-100ms (not measured here)")
        print(f"\n  For inter-device synchronization with Muse:")
        print(f"    - Expect ±{stdev_interval + 20:.0f} to ±{max_interval + 50:.0f} ms uncertainty")
        print("="*70)

async def main(mac_address):
    print(f"Connecting to Polar H10: {mac_address}")
    print("Measuring Bluetooth notification timing...")
    print("Press Ctrl+C to stop and see results\n")

    tester = LatencyTester()

    try:
        async with BleakClient(mac_address) as client:
            await client.start_notify(HR_MEAS, tester.callback)
            print("Connected! Collecting data...\n")

            # Run until interrupted
            await asyncio.Event().wait()

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        tester.print_summary()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_bluetooth_latency.py <MAC_ADDRESS>")
        print("Example: python test_bluetooth_latency.py A0:9E:1A:XX:XX:XX")
        sys.exit(1)

    try:
        asyncio.run(main(sys.argv[1]))
    except KeyboardInterrupt:
        pass
