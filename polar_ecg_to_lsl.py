#!/usr/bin/env python3
"""
Stream RAW ECG from Polar H10 using the PMD (Measurement Data) service.

This streams actual ECG waveform at 130 Hz, not just HR + RR intervals.

Usage: python polar_ecg_to_lsl.py <MAC_ADDRESS>

WARNING: This drains battery ~10x faster than HR-only streaming.
Cannot run simultaneously with polar_hr_to_lsl.py (same device).
"""

import struct
import asyncio
from bleak import BleakClient
from pylsl import StreamInfo, StreamOutlet

# Polar PMD Service UUIDs (proprietary, not Bluetooth SIG standard)
PMD_SERVICE = "FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_DATA    = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"

ECG_SAMPLE_RATE = 130  # Hz (Polar H10 supports 130 Hz)

def parse_ecg_data(data: bytes):
    """
    Parse Polar ECG data packet.

    Format (Polar proprietary):
    - Byte 0: Measurement type (0x00 = ECG)
    - Byte 1-8: Timestamp (64-bit nanoseconds, from device internal clock)
    - Byte 9: Frame type/resolution
    - Byte 10+: ECG samples (variable encoding, typically 3 bytes per sample)

    Returns list of ECG sample values in microvolts.
    """
    if len(data) < 10 or data[0] != 0x00:
        return []

    # Polar timestamp (device clock, not used - we use LSL timestamp instead)
    # polar_timestamp_ns = struct.unpack('<Q', data[1:9])[0]

    # Resolution info
    resolution = data[9]

    # Parse samples (3 bytes each, little-endian signed 24-bit)
    ecg_samples = []
    for i in range(10, len(data), 3):
        if i + 2 < len(data):
            # Extract 3 bytes
            byte0 = data[i]
            byte1 = data[i + 1]
            byte2 = data[i + 2]

            # Combine into 24-bit signed integer
            value = (byte2 << 16) | (byte1 << 8) | byte0

            # Convert to signed (two's complement)
            if value & 0x800000:  # Check sign bit
                value -= 0x1000000

            ecg_samples.append(value)

    return ecg_samples

async def stream_ecg(mac_address):
    """Stream raw ECG from Polar H10 to LSL."""

    # Create LSL outlet for ECG
    info = StreamInfo(
        name='PolarH10_ECG',
        type='ECG',
        channel_count=1,
        nominal_srate=ECG_SAMPLE_RATE,
        channel_format='float32',
        source_id=f'polar_ecg_{mac_address.replace(":", "")}'
    )

    # Add metadata
    chns = info.desc().append_child("channels")
    chn = chns.append_child("channel")
    chn.append_child_value("label", "ECG")
    chn.append_child_value("unit", "microvolts")
    chn.append_child_value("type", "ECG")

    info.desc().append_child_value("manufacturer", "Polar Electro")
    info.desc().append_child_value("model", "H10")

    outlet = StreamOutlet(info, chunk_size=32)

    print(f"Connecting to Polar H10: {mac_address}")
    print("Requesting ECG streaming mode...")

    async with BleakClient(mac_address, timeout=10.0) as client:
        if not client.is_connected:
            print("ERROR: Failed to connect to Polar H10")
            return

        print("Connected!")

        # Request ECG stream via PMD Control Point
        # Command format: [0x02, measurement_type, settings...]
        # 0x02 = start measurement
        # 0x00 = ECG type
        # 0x00, 0x01 = settings structure
        # 0x82, 0x00 = 130 Hz sample rate

        # Simple format that works with firmware 5.0.0+
        start_cmd = bytearray([
            0x02,        # Start measurement command
            0x00,        # ECG measurement type
            0x00, 0x01,  # Settings
            0x82, 0x00   # Sample rate: 130 Hz (0x82 = 130)
        ])

        try:
            await client.write_gatt_char(PMD_CONTROL, start_cmd, response=True)
            print("ECG streaming request sent!")
        except Exception as e:
            print(f"ERROR: Failed to start ECG stream: {e}")
            print("Make sure the device is not already streaming to another app.")
            return

        sample_count = 0
        packet_count = 0

        # Callback for ECG data notifications
        def ecg_callback(sender, data):
            nonlocal sample_count, packet_count

            ecg_samples = parse_ecg_data(data)
            packet_count += 1

            # Push samples to LSL
            for sample in ecg_samples:
                outlet.push_sample([float(sample)])
                sample_count += 1

            # Progress indicator
            if packet_count % 50 == 0:
                print(f"\rStreaming... {sample_count} samples ({sample_count/ECG_SAMPLE_RATE:.1f}s)",
                      end='', flush=True)

        # Subscribe to ECG data notifications
        await client.start_notify(PMD_DATA, ecg_callback)

        print(f"\n✓ Streaming RAW ECG at {ECG_SAMPLE_RATE} Hz to LSL")
        print("  Stream name: 'PolarH10_ECG'")
        print("  Type: 'ECG'")
        print("  Units: microvolts")
        print("\nPress Ctrl+C to stop.\n")

        # Run until interrupted
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        finally:
            # Stop ECG streaming
            stop_cmd = bytearray([0x03, 0x00])  # 0x03 = stop measurement
            try:
                await client.write_gatt_char(PMD_CONTROL, stop_cmd, response=True)
            except:
                pass

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python polar_ecg_to_lsl.py <MAC_ADDRESS>")
        print("Example: python polar_ecg_to_lsl.py A0:9E:1A:XX:XX:XX")
        print()
        print("This streams RAW ECG waveform (130 Hz), not just HR+RR intervals.")
        print("Cannot run simultaneously with polar_hr_to_lsl.py")
        sys.exit(1)

    try:
        asyncio.run(stream_ecg(sys.argv[1]))
    except KeyboardInterrupt:
        print("\n\nStopped ECG streaming.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
