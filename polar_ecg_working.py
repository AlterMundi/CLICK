#!/usr/bin/env python3
"""
Working Polar H10 ECG streaming based on web research.

KEY FIX: Subscribe to BOTH PMD Control AND PMD Data characteristics.
The device won't stream until it knows you're listening to control responses!
"""

import struct
import asyncio
from bleak import BleakClient
from pylsl import StreamInfo, StreamOutlet

PMD_SERVICE = "FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_DATA    = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"

sample_count = 0
outlet = None
control_responses = []

def parse_ecg_data(data: bytes):
    """Parse Polar ECG data packet."""
    if len(data) < 10 or data[0] != 0x00:
        return []

    ecg_samples = []
    for i in range(10, len(data), 3):
        if i + 2 < len(data):
            byte0, byte1, byte2 = data[i], data[i+1], data[i+2]
            value = (byte2 << 16) | (byte1 << 8) | byte0
            if value & 0x800000:
                value -= 0x1000000
            ecg_samples.append(value)

    return ecg_samples

def control_callback(sender, data):
    """
    Handle PMD Control responses (acknowledgments).
    This is CRITICAL - device won't stream without this!
    """
    control_responses.append(data)
    print(f"[Control] Response: {data.hex()}")

def data_callback(sender, data):
    """Handle incoming ECG data."""
    global sample_count, outlet

    samples = parse_ecg_data(data)

    if samples and outlet:
        for sample in samples:
            outlet.push_sample([float(sample)])
            sample_count += 1

        if sample_count % 130 == 0:
            print(f"\r  Streaming... {sample_count} samples ({sample_count/130:.1f}s)", end='', flush=True)

async def stream_ecg(mac_address):
    """Stream ECG from Polar H10 to LSL."""
    global outlet, sample_count, control_responses

    # Create LSL outlet
    info = StreamInfo(
        name='PolarH10_ECG',
        type='ECG',
        channel_count=1,
        nominal_srate=130,
        channel_format='float32',
        source_id=f'polar_ecg_{mac_address.replace(":", "")}'
    )
    outlet = StreamOutlet(info, chunk_size=32)

    print(f"Connecting to Polar H10: {mac_address}")

    async with BleakClient(mac_address, timeout=15.0) as client:
        if not client.is_connected:
            print("ERROR: Failed to connect")
            return

        print("✓ Connected!")

        # Read firmware
        try:
            firmware = await client.read_gatt_char("00002a26-0000-1000-8000-00805f9b34fb")
            print(f"  Firmware: {firmware.decode()}")
        except:
            pass

        print("\n" + "="*70)
        print("CRITICAL FIX: Subscribing to BOTH Control AND Data characteristics")
        print("="*70)

        # KEY FIX: Subscribe to PMD Control FIRST (to receive acknowledgments)
        print("\n1. Subscribing to PMD Control (for acknowledgments)...")
        try:
            await client.start_notify(PMD_CONTROL, control_callback)
            print("   ✓ Control notifications enabled")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return

        # 2. Subscribe to PMD Data (for ECG samples)
        print("2. Subscribing to PMD Data (for ECG samples)...")
        try:
            await client.start_notify(PMD_DATA, data_callback)
            print("   ✓ Data notifications enabled")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return

        print("\n" + "="*70)
        print("Sending ECG start commands...")
        print("="*70)

        # Try different command variants
        commands = [
            ("Simple (fw 5.x)", bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00])),
            ("Standard", bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00])),
            ("Alternative", bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x10, 0x00])),
            ("Minimal", bytearray([0x02, 0x00])),
        ]

        for desc, cmd in commands:
            print(f"\nTrying: {desc}")
            print(f"  Command: {cmd.hex()}")

            control_responses.clear()
            sample_count_before = sample_count

            try:
                await client.write_gatt_char(PMD_CONTROL, cmd, response=True)
                print(f"  ✓ Sent")

                # Wait for control response AND data
                await asyncio.sleep(2)

                if control_responses:
                    print(f"  ✓ Got {len(control_responses)} control response(s)")

                if sample_count > sample_count_before:
                    received = sample_count - sample_count_before
                    print(f"  ✓ SUCCESS! Received {received} ECG samples")
                    break
                else:
                    print(f"  ✗ No ECG data received")

            except Exception as e:
                print(f"  ✗ Error: {e}")

        if sample_count == 0:
            print("\n" + "="*70)
            print("ECG STREAMING FAILED")
            print("="*70)
            print("\nPossible causes:")
            print("  1. Firmware doesn't support ECG (need 3.0.35+)")
            print("  2. Device busy with another connection")
            print("  3. Low battery")
            print("  4. Electrodes not making contact")
            print("\nTroubleshooting:")
            print("  - Check firmware: python check_polar_firmware.py <MAC>")
            print("  - Disconnect from all other apps (Polar Flow, etc.)")
            print("  - Try restarting the Polar H10")
            print("  - Use HR service instead: python polar_hr_to_lsl.py <MAC>")
            return

        # Keep streaming
        print(f"\n{'='*70}")
        print("✓ ECG STREAMING ACTIVE")
        print(f"{'='*70}")
        print(f"  Stream: PolarH10_ECG")
        print(f"  Rate: 130 Hz")
        print(f"  Samples: {sample_count}")
        print("\nPress Ctrl+C to stop.\n")

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        finally:
            print("\n\nStopping ECG stream...")
            stop_cmd = bytearray([0x03, 0x00])
            try:
                await client.write_gatt_char(PMD_CONTROL, stop_cmd, response=True)
                print("✓ Stop command sent")
            except:
                pass

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python polar_ecg_working.py <MAC_ADDRESS>")
        print("Example: python polar_ecg_working.py 24:AC:AC:04:2A:5A")
        sys.exit(1)

    try:
        asyncio.run(stream_ecg(sys.argv[1]))
    except KeyboardInterrupt:
        print(f"\n\nStopped. Total samples: {sample_count}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
