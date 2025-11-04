#!/usr/bin/env python3
"""
Cross-platform device discovery for CLICK system.
Works on Linux, macOS, and Windows.

Usage:
    python discover_devices.py              # Scan for both BLE and LSL
    python discover_devices.py --ble-only   # BLE devices only
    python discover_devices.py --lsl-only   # LSL streams only
"""

import asyncio
import argparse
import sys
from typing import List, Optional

try:
    from bleak import BleakScanner
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False
    print("Warning: bleak not installed. BLE scanning disabled.", file=sys.stderr)

try:
    from pylsl import resolve_streams, StreamInfo
    PYLSL_AVAILABLE = True
except ImportError:
    PYLSL_AVAILABLE = False
    print("Warning: pylsl not installed. LSL scanning disabled.", file=sys.stderr)


async def scan_ble_devices(timeout: float = 10.0) -> List[tuple]:
    """
    Scan for Bluetooth Low Energy devices.

    Args:
        timeout: Scan duration in seconds

    Returns:
        List of (address, name, rssi) tuples
    """
    if not BLEAK_AVAILABLE:
        print("ERROR: bleak not installed. Run: pip install bleak")
        return []

    print(f"Scanning for BLE devices ({timeout}s)...")
    devices = await BleakScanner.discover(timeout=timeout)

    results = []
    for device in devices:
        results.append((device.address, device.name or "Unknown",
                       device.rssi if hasattr(device, 'rssi') else None))

    return results


def scan_lsl_streams(timeout: float = 5.0) -> List[StreamInfo]:
    """
    Scan for active LSL streams on the network.

    Args:
        timeout: Discovery timeout in seconds

    Returns:
        List of StreamInfo objects
    """
    if not PYLSL_AVAILABLE:
        print("ERROR: pylsl not installed. Run: pip install pylsl")
        return []

    print(f"Scanning for LSL streams ({timeout}s)...")
    streams = resolve_streams(wait_time=timeout)
    return streams


def print_ble_results(devices: List[tuple], filter_keyword: Optional[str] = None):
    """Print BLE scan results in formatted table."""
    if not devices:
        print("\nNo BLE devices found.")
        return

    print(f"\nFound {len(devices)} BLE device(s):")
    print("-" * 80)
    print(f"{'ADDRESS':<20} {'NAME':<30} {'RSSI':<10}")
    print("-" * 80)

    for address, name, rssi in sorted(devices, key=lambda x: x[2] or -999, reverse=True):
        if filter_keyword and filter_keyword.lower() not in name.lower():
            continue
        rssi_str = f"{rssi} dBm" if rssi else "N/A"
        print(f"{address:<20} {name:<30} {rssi_str:<10}")

    # Highlight relevant devices
    polar_devices = [d for d in devices if d[1] and 'polar' in d[1].lower()]
    if polar_devices:
        print("\n** Polar H10 devices detected **")
        for addr, name, _ in polar_devices:
            print(f"   Use this MAC: {addr}")


def print_lsl_results(streams: List[StreamInfo]):
    """Print LSL stream results in formatted table."""
    if not streams:
        print("\nNo LSL streams found.")
        print("Hint: Ensure devices are streaming before running this script.")
        return

    print(f"\nFound {len(streams)} LSL stream(s):")
    print("-" * 100)
    print(f"{'NAME':<25} {'TYPE':<15} {'CHANNELS':<10} {'RATE (Hz)':<12} {'SOURCE_ID':<30}")
    print("-" * 100)

    for stream in streams:
        name = stream.name()
        stype = stream.type()
        channels = stream.channel_count()
        rate = stream.nominal_srate()
        source_id = stream.source_id()

        print(f"{name:<25} {stype:<15} {channels:<10} {rate:<12.1f} {source_id:<30}")

    # Categorize streams
    eeg_streams = [s for s in streams if s.type().lower() == 'eeg']
    hr_streams = [s for s in streams if 'heart' in s.type().lower() or 'hr' in s.name().lower()]
    ecg_streams = [s for s in streams if 'ecg' in s.type().lower() or 'ecg' in s.name().lower()]

    if eeg_streams:
        print(f"\n** EEG streams: {len(eeg_streams)} found (Muse S expected)")
    if hr_streams:
        print(f"** Heart Rate streams: {len(hr_streams)} found")
    if ecg_streams:
        print(f"** ECG streams: {len(ecg_streams)} found")


async def main():
    parser = argparse.ArgumentParser(
        description="Discover BLE devices and LSL streams for CLICK system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python discover_devices.py                    # Scan everything
    python discover_devices.py --ble-only         # BLE devices only
    python discover_devices.py --lsl-only         # LSL streams only
    python discover_devices.py --filter Polar    # Show only Polar devices
    python discover_devices.py --ble-timeout 15   # Longer BLE scan
        """
    )

    parser.add_argument('--ble-only', action='store_true',
                       help='Scan BLE devices only')
    parser.add_argument('--lsl-only', action='store_true',
                       help='Scan LSL streams only')
    parser.add_argument('--ble-timeout', type=float, default=10.0,
                       help='BLE scan timeout in seconds (default: 10)')
    parser.add_argument('--lsl-timeout', type=float, default=5.0,
                       help='LSL scan timeout in seconds (default: 5)')
    parser.add_argument('--filter', type=str, default=None,
                       help='Filter BLE devices by name keyword')

    args = parser.parse_args()

    # Determine what to scan
    scan_ble = not args.lsl_only
    scan_lsl = not args.ble_only

    print("=" * 80)
    print("CLICK Device Discovery Tool")
    print(f"Platform: {sys.platform}")
    print("=" * 80)

    # BLE scan
    if scan_ble:
        devices = await scan_ble_devices(timeout=args.ble_timeout)
        print_ble_results(devices, filter_keyword=args.filter)

    # LSL scan
    if scan_lsl:
        if scan_ble:
            print("\n" + "=" * 80 + "\n")
        streams = scan_lsl_streams(timeout=args.lsl_timeout)
        print_lsl_results(streams)

    print("\n" + "=" * 80)
    print("Scan complete.")
    print("=" * 80)

    # System-specific hints
    if sys.platform == 'darwin':  # macOS
        print("\nmacOS Tips:")
        print("  - Ensure Bluetooth permissions granted in System Settings")
        print("  - Use system_profiler for native BLE info:")
        print("    system_profiler SPBluetoothDataType")
    elif sys.platform == 'linux':
        print("\nLinux Tips:")
        print("  - Check user in bluetooth group: groups $USER")
        print("  - Use bluetoothctl for pairing: bluetoothctl scan on")
    elif sys.platform == 'win32':
        print("\nWindows Tips:")
        print("  - Ensure Bluetooth is enabled in Settings")
        print("  - Check device manager for Bluetooth adapter status")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nScan interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
