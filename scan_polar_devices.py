#!/usr/bin/env python3
"""
Scan for Polar devices on Bluetooth.
"""

import asyncio
from bleak import BleakScanner

async def scan_for_polar():
    """Scan for all Polar devices."""

    print("Scanning for Bluetooth LE devices...")
    print("This will take ~10 seconds...\n")

    devices = await BleakScanner.discover(timeout=10.0, return_adv=True)

    polar_devices = []
    all_devices = []

    for address, (device, adv_data) in devices.items():
        # Store all devices for debugging
        all_devices.append({
            'name': device.name or '<No Name>',
            'address': address,
            'rssi': adv_data.rssi if hasattr(adv_data, 'rssi') else 'N/A'
        })

        # Check if it's a Polar device
        name = device.name or ''
        if 'polar' in name.lower() or 'h10' in name.lower() or 'h9' in name.lower():
            polar_devices.append({
                'name': device.name,
                'address': address,
                'rssi': adv_data.rssi if hasattr(adv_data, 'rssi') else 'N/A',
                'details': device.details
            })

    print("="*70)
    print("POLAR DEVICES FOUND")
    print("="*70)

    if polar_devices:
        for i, dev in enumerate(polar_devices, 1):
            print(f"\n{i}. {dev['name']}")
            print(f"   MAC Address: {dev['address']}")
            print(f"   Signal (RSSI): {dev['rssi']} dBm")

        print(f"\n{'='*70}")
        print(f"Found {len(polar_devices)} Polar device(s)")
        print(f"{'='*70}")

    else:
        print("\n✗ No Polar devices found")
        print("\nTroubleshooting:")
        print("  1. Make sure Polar H10 is turned on")
        print("  2. Check battery level")
        print("  3. Move device closer to computer")
        print("  4. Unpair from other devices (phone, etc.)")

        # Show all devices for debugging
        print(f"\n{'='*70}")
        print(f"ALL BLUETOOTH DEVICES DETECTED ({len(all_devices)})")
        print(f"{'='*70}")

        # Sort by RSSI (strongest signal first)
        all_devices.sort(key=lambda x: x['rssi'] if isinstance(x['rssi'], (int, float)) else -999, reverse=True)

        for dev in all_devices[:20]:  # Show top 20
            print(f"  {dev['name'][:30]:30s} | {dev['address']:17s} | RSSI: {dev['rssi']}")

        if len(all_devices) > 20:
            print(f"  ... and {len(all_devices) - 20} more devices")

    print()

if __name__ == "__main__":
    try:
        asyncio.run(scan_for_polar())
    except KeyboardInterrupt:
        print("\n\nScan interrupted")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
