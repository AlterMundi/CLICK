#!/usr/bin/env python3
from muselsl import list_muses

print("Scanning for Muse devices… (press Ctrl+C to stop)\n")
muses = list_muses()
if not muses:
    print("No Muse found. Make sure it is powered on and in pairing mode (blinking blue).")
else:
    for i, m in enumerate(muses):
        print(f"[{i}] Name: {m['name']}")
        print(f"    MAC : {m['address']}")
        print()
