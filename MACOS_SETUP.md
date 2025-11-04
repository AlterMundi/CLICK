# macOS Setup Guide for CLICK

## Prerequisites

1. **Install Homebrew** (if not already installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. **Install Python 3.11+**:
```bash
brew install python@3.11
python3 --version  # Verify >= 3.11
```

3. **Grant Bluetooth Permissions**:
- System Settings → Privacy & Security → Bluetooth
- Enable for Terminal.app (or your IDE)

## Installation

### 1. Clone Repository
```bash
cd ~/REPOS
git clone <repo-url> CLICK
cd CLICK
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install bleak pylsl matplotlib pandas scipy numpy neurokit2
```

### 4. Verify Installation
```bash
# Test LSL
python3 -c "import pylsl; print(f'LSL version: {pylsl.version_info()}')"

# Test Bleak
python3 -c "import bleak; print('Bleak OK')"

# Test complete stack
python3 -c "import bleak, pylsl, matplotlib, pandas, scipy, numpy; print('All dependencies OK')"
```

## Device Discovery

### Find Polar H10 MAC Address
```bash
# Method 1: Use system_profiler (macOS native)
system_profiler SPBluetoothDataType | grep -A 10 "Polar"

# Method 2: Use bleak scanner (recommended - cross-platform)
python3 << 'EOF'
import asyncio
from bleak import BleakScanner

async def scan_polar():
    print("Scanning for Polar H10...")
    devices = await BleakScanner.discover(timeout=10.0)
    for d in devices:
        if d.name and 'Polar' in d.name:
            print(f"Found: {d.name} - {d.address}")

asyncio.run(scan_polar())
EOF
```

### Find Muse S Device
```bash
# Install muselsl for Muse streaming
pip install muselsl

# List available Muse devices
muselsl list

# Start streaming (in separate terminal)
muselsl stream
```

## Running the Pipeline

### Terminal 1: Start Muse EEG Streamer
```bash
cd ~/REPOS/CLICK
source venv/bin/activate
muselsl stream
```

### Terminal 2: Start Polar HR Bridge
```bash
cd ~/REPOS/CLICK
source venv/bin/activate
# Replace with your Polar MAC address
python polar_hr_to_lsl.py AA:BB:CC:DD:EE:FF
```

### Terminal 3: Live Visualization
```bash
cd ~/REPOS/CLICK
source venv/bin/activate
python live_plot_muse_polar.py
```

### Terminal 4: Record Data
```bash
cd ~/REPOS/CLICK
source venv/bin/activate
# Record for 120 seconds
python record_muse_polar_csv_improved.py 120
```

## macOS-Specific Notes

### Bluetooth MAC Address Format
- **Linux**: Uses colon-separated format (`AA:BB:CC:DD:EE:FF`)
- **macOS**: Bleak handles both formats, prefer uppercase with colons

### Permission Issues
If you see "Bluetooth permission denied":
1. Go to System Settings → Privacy & Security → Bluetooth
2. Enable for Terminal.app (or PyCharm, VSCode, etc.)
3. Restart your terminal/IDE

### Python Backend Issues
If matplotlib doesn't display plots:
```bash
# Try TkAgg backend
export MPLBACKEND=TkAgg
python live_plot_muse_polar.py

# Or install native macOS backend
pip install pyobjc-framework-Cocoa
export MPLBACKEND=MacOSX
```

### Multiple Python Versions
macOS might have system Python (2.7 or 3.9) and Homebrew Python:
```bash
# Always use Homebrew Python
which python3  # Should show /opt/homebrew/bin/python3 or /usr/local/bin/python3

# Add to ~/.zshrc or ~/.bash_profile if needed
export PATH="/opt/homebrew/bin:$PATH"
```

## Troubleshooting

### "No module named 'pylsl'" after installation
```bash
# Ensure you're in the venv
which python3  # Should show path containing 'venv'
source venv/bin/activate  # Re-activate if needed
pip list | grep pylsl  # Verify installation
```

### Polar H10 not connecting
```bash
# Ensure device is in pairing mode (blinking blue light)
# Check device battery level
# Try moving closer to reduce BLE interference
# Restart Bluetooth: System Settings → Bluetooth → Toggle off/on
```

### LSL Stream not found
```bash
# Test stream discovery
python3 << 'EOF'
from pylsl import resolve_streams
import time
print("Searching for LSL streams (10 seconds)...")
streams = resolve_streams(wait_time=10)
print(f"Found {len(streams)} stream(s):")
for s in streams:
    print(f"  - {s.name()} ({s.type()}) @ {s.nominal_srate()} Hz")
EOF
```

### High CPU usage with matplotlib
```bash
# Reduce update rate in live_plot_muse_polar.py
# Change: interval=50 (20 fps) → interval=100 (10 fps)
# Or use hardware acceleration:
defaults write org.python.python ApplePersistenceIgnoreState NO
```

## Performance Optimization

### Recommended Settings for M1/M2 Macs
```bash
# Use native ARM64 Python (not Rosetta)
arch  # Should show 'arm64', not 'i386'

# Install ARM-native dependencies
pip install --no-cache-dir --force-reinstall numpy scipy
```

### Battery Life (MacBook)
- HR-only streaming: ~2-3 hours
- ECG waveform streaming: ~20-30 minutes (high BLE usage)
- Use "Low Power Mode" OFF for best Bluetooth performance

## Next Steps

1. Test basic LSL discovery: `python -c "from pylsl import resolve_streams; print(resolve_streams())"`
2. Test Polar discovery: Run bleak scanner code above
3. Run shortened recording test: `python record_muse_polar_csv_improved.py 10`
4. Analyze recording: `python analyze_ecg_recording.py recording_*_ecg.csv`

## Differences from Linux

| Feature | Linux (Debian) | macOS |
|---------|---------------|-------|
| Bluetooth tool | `bluetoothctl` | `system_profiler SPBluetoothDataType` |
| Python location | `/usr/bin/python3` | `/opt/homebrew/bin/python3` |
| LSL library | `liblsl.so` | `liblsl.dylib` |
| Sudo for BLE | Sometimes needed | Never needed |
| Default shell | bash | zsh |
| Permissions | Group-based | Privacy Settings GUI |

## Getting Help

- Check device battery levels first
- Run LSL stream discovery to verify connectivity
- Use verbose mode: `python -u script.py` (unbuffered output)
- Check Console.app for system Bluetooth errors
