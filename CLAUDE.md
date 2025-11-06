# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLICK is a biomedical data acquisition system for streaming and recording physiological signals from:
- **Muse S EEG headband** (4 channels: TP9, AF7, AF8, TP10 @ 256 Hz)
- **Polar H10 heart rate monitor** (HR in BPM + RR intervals)

Data flows through Lab Streaming Layer (LSL) for **approximate temporal alignment**. Note: There is NO hardware synchronization between devices - timing accuracy is limited by Bluetooth latency (±50-100ms). See `SYNCHRONIZATION_ANALYSIS.md` for detailed accuracy specifications.

## Architecture

### Data Flow Pipeline
1. **BLE Acquisition** → `polar_hr_to_lsl.py` reads Polar H10 via Bluetooth, publishes to LSL
2. **LSL Streaming** → Both Muse (assumed external streamer) and Polar streams available on network
3. **Visualization** → `live_plot_muse_polar.py` consumes both LSL streams, plots in real-time
4. **Recording** → `record_muse_polar_csv.py` saves synchronized data to CSV files
5. **Analysis** → `graficar_canales_separados.py` loads CSV, filters, and plots EEG channels

### Key Components

**`polar_hr_to_lsl.py`** - Polar H10 BLE-to-LSL bridge (HR + RR only)
- Connects via BLE using `bleak` library
- Uses standard Bluetooth HR Service (UUID `0x180D`)
- Extracts HR (bpm) and RR intervals (seconds)
- **Does NOT stream raw ECG waveform** - only derived metrics
- Publishes 2-channel LSL stream: `[hr_bpm, rr_seconds]`
- Usage: `python polar_hr_to_lsl.py <MAC_ADDRESS>`

**`polar_ecg_to_lsl.py`** - Polar H10 RAW ECG streaming (130 Hz waveform)
- Uses Polar proprietary PMD service (not Bluetooth standard)
- Streams actual ECG voltage samples at 130 Hz
- Battery intensive (~10x drain vs HR-only)
- Cannot run simultaneously with `polar_hr_to_lsl.py`
- Publishes 1-channel LSL stream at 130 Hz
- Usage: `python polar_ecg_to_lsl.py <MAC_ADDRESS>`
- See `ECG_RECOVERY_ANALYSIS.md` for details

**`extract_cardiac_from_eeg.py`** - Extract cardiac artifact from EEG
- Alternative to separate HR device - uses cardiac artifact already in EEG
- **Already synchronized with EEG** (same device, no BLE latency!)
- Detects R-peaks, calculates RR intervals and HRV metrics
- Works best on temporal channels (TP9, TP10)
- Usage: `python extract_cardiac_from_eeg.py <eeg_csv_file>`

**`live_plot_ecg.py`** - Real-time ECG visualization
- Displays ECG waveform with R-peak detection
- Shows RR intervals (tachogram) and heart rate
- Requires `polar_ecg_to_lsl.py` streaming
- Usage: `python live_plot_ecg.py`

**`record_ecg_csv.py`** - Record raw ECG to CSV
- Saves 130 Hz ECG waveform data
- Includes inlet flush for clean start
- Usage: `python record_ecg_csv.py [duration_seconds]`

**`analyze_ecg_recording.py`** - Post-processing ECG analysis
- R-peak detection with Pan-Tompkins algorithm
- HRV metrics (SDNN, RMSSD, pNN50)
- Comprehensive plots and CSV output
- Usage: `python analyze_ecg_recording.py <ecg_csv_file>`

**`live_plot_muse_polar.py`** - Dual-stream real-time plotter
- Two matplotlib subplots: EEG ch0 (top), HR/RR (bottom)
- Thread-per-stream architecture with queue-based data passing
- 10-second sliding window for EEG, full history for HR
- Resolves streams by properties: `type='EEG'` and `name='PolarH10_HR'`

**`record_muse_polar_csv.py`** - CSV recorder (basic version)
- Saves two files: `<prefix>_eeg.csv` (all EEG channels) and `<prefix>_hr.csv` (HR/RR)
- Default 60-second recordings, configurable via CLI
- Thread-safe writing using queues
- Usage: `python record_muse_polar_csv.py [duration_seconds]`

**`record_muse_polar_csv_improved.py`** - CSV recorder (improved version)
- All features from basic version, plus:
- Flushes inlet buffers before recording (removes stale data)
- Generates metadata file with timing analysis
- Reports first-sample delays and BLE latency estimates
- Usage: `python record_muse_polar_csv_improved.py [duration_seconds]`

**`graficar_canales_separados.py`** - EEG visualization tool
- Hardcoded to specific recording file (edit `ARCHIVO` variable)
- Applies 1-40 Hz bandpass filter (configurable)
- 4-subplot layout for Muse channels (TP9, AF7, AF8, TP10)
- Uses scipy Butterworth filter (4th order, zero-phase)

## Common Commands

### Running the Full Pipeline

```bash
# Terminal 1: Start Muse LSL streamer (assuming muselsl or similar)
# muselsl stream --name MuseS

# Terminal 2: Start Polar H10 bridge (replace MAC)
python polar_hr_to_lsl.py AA:BB:CC:DD:EE:FF

# Terminal 3: Live visualization
python live_plot_muse_polar.py

# Terminal 4: Record 120-second session
python record_muse_polar_csv.py 120
```

### Post-Processing

```bash
# Edit ARCHIVO variable in graficar_canales_separados.py first
python graficar_canales_separados.py
```

### Testing Individual Components

```bash
# Discover Polar H10 MAC address
bluetoothctl
# In bluetoothctl: scan on, wait for device, note MAC, scan off, exit

# Test LSL stream discovery (list all active streams)
python -c "from pylsl import resolve_streams; streams = resolve_streams(wait_time=2); print(f'Found {len(streams)} streams'); [print(f'  - {s.name()} ({s.type()}, {s.channel_count()}ch @ {s.nominal_srate()}Hz)') for s in streams]"

# Test Polar connection only (will stream to LSL, Ctrl+C to stop)
python polar_hr_to_lsl.py <MAC>

# Monitor LSL stream data (requires stream to be active)
python -c "from pylsl import resolve_byprop, StreamInlet; inlet = StreamInlet(resolve_byprop('name', 'PolarH10_HR', timeout=5)[0]); [print(inlet.pull_sample()) for _ in range(10)]"
```

### Testing Synchronization Accuracy

```bash
# Measure actual BLE latency and jitter for your system
python test_bluetooth_latency.py <POLAR_MAC>
# Run for 60+ seconds, shows connection interval and jitter

# Test whether flushing improves synchronization (requires both streams active)
python test_flush_synchronization.py
# Compares recording with and without inlet flush
```

### ECG Waveform Options

```bash
# Option 1: Stream RAW ECG from Polar (130 Hz waveform)
# Terminal 1: Start streaming
python polar_ecg_to_lsl.py <POLAR_MAC>

# Terminal 2: Live visualization
python live_plot_ecg.py

# Terminal 3: Record 60 seconds
python record_ecg_csv.py 60

# Analyze recording
python analyze_ecg_recording.py recording_*_ecg.csv

# Option 2: Extract cardiac artifact from existing EEG recording
python extract_cardiac_from_eeg.py recording_20251104_022518_eeg.csv
# RECOMMENDED: Already synchronized, no additional hardware
# Generates cardiac timing CSV and plots
```

See **ECG_WORKFLOW_GUIDE.md** for complete workflow examples.

## Technical Constraints

### Hardware Requirements
- Muse S EEG headband with external LSL streamer (e.g., `muselsl`)
- Polar H10 chest strap with known Bluetooth MAC address
- Bluetooth LE support on host system

### Python Dependencies
```python
# Core data streaming
bleak          # BLE communication
pylsl          # Lab Streaming Layer

# Visualization & analysis
matplotlib     # Real-time and static plotting
pandas         # CSV loading and manipulation
scipy          # Signal processing (Butterworth filter)
```

### Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install bleak pylsl matplotlib pandas scipy

# For Muse S streaming (separate installation)
pip install muselsl

# Verify LSL installation
python -c "import pylsl; print(f'LSL version: {pylsl.version_info()}')"

# Test Bluetooth access (may require adding user to bluetooth group)
bluetoothctl devices
```

### LSL Stream Specifications

**EEG Stream** (from Muse):
- Type: `'EEG'`
- Channels: 4 (TP9, AF7, AF8, TP10)
- Sample rate: 256 Hz
- Units: microvolts (µV)

**Polar Stream** (from `polar_hr_to_lsl.py`):
- Name: `'PolarH10_HR'`
- Type: `'HEART_RATE'`
- Channels: 2 (HR in bpm, RR in seconds)
- Sample rate: Irregular (driven by HR measurement updates)
- Stream ID: `polar_hr_<MAC without colons>`

## Development Notes

### IMPORTANT: HR vs ECG Data Streams

**Current default (`polar_hr_to_lsl.py`):**
- Streams HR (beats per minute) + RR intervals
- ~1 Hz update rate (per heartbeat)
- Low battery drain, standard Bluetooth service
- **Cannot reconstruct ECG waveform** - only timing information

**Alternative ECG streaming (`polar_ecg_to_lsl.py`):**
- Streams raw ECG voltage samples at 130 Hz
- Provides actual waveform (R, P, Q, S, T waves)
- High battery drain (~10x), Polar proprietary protocol
- Better for heart-evoked potentials, precise R-peak timing

**Best for EEG sync (`extract_cardiac_from_eeg.py`):**
- Extract cardiac artifact already present in EEG channels
- **Perfect synchronization** (same device, no BLE latency!)
- **±5-10ms accuracy** vs ±50-100ms from separate Polar device
- Works post-recording on existing CSV files
- See `CARDIAC_ARTIFACT_EXTRACTION_EXPLAINED.md` for complete explanation
- See `CARDIAC_EXTRACTION_QUICK_REF.md` for quick start guide
- Try `cardiac_extraction_demo.py` for interactive visualization

**Production Alternative: BleakHeart Library**
- Professional library: `pip install bleakheart`
- Supports ECG, ACC, PPG, HR with unified API
- Battery monitoring, skin contact detection
- Queue-based architecture for complex processing
- See `BLEAKHEART_VS_CUSTOM_COMPARISON.md` for detailed comparison

### Threading Model
- `live_plot_muse_polar.py` and `record_muse_polar_csv.py` use daemon threads for LSL inlet reading
- Main thread handles matplotlib animation or CSV writing
- Queues provide thread-safe data passing

### Data Synchronization (IMPORTANT LIMITATIONS)

**What LSL provides:**
- Common timestamp format (`lsl_local_clock()`)
- ~1ms clock sync between computers (if multi-machine)
- CSV files use LSL timestamps (seconds since LSL epoch)

**What LSL does NOT provide:**
- Hardware sync between Muse and Polar devices
- Timestamps applied AFTER Bluetooth transmission (polar_hr_to_lsl.py:34)
- Actual sync accuracy: ±50-100ms between devices (Bluetooth latency dominates)
- Jitter: ±15-50ms due to BLE connection intervals

**Timing chain example (Polar):**
`Heart beats` → `Polar samples` → `BLE transmission (7.5-50ms)` → `Python callback` → **`LSL timestamp assigned`**

The timestamp represents when data arrived at the computer, NOT when the physiological event occurred.

**Valid use cases:**
- Long-term correlations (>5 second timescales)
- Frequency analysis within each device
- Exploratory data analysis

**Invalid use cases:**
- Precise inter-device timing (<100ms resolution)
- Heart-evoked potentials from HR data
- Phase-locking analysis between devices

See `SYNCHRONIZATION_ANALYSIS.md` for detailed technical analysis and improvement strategies.

### Filter Design
- EEG bandpass: 1-40 Hz (removes DC drift and high-frequency noise)
- 4th-order Butterworth, zero-phase (`filtfilt`)
- Nyquist frequency: 128 Hz (FS/2 = 256/2)

### File Naming Convention
- Recordings: `recording_YYYYMMDD_HHMMSS_<stream>.csv`
- EEG files: suffix `_eeg.csv`
- HR files: suffix `_hr.csv`

### Common Code Modification Patterns

**Adding new sensors:**
1. Create BLE-to-LSL bridge script (follow `polar_hr_to_lsl.py` pattern)
2. Add thread in `live_plot_muse_polar.py` with `resolve_byprop()`
3. Add recorder thread in `record_muse_polar_csv.py`
4. Update subplot layout in visualization scripts

**Changing filter parameters:**
- Edit `LOW` and `HIGH` constants in `graficar_canales_separados.py`
- Modify butter order (currently 4) for steeper/gentler rolloff
- Set `FILTRO = False` to disable filtering entirely

**Adjusting recording duration:**
- Pass seconds as CLI arg: `python record_muse_polar_csv.py 300` (5 minutes)
- Default is 60 seconds if no argument provided

**Changing visualization window:**
- Edit `win_sec = 10` in `live_plot_muse_polar.py` for different time windows
- Buffer sizes scale automatically: `maxlen=256*win_sec` for EEG

## Troubleshooting

### "Stream not found" errors
- Ensure devices are powered on and connected
- Check LSL stream availability: `python -c "from pylsl import resolve_streams; print(resolve_streams())"`
- Verify Muse streamer is running before starting other scripts
- Confirm Polar MAC address is correct

### Bluetooth connection failures
- Run `bluetoothctl` to scan and verify device visibility
- Ensure Polar H10 is in pairing mode (no need to pair, just discoverable)
- Check BLE permissions: may require `sudo` or user in `bluetooth` group

### Plot not updating
- `live_plot_muse_polar.py` requires active X11/Wayland display
- Matplotlib backend issues: try `export MPLBACKEND=TkAgg` before running
- Check queue is draining: high CPU usage indicates stream processing

### CSV analysis file not found
- Update `ARCHIVO` variable in `graficar_canales_separados.py` (line 8)
- Ensure file is in current working directory or use absolute path
