# ECG Signal Recovery from Polar H10: Reality vs. Expectations

## Critical Discovery: Current System Does NOT Stream ECG

### What `polar_hr_to_lsl.py` Actually Receives

Looking at the code:
```python
HR_SERVICE = "0000180d-0000-1000-8000-00805f9b34fb"  # Standard BLE Heart Rate Service
HR_MEAS    = "00002a37-0000-1000-8000-00805f9b34fb"  # HR Measurement Characteristic

def parse_hr_measurement(data: bytes):
    # Extracts:
    # 1. HR in BPM (beats per minute) - single integer
    # 2. RR intervals - time between consecutive R-peaks (milliseconds)
```

**This is the Bluetooth SIG standard "Heart Rate Service"** which provides:
- ✓ **HR (Heart Rate)**: 60-220 BPM (1 number per heartbeat, ~1 Hz update rate)
- ✓ **RR Intervals**: Time between consecutive R-waves (e.g., [0.850, 0.845, 0.848] seconds)

**This is NOT:**
- ✗ Raw ECG waveform (voltage over time)
- ✗ Individual R-peak timestamps
- ✗ QRS complex shape
- ✗ P-wave or T-wave information

### Visualization of What You're Getting

```
Actual ECG Waveform (what's NOT transmitted):
     R
    /|\
   / | \
  /  |  \    R
 /   |   \  /|\
|    |    \/ | \
|    |       |  \
P    Q   S   P   Q...
└────┴───┴───┴───┴─────> Time

What Polar Transmits via HR Service:
HR: 72 BPM
RR: [0.833s, 0.830s, 0.835s, ...]
     └─────┴─────┴─────> Only the R-R intervals
```

## The Truth: You Cannot Recover the ECG Waveform from HR Service Data

### What You CAN Do

#### 1. Reconstruct R-Peak Timing (Approximate)

From RR intervals, you can estimate WHEN R-peaks occurred:

```python
def reconstruct_r_peak_times(hr_timestamps, rr_intervals):
    """
    Reconstruct approximate R-peak times from RR intervals.

    WARNING: This gives relative timing within each heartbeat cluster,
    NOT absolute timing synchronized with other sensors!
    """
    r_peaks = []

    for i, (ts, rr) in enumerate(zip(hr_timestamps, rr_intervals)):
        # ts = timestamp when HR notification arrived (NOT when R-peak occurred)
        # rr = time since PREVIOUS R-peak

        # Approximate R-peak time (backdate from notification)
        # This is a rough estimate with ±50-100ms error
        estimated_r_peak = ts - 0.05  # Assume 50ms BLE latency (crude!)
        r_peaks.append(estimated_r_peak)

    return r_peaks

# Example output:
# r_peaks = [100.234, 100.889, 101.543, 102.198, ...]
#            └──0.655s──┴──0.654s──┴──0.655s──┘
```

**Limitations:**
- R-peak times have ±50-100ms error (BLE latency)
- No waveform shape information
- Cannot detect arrhythmias that don't change RR intervals
- Cannot measure QT interval, ST segment, etc.

#### 2. Calculate Heart Rate Variability (HRV)

```python
import numpy as np

def calculate_hrv_metrics(rr_intervals):
    """
    Calculate time-domain HRV metrics from RR intervals.
    These are valid even without exact R-peak timing!
    """
    rr_ms = np.array(rr_intervals) * 1000  # Convert to milliseconds

    # SDNN: Standard deviation of NN intervals
    sdnn = np.std(rr_ms)

    # RMSSD: Root mean square of successive differences
    diff_rr = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))

    # pNN50: Percentage of successive intervals differing by >50ms
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100

    return {
        'sdnn': sdnn,      # Overall HRV
        'rmssd': rmssd,    # Short-term HRV (parasympathetic activity)
        'pnn50': pnn50,    # Another parasympathetic measure
        'mean_rr': np.mean(rr_ms),
        'mean_hr': 60000 / np.mean(rr_ms)
    }

# These metrics are scientifically valid!
# They don't require exact R-peak timing or waveform shape
```

#### 3. Generate "Cardiac Pace" Signal

```python
def generate_cardiac_pulse_train(r_peak_times, fs=256):
    """
    Generate a pulse train at R-peak times.
    Useful for aligning with EEG (despite timing uncertainty).
    """
    duration = r_peak_times[-1] - r_peak_times[0]
    n_samples = int(duration * fs)
    pulse_train = np.zeros(n_samples)

    for r_time in r_peak_times:
        sample_idx = int((r_time - r_peak_times[0]) * fs)
        if 0 <= sample_idx < n_samples:
            pulse_train[sample_idx] = 1.0  # Delta function at R-peak

    return pulse_train

# Can be used for:
# - Heart-evoked potentials (with caution due to timing errors)
# - Phase-locking analysis (qualitative only)
# - Visualizing cardiac timing alongside EEG
```

### What You CANNOT Do

1. ✗ **Measure QRS duration** - no waveform shape
2. ✗ **Detect ST elevation/depression** - no baseline
3. ✗ **Identify P-waves or T-waves** - only R-peaks detected
4. ✗ **Diagnose most arrhythmias** - need actual ECG
5. ✗ **Precise timing for HEP** - ±50-100ms error too large

## How to Actually Get Raw ECG from Polar H10

### The Polar PMD (Measurement Data) Service

Polar H10 CAN stream raw ECG, but through a different BLE service:

```
Standard HR Service (current implementation):
  UUID: 0x180D
  Data: HR + RR intervals
  Rate: ~1 Hz (per heartbeat)

Polar PMD Service (for raw ECG):
  UUID: FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8
  Data: Raw ECG voltage samples
  Rate: 130 Hz or 250 Hz
  Format: Signed 14-bit integers (microvolts)
```

### Implementation: Raw ECG Streaming

```python
#!/usr/bin/env python3
# polar_ecg_to_lsl.py - Stream RAW ECG from Polar H10

import struct
import asyncio
from bleak import BleakClient
from pylsl import StreamInfo, StreamOutlet

# Polar PMD Service UUIDs
PMD_SERVICE = "FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_DATA    = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"

# ECG settings
ECG_SAMPLE_RATE = 130  # Hz (Polar supports 130 Hz)

def parse_ecg_data(data: bytes):
    """
    Parse Polar ECG data packet.

    Format:
    - Byte 0: Measurement type (0x00 = ECG)
    - Byte 1-8: Timestamp (64-bit, nanoseconds)
    - Byte 9: Frame type
    - Remaining: ECG samples (signed 16-bit, microvolts)
    """
    if data[0] != 0x00:  # Not ECG data
        return []

    # Timestamp in nanoseconds (not used - we use LSL timestamp)
    # timestamp_ns = struct.unpack('<Q', data[1:9])[0]

    # ECG samples start at byte 10
    ecg_samples = []
    for i in range(10, len(data), 3):
        if i + 2 < len(data):
            # 3 bytes per sample, signed 24-bit (stored in 3 bytes)
            sample_bytes = data[i:i+3]
            # Convert to signed integer (little-endian, 24-bit)
            value = int.from_bytes(sample_bytes, byteorder='little', signed=False)
            # Convert to signed
            if value > 0x7FFFFF:
                value -= 0x1000000
            ecg_samples.append(value)

    return ecg_samples

async def stream_ecg(mac_address):
    """Stream raw ECG from Polar H10 to LSL."""

    # Create LSL outlet
    info = StreamInfo(
        name='PolarH10_ECG',
        type='ECG',
        channel_count=1,
        nominal_srate=ECG_SAMPLE_RATE,
        channel_format='float32',
        source_id=f'polar_ecg_{mac_address.replace(":", "")}'
    )
    info.desc().append_child_value("manufacturer", "Polar")
    info.desc().append_child_value("units", "microvolts")
    outlet = StreamOutlet(info)

    print(f"Connecting to Polar H10: {mac_address}")

    async with BleakClient(mac_address) as client:
        print("Connected! Requesting ECG stream...")

        # Start ECG streaming (write to PMD Control Point)
        # Command: 0x02 (start measurement), 0x00 (ECG), settings...
        start_cmd = bytearray([
            0x02,  # Start measurement
            0x00,  # ECG type
            0x00, 0x01, 0x82, 0x00,  # Settings for 130 Hz
            0x01, 0x01, 0x10, 0x00   # Additional settings
        ])

        await client.write_gatt_char(PMD_CONTROL, start_cmd)
        print("ECG streaming started!")

        # Callback for ECG data
        def ecg_callback(sender, data):
            ecg_samples = parse_ecg_data(data)
            for sample in ecg_samples:
                outlet.push_sample([float(sample)])

        # Subscribe to ECG data
        await client.start_notify(PMD_DATA, ecg_callback)

        print(f"Streaming ECG at {ECG_SAMPLE_RATE} Hz to LSL...")
        print("Press Ctrl+C to stop.\n")

        # Run until interrupted
        await asyncio.Event().wait()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python polar_ecg_to_lsl.py <MAC_ADDRESS>")
        sys.exit(1)

    try:
        asyncio.run(stream_ecg(sys.argv[1]))
    except KeyboardInterrupt:
        print("\nStopped.")
```

### Important Notes About Polar ECG Streaming

**Limitations:**
1. **Battery drain**: Raw ECG at 130 Hz consumes ~10x more power
2. **Bluetooth bandwidth**: Cannot stream both HR+RR AND raw ECG simultaneously
3. **Data rate**: ~260 bytes/sec (manageable, but higher than HR service)
4. **Polar-specific**: This is NOT a Bluetooth SIG standard service
5. **Still has BLE latency**: Timestamps still applied after transmission

**Advantages:**
1. ✓ Actual ECG waveform with R, P, T waves
2. ✓ Can detect R-peaks with <10ms precision (from waveform)
3. ✓ QRS duration, amplitude measurable
4. ✓ Can implement custom R-peak detection algorithms
5. ✓ Suitable for heart-evoked potential studies (with caveats)

## Comparison: HR Service vs. ECG Stream

| Feature | HR Service (current) | ECG Stream (PMD) |
|---------|---------------------|------------------|
| **Data type** | HR + RR intervals | Raw voltage samples |
| **Sample rate** | ~1 Hz | 130 Hz |
| **Waveform** | No | Yes (R, P, Q, S, T) |
| **R-peak timing** | ±50-100ms | ±10-20ms (from waveform) |
| **HRV analysis** | ✓ Valid | ✓ Valid (better) |
| **ECG diagnostics** | ✗ No | ✓ Limited (single lead) |
| **Battery life** | ~40 hours | ~4 hours |
| **Data rate** | ~10 bytes/sec | ~260 bytes/sec |
| **Bluetooth std** | Yes (0x180D) | No (Polar proprietary) |

## Practical Recommendations

### For Your Current Setup (HR Service Only)

**Use RR intervals for:**
1. ✓ Heart rate variability (HRV) analysis - fully valid
2. ✓ Autonomic nervous system studies (frequency domain HRV)
3. ✓ Long-term correlation with EEG (>5 second timescales)
4. ✓ Exploratory arousal/relaxation studies

**Do NOT attempt:**
1. ✗ ECG waveform reconstruction - mathematically impossible
2. ✗ Precise heart-evoked potentials - timing error too large
3. ✗ ECG morphology analysis - no waveform available
4. ✗ Sub-second cardiac-EEG phase locking - synchronization inadequate

### If You Need Actual ECG

**Option 1: Implement Polar PMD streaming**
- Use the code template above
- Requires understanding Polar protocol
- See Polar SDK documentation: https://github.com/polarofficial/polar-ble-sdk

**Option 2: Add hardware ECG to your EEG system**
- OpenBCI Cyton + ECG electrodes (truly synchronized with EEG)
- Muse + DIY ECG channel (if hardware modification possible)
- Separate research-grade ECG with hardware trigger

**Option 3: Use cardiac artifact in EEG**
- EEG often contains cardiac artifact (visible in temporal channels)
- Can be extracted with ICA or bandpass filtering
- Already synchronized (same device!)
- See example code below

## Extract Cardiac Artifact from EEG (Alternative Approach)

```python
#!/usr/bin/env python3
# extract_cardiac_artifact_from_eeg.py

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks

def extract_cardiac_artifact(eeg_data, fs=256, channel_idx=0):
    """
    Extract cardiac artifact from EEG channel.

    Best channels: Temporal (TP9, TP10) or occipital (closer to blood vessels)
    """

    # 1. Bandpass filter for cardiac frequency (0.8-2 Hz for 40-120 BPM)
    nyq = fs / 2
    low = 0.8 / nyq
    high = 2.0 / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    cardiac_signal = signal.filtfilt(b, a, eeg_data[:, channel_idx])

    # 2. Detect R-peaks in filtered signal
    # Use adaptive threshold (median absolute deviation)
    threshold = np.median(np.abs(cardiac_signal)) * 3
    peaks, properties = find_peaks(cardiac_signal,
                                   height=threshold,
                                   distance=int(fs * 0.4))  # Min 0.4s between peaks

    # 3. Calculate RR intervals
    rr_intervals = np.diff(peaks) / fs

    # 4. Heart rate
    hr = 60 / np.mean(rr_intervals)

    return {
        'cardiac_signal': cardiac_signal,
        'r_peaks': peaks,
        'rr_intervals': rr_intervals,
        'mean_hr': hr,
        'peak_times': peaks / fs
    }

# Example usage:
if __name__ == "__main__":
    # Load your EEG data
    df = pd.read_csv("recording_20251104_022518_eeg.csv")
    eeg_data = df[['ch0', 'ch1', 'ch2', 'ch3']].values

    # Extract from TP9 (channel 0 - often has strongest artifact)
    result = extract_cardiac_artifact(eeg_data, fs=256, channel_idx=0)

    print(f"Detected {len(result['r_peaks'])} heartbeats")
    print(f"Mean HR: {result['mean_hr']:.1f} BPM")
    print(f"Mean RR: {np.mean(result['rr_intervals'])*1000:.0f} ms")

    # Now you have R-peak times synchronized with EEG!
    # These are MORE accurate than Polar timestamps for EEG alignment
```

## Summary & Recommendations

### Current Situation
Your `polar_hr_to_lsl.py` streams **HR + RR intervals**, NOT raw ECG waveform.

### What You Can Do Now
1. ✓ Calculate HRV metrics (SDNN, RMSSD, pNN50)
2. ✓ Reconstruct approximate R-peak times
3. ✓ Study long-term HR-EEG correlations
4. ✓ Generate cardiac pulse trains for visualization

### What Requires Changes
To get actual ECG waveform, you need to:
1. Implement Polar PMD service streaming (see code above)
2. OR extract cardiac artifact from EEG (already synchronized!)
3. OR add hardware ECG channel to your system

### Best Immediate Solution
**Extract cardiac artifact from your EEG data** - it's already there, already synchronized, and requires no additional hardware!

Files created:
- `ECG_RECOVERY_ANALYSIS.md` (this file)
- Next: Implementation examples?
