# Time Synchronization Analysis: Consumer BLE Devices via LSL

## Executive Summary

**The claim of "real-time synchronization" in this system is misleading.** While LSL provides timestamp alignment at the software level, there is NO hardware synchronization between the Muse S and Polar H10 devices. The actual synchronization accuracy is limited by Bluetooth latency (±20-100ms) and lacks any ground truth timing reference.

## Critical Issues Identified

### 1. Timestamping Occurs AFTER Bluetooth Transmission

**polar_hr_to_lsl.py:34**
```python
await client.start_notify(HR_MEAS, lambda _, d: outlet.push_sample([...]))
```

**Problem:** `push_sample()` is called without an explicit timestamp, so LSL automatically assigns `lsl_local_clock()` at the moment of the callback execution. This timestamp represents:
- ✓ When the computer received the BLE notification
- ✗ **NOT** when the heart beat occurred
- ✗ **NOT** when the Polar device sampled the HR

**Timing Chain for Polar H10:**
1. Heart beats at time `T_real`
2. Polar H10 detects R-wave and processes ECG → +5-20ms
3. Polar firmware packages HR measurement → +0-10ms
4. BLE transmission queued based on connection interval → +7.5-50ms
5. Bluetooth radio transmission → +1-5ms
6. OS Bluetooth stack processes packet → +1-10ms
7. Python callback invoked, `push_sample()` called → +0.1-5ms
8. **Timestamp assigned** = `T_real` + **~15-100ms** (variable)

**For Muse S (via muselsl or similar):**
Similar chain through:
- EEG sample acquired → BLE buffer → BLE transmission → USB/BLE adapter → OS → Python → LSL timestamp
- Additional buffering in Muse firmware (typically 12-25 samples before transmission)
- Total latency: ~20-150ms variable

### 2. What LSL Actually Synchronizes

**LSL Clock Synchronization:**
- Synchronizes `lsl_local_clock()` across different **computers** on a network
- Uses Cristian's algorithm with round-trip time measurement
- Achieves ~0.5-2ms accuracy **between computers**
- **Does NOT** affect sensor-to-timestamp latency

**In this single-computer setup:**
- Both streams use the same `lsl_local_clock()` → no drift between streams
- But timestamps are applied at **different, unknown latencies** from actual physiological events
- Clock synchronization is irrelevant when latency dominates

### 3. Bluetooth LE Connection Interval Effects

**BLE Connection Parameters (typical for consumer devices):**
```
Connection Interval: 7.5ms - 50ms (negotiated, often 30ms)
Slave Latency: 0-4 intervals
Supervision Timeout: 2-6 seconds
```

**Implications:**
- Notifications can only be sent at connection interval boundaries
- A heart beat occurring 1ms after a connection event must wait ~29ms for the next transmission opportunity
- **Jitter:** Up to one full connection interval (7.5-50ms)
- Different devices may have different connection intervals (negotiated independently)

### 4. No Ground Truth Timing

**Missing synchronization signals:**
- ❌ No shared clock between devices
- ❌ No hardware sync pulse (e.g., TTL trigger)
- ❌ No GPS/NTP timestamp at sensor level
- ❌ No time-of-sampling metadata from devices

**What exists:**
- ✓ LSL timestamp = when data arrived at computer
- ✓ Relative timing within each device (e.g., Muse's 256 Hz is regular)
- ✓ RR intervals from Polar (relative timing, but no absolute reference)

## Quantitative Synchronization Accuracy

### Best Case Scenario (Ideal Conditions)

| Component | Latency | Jitter | Notes |
|-----------|---------|--------|-------|
| **Polar H10** |
| Sensor to BLE TX | 10-30ms | ±5ms | Depends on HR (faster at high HR) |
| BLE transmission | 7.5-30ms | ±7.5ms | Connection interval dependent |
| OS + Python | 2-10ms | ±3ms | System load dependent |
| **Total Polar** | **20-70ms** | **±15ms** | |
| **Muse S** |
| Sensor to BLE TX | 47-98ms | ±47ms | Buffering (12-25 samples @ 256Hz) |
| BLE transmission | 7.5-30ms | ±7.5ms | Connection interval dependent |
| OS + Python | 2-10ms | ±3ms | System load dependent |
| **Total Muse** | **57-138ms** | **±57ms** | |
| **Inter-device sync** | **±37-68ms** | **±74ms** | Difference between devices |

### Worst Case Scenario (Real-World)

- Bluetooth congestion (multiple devices, WiFi interference): +50-200ms
- CPU load (GC pauses, thread scheduling): +10-50ms
- BLE connection interval negotiation failures: +100-500ms
- **Worst case desynchronization: ±200-500ms**

## What This System CAN and CANNOT Do

### ✓ Valid Use Cases

1. **Long-term correlation studies** (>5 second time scales)
   - Example: Does HR increase correlate with EEG arousal over minutes?
   - Timing errors are negligible compared to phenomena duration

2. **Event-related averaging** with external markers
   - Add third LSL stream with precise event markers (e.g., stimulus computer)
   - Average EEG/HR responses relative to marker (not to each other)

3. **Frequency domain analysis** (within-device)
   - Each device maintains regular sampling internally
   - FFT/spectral analysis valid within each stream

4. **Approximate temporal alignment**
   - "HR increased roughly when EEG showed activity"
   - Not suitable for claims like "HR peaked 50ms before EEG spike"

### ✗ Invalid Use Cases

1. **Precise inter-device timing** (<100ms resolution)
   - Example: "Cardiac R-wave occurred 23ms before EEG spike"
   - **Impossible** - timing uncertainty exceeds claimed precision

2. **Heart-rate evoked potentials** (HEP) without ECG
   - Requires knowing exact R-wave timing (~1ms precision)
   - Polar HR timestamps have ~±50ms uncertainty

3. **Phase-locking analysis** between devices
   - Example: EEG phase-locked to cardiac cycle
   - Would require ECG channel synchronized with EEG, not separate HR device

4. **Causality claims** with tight temporal coupling
   - Cannot determine which signal preceded the other at millisecond scales

## Potential Improvements

### Level 1: Software Optimization (Minor improvement)

**Estimate and correct for latency:**
```python
# In polar_hr_to_lsl.py
import time

ESTIMATED_LATENCY = 0.045  # 45ms average delay

async def main(mac: str):
    outlet = StreamOutlet(info)

    async with BleakClient(mac) as client:
        def callback(_, data):
            # Backdate timestamp by estimated latency
            ts = lsl.local_clock() - ESTIMATED_LATENCY
            outlet.push_sample([parse_hr_measurement(data)[0], ...], ts)

        await client.start_notify(HR_MEAS, callback)
```

**Limitations:**
- Still just an estimate (actual latency varies ±15-50ms)
- Doesn't fix jitter
- Requires calibration per device/system

**Improvement: ±30-40ms (from ±50-70ms)**

### Level 2: Post-Hoc Alignment (Moderate improvement)

**Use physiological markers:**
```python
# After recording, align using cardiac artifact in EEG
import numpy as np
from scipy.signal import correlate

def align_streams(eeg_data, hr_rr_intervals):
    # Detect cardiac artifact in EEG (if visible)
    # Cross-correlate with RR intervals
    # Find optimal time shift
    lag = find_optimal_lag(eeg_data, hr_rr_intervals)
    return lag

# Apply correction to timestamps
hr_timestamps_corrected = hr_timestamps + optimal_lag
```

**Requirements:**
- Strong cardiac artifact in EEG (not always present)
- Good SNR in both signals
- Assumes constant offset (doesn't fix jitter)

**Improvement: ±10-20ms (after alignment)**

### Level 3: Hardware Synchronization (Significant improvement)

**Add synchronized ECG channel:**
- Use research-grade device with hardware sync
- Example: BioSemi (EEG + ECG + trigger inputs)
- Or: OpenBCI with ECG module + hardware trigger

**Trigger-based synchronization:**
```
Computer sends TTL pulse →
  - Recorded in EEG trigger channel
  - Also sent to Polar (if it had trigger input - it doesn't)
  - Software logs LSL timestamp of trigger

Post-hoc: Align streams using trigger timestamps
```

**Polar H10 limitation:** No trigger input support

**Better alternative:**
- Replace Polar with ECG channel on same device as EEG
- Example: Muse S + separate ECG module on same LSL outlet
- Hardware-synchronized at sampling level (<1ms accuracy)

**Improvement: ±1-5ms (hardware-limited)**

### Level 4: Research-Grade Solution (Best practice)

**Unified acquisition system:**
- Single device with synchronized ADCs
- Example: BioSemi ActiveTwo, ANT Neuro eego, g.tec
- All channels sampled on same clock
- Hardware trigger inputs for external events

**LSL used only for:**
- Streaming from single unified source
- Timestamp applied once at source

**Accuracy: <1ms between channels**

## Recommendations

### For Current System (Quick Fixes)

1. **Update documentation** to accurately describe limitations
   - Change "real-time synchronization" → "approximate temporal alignment"
   - Add accuracy specifications: ±50-100ms inter-device
   - List valid vs invalid analysis types

2. **Add latency estimation** to BLE bridges
   - Implement timestamp backdating with estimated latency
   - Log connection parameters for post-hoc analysis

3. **Record metadata**
   - BLE connection interval
   - System timestamps (not just LSL)
   - Device battery levels (affects latency)

4. **Implement post-hoc alignment**
   - If cardiac artifact visible in EEG, use it for alignment
   - Cross-correlation between streams
   - Apply corrections before analysis

### For Research-Grade Accuracy

1. **Replace Polar with ECG electrode** on EEG system
   - Add ECG channel to Muse (if possible) or switch to OpenBCI
   - True hardware synchronization with EEG

2. **Add external event markers**
   - LSL stream from stimulus computer
   - Use as ground truth timing reference
   - Analyze responses relative to markers, not between bio signals

3. **Consider research-grade system**
   - If millisecond timing is critical
   - If publication requires defendable synchronization claims
   - Cost: $5k-50k depending on channel count

## Conclusion

**The current system achieves:**
- ✓ Common timestamp format (LSL clock)
- ✓ Approximate temporal alignment (±50-100ms)
- ✓ Suitable for exploratory correlation analysis
- ✗ NOT true hardware synchronization
- ✗ NOT suitable for precise timing claims

**For your specific research questions, evaluate:**
- What temporal resolution do your hypotheses require?
- Can your phenomena be studied at 100ms resolution?
- Do you need to make causal claims about timing?

**If answers suggest tighter synchronization needed:**
- Implement post-hoc alignment (Level 2)
- Consider hardware upgrades (Level 3-4)
- Collaborate with lab that has synchronized acquisition

**Current system is appropriate for:**
- Proof-of-concept studies
- Exploration of long-timescale phenomena
- Feature extraction for ML (timing-agnostic)
- Demonstrations and teaching

**Not appropriate for:**
- Publication claims requiring precise timing
- Heart-evoked potential studies
- Phase-coupling analysis between devices
- Any sub-100ms temporal analysis
