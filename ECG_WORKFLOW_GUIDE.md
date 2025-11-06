# Raw ECG Streaming - Complete Workflow Guide

## Quick Start (3 Steps)

### Step 1: Start ECG Stream
```bash
# Terminal 1: Stream raw ECG from Polar H10
python polar_ecg_to_lsl.py AA:BB:CC:DD:EE:FF

# You should see:
# Connected!
# ✓ Streaming RAW ECG at 130 Hz to LSL
# Press Ctrl+C to stop.
```

### Step 2: Visualize OR Record (choose one or both)

**Option A: Live Visualization**
```bash
# Terminal 2: Real-time ECG plotter
python live_plot_ecg.py

# Shows 3 plots:
# - ECG waveform with R-peak detection
# - RR intervals (tachogram)
# - Heart rate over time
```

**Option B: Record to CSV**
```bash
# Terminal 2: Record 60 seconds
python record_ecg_csv.py 60

# Output: recording_YYYYMMDD_HHMMSS_ecg.csv
```

### Step 3: Analyze Recording (if you recorded)
```bash
# Analyze the CSV file
python analyze_ecg_recording.py recording_20251104_123456_ecg.csv

# Outputs:
# - R-peak detection results
# - HRV metrics (SDNN, RMSSD, pNN50)
# - Multiple analysis plots
# - recording_*_analysis.csv with R-peak times
```

---

## Detailed Workflow Examples

### Example 1: Quick Test (Just Visualize)

```bash
# Terminal 1
python polar_ecg_to_lsl.py A0:9E:1A:XX:XX:XX

# Terminal 2
python live_plot_ecg.py
# Watch the live ECG, verify good signal quality
# Close window when done
```

### Example 2: Record for HRV Analysis

```bash
# Terminal 1: Start streaming
python polar_ecg_to_lsl.py A0:9E:1A:XX:XX:XX

# Terminal 2: Record 5 minutes
python record_ecg_csv.py 300

# After recording completes:
python analyze_ecg_recording.py recording_*_ecg.csv

# Results:
# - HRV metrics printed to console
# - recording_*_analysis.csv with R-peaks
# - Comprehensive plots displayed
```

### Example 3: Synchronized EEG + ECG Recording

```bash
# Terminal 1: Stream Muse EEG
muselsl stream

# Terminal 2: Stream Polar ECG
python polar_ecg_to_lsl.py A0:9E:1A:XX:XX:XX

# Terminal 3: Record both streams
python record_muse_ecg_csv.py 120  # (need to create this script)

# Or record separately:
# Terminal 3a:
python record_muse_polar_csv_improved.py 120
# Terminal 3b:
python record_ecg_csv.py 120
```

### Example 4: Real-time Monitoring While Recording

```bash
# Terminal 1: Stream ECG
python polar_ecg_to_lsl.py A0:9E:1A:XX:XX:XX

# Terminal 2: Live plot (monitor quality)
python live_plot_ecg.py

# Terminal 3: Record simultaneously
python record_ecg_csv.py 300

# All three can run simultaneously!
```

---

## Files Created

### Scripts
1. **polar_ecg_to_lsl.py** - BLE-to-LSL bridge for raw ECG
2. **live_plot_ecg.py** - Real-time ECG visualization with R-peak detection
3. **record_ecg_csv.py** - Record ECG stream to CSV
4. **analyze_ecg_recording.py** - Post-processing analysis with HRV metrics

### Output Files (after recording)
- `recording_YYYYMMDD_HHMMSS_ecg.csv` - Raw ECG data (timestamp, ecg_uv)
- `recording_YYYYMMDD_HHMMSS_analysis.csv` - R-peaks and RR intervals

---

## Troubleshooting

### "ECG stream not found"
```bash
# Make sure polar_ecg_to_lsl.py is running
# Check with:
python -c "from pylsl import resolve_byprop; print(resolve_byprop('type', 'ECG', timeout=5))"
```

### "Failed to start ECG stream"
- Polar H10 may already be connected to another app (Polar Beat, Polar Flow, etc.)
- Turn off Polar H10, wait 10 seconds, turn on, try again
- Make sure it's not connected via Bluetooth settings (should be unpaired)

### Poor signal quality / No R-peaks detected
- **Wet the electrodes** - dry electrodes give terrible signal
- Ensure strap is tight around chest
- Check battery level
- Move away from WiFi routers (2.4 GHz interference)

### Battery drains quickly
- Raw ECG streaming uses ~10x more power than HR service
- Expect ~4 hours vs ~40 hours for HR-only
- Turn off device when not in use

### Plot is laggy
- Reduce window size in live_plot_ecg.py (change `win_sec = 5` to `win_sec = 3`)
- Close other applications

---

## Understanding the Data

### CSV Format (record_ecg_csv.py output)
```csv
timestamp,ecg_uv
12345.678,-12
12345.686,-10
12345.694,180  ← R-peak
12345.702,95
...
```

- **timestamp**: LSL local clock (seconds since LSL epoch)
- **ecg_uv**: ECG voltage in microvolts (µV)
- **Sample rate**: 130 Hz (one sample every ~7.7ms)

### Analysis Output (analyze_ecg_recording.py)
```csv
r_peak_sample,r_peak_time,rr_interval_s,hr_bpm
145,1.115,NaN,NaN
275,2.008,0.893,67.2
402,2.901,0.893,67.2
...
```

- **r_peak_sample**: Index in original ECG array
- **r_peak_time**: Time in seconds (relative to recording start)
- **rr_interval_s**: Time since previous R-peak
- **hr_bpm**: Instantaneous heart rate

---

## HRV Metrics Explained

### SDNN (Standard Deviation of NN intervals)
- Overall heart rate variability
- Higher = more variability = generally healthier
- Typical: 20-100 ms
- Reflects both sympathetic and parasympathetic activity

### RMSSD (Root Mean Square of Successive Differences)
- Short-term heart rate variability
- Reflects parasympathetic (vagal) activity
- Typical: 20-50 ms
- Higher = more relaxed state

### pNN50 (Percentage of NN intervals > 50ms different)
- Another parasympathetic measure
- Percentage of consecutive RR intervals differing by >50ms
- Typical: 5-40%
- Higher = more parasympathetic activity

---

## Comparing ECG Service vs HR Service

### ECG Stream (polar_ecg_to_lsl.py)
```
Advantages:
✓ Actual waveform (can see QRS, P, T waves)
✓ Precise R-peak timing (<10ms from waveform analysis)
✓ Can detect arrhythmias
✓ Better for heart-evoked potentials

Disadvantages:
✗ 10x battery drain (~4 hours)
✗ Higher Bluetooth bandwidth
✗ Still has ±20-50ms BLE latency for EEG sync
✗ More complex processing
```

### HR Service (polar_hr_to_lsl.py)
```
Advantages:
✓ Long battery life (~40 hours)
✓ Lower bandwidth
✓ Simple data (HR + RR)
✓ Valid for HRV analysis

Disadvantages:
✗ No waveform information
✗ Only RR intervals (±50-100ms timing)
✗ Cannot detect subtle arrhythmias
✗ Not suitable for HEP studies
```

### EEG Artifact Extraction (extract_cardiac_from_eeg.py)
```
Advantages:
✓ PERFECT EEG synchronization (same device!)
✓ No extra hardware
✓ No battery drain
✓ Post-processing (works on existing files)
✓ Best for EEG-cardiac studies

Disadvantages:
✗ Lower SNR than dedicated ECG
✗ May not work if artifact too weak
✗ Not true clinical ECG
```

---

## Next Steps

### For HRV Research
1. Record 5-10 minute ECG sessions
2. Analyze with analyze_ecg_recording.py
3. Extract SDNN, RMSSD, pNN50 from output
4. Compare across conditions/subjects

### For Heart-Evoked Potentials (with EEG)
**Option A: Use ECG stream (sub-optimal due to BLE latency)**
1. Record EEG + ECG simultaneously
2. Analyze ECG to get R-peak times
3. Align with EEG (expect ±20-50ms error)
4. Average EEG epochs around R-peaks

**Option B: Use EEG artifact (RECOMMENDED)**
1. Record only EEG
2. Extract cardiac from EEG: `python extract_cardiac_from_eeg.py recording_*_eeg.csv`
3. Perfect synchronization - no alignment needed!
4. Average EEG epochs around R-peaks

### For Clinical ECG Analysis
1. Record long sessions (hours) using ECG stream
2. Analyze waveform morphology
3. Detect arrhythmias, QRS duration, ST segments
4. **Note: Single-lead ECG has limitations - not a replacement for 12-lead**

---

## Need Help?

Check these files:
- **ECG_RECOVERY_ANALYSIS.md** - Technical details about ECG data
- **SYNCHRONIZATION_ANALYSIS.md** - Timing accuracy limitations
- **QUICK_REFERENCE.md** - Visual comparison of data types

Questions? Issues? Check signal quality first (wet electrodes!), then review troubleshooting section above.
