#!/usr/bin/env python3
# live_plot_muse_polar_stacked.py
import threading, queue, collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylsl import resolve_byprop, StreamInlet, local_clock

def inlet_thread(selector, out_q):
    streams = selector()
    if not streams: return
    inlet = StreamInlet(streams[0], max_buflen=30)
    while True:
        s, ts = inlet.pull_sample()
        out_q.put((ts, s))

def main():
    win_sec = 10
    eeg_q, hr_q = queue.Queue(), queue.Queue()

    # Resolve once to know EEG channel count and name
    eeg_streams = resolve_byprop('type','EEG', timeout=5)
    if not eeg_streams:
        print("EEG stream not found"); return
    eeg_info = eeg_streams[0]
    num_ch = eeg_info.channel_count()

    # Start threads
    threading.Thread(target=inlet_thread, args=(lambda: eeg_streams, eeg_q), daemon=True).start()
    threading.Thread(target=inlet_thread, args=(lambda: resolve_byprop('name','PolarH10_HR',timeout=5), hr_q), daemon=True).start()

    # Buffers
    t0 = local_clock()
    times = collections.deque(maxlen=512*win_sec)
    eeg_bufs = [collections.deque(maxlen=512*win_sec) for _ in range(num_ch)]
    hr_ts = collections.deque(maxlen=60*win_sec)
    hr_bpm = collections.deque(maxlen=60*win_sec)
    rr_s = collections.deque(maxlen=60*win_sec)

    # Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=False, gridspec_kw={'height_ratios':[3,2]})
    lines = [ax1.plot([], [], lw=1, label=f'ch{i}')[0] for i in range(num_ch)]
    ax1.set_title('EEG (stacked)'); ax1.set_ylabel('uV (offset)')
    ax1.legend(ncol=min(6, num_ch), fontsize=8, loc='upper right')

    ln_hr, = ax2.plot([], [], label='HR (bpm)', alpha=0.85)
    ln_rr, = ax2.plot([], [], 'o', ms=2, label='RR (s)', alpha=0.8)
    ax2.set_title('Polar HR/RR'); ax2.set_ylabel('bpm / s'); ax2.legend(loc='upper right')

    def update(_):
        # Drain EEG
        drained = 0
        while True:
            try:
                ts, s = eeg_q.get_nowait()
                times.append(ts - t0)
                for i in range(num_ch):
                    eeg_bufs[i].append(s[i])
                drained += 1
            except queue.Empty:
                break

        # Drain HR
        while True:
            try:
                ts, s = hr_q.get_nowait()
                hr_ts.append(ts - t0)
                hr_bpm.append(s[0])
                rr_s.append(s[1])
            except queue.Empty:
                break

        # Plot EEG stacked
        if len(times) > 2:
            tmin = max(0, times[-1] - win_sec)
            x = np.array([t for t in times if t >= tmin])
            if x.size:
                # Build matrix (channels x time) with same mask and light decimation
                mask = np.array([t >= tmin for t in times])
                dec = 4
                x = x[::dec]
                # Compute per-channel offset from robust scale
                stacks = []
                scales = []
                for i in range(num_ch):
                    y_full = np.array(list(eeg_bufs[i]))
                    y = y_full[mask][::dec] if y_full.size == len(mask) else y_full[::dec]
                    stacks.append(y)
                    scales.append(np.nanpercentile(np.abs(y), 95) or 1.0)
                offset = np.nanmedian(scales) * 4.0
                for i, ln in enumerate(lines):
                    y = stacks[i] if i < len(stacks) else np.array([])
                    if y.size:
                        ln.set_data(x, y + i*offset)
                ax1.set_xlim(x[0], x[-1])
                ax1.set_ylim(-offset, (num_ch)*offset)

        # Plot HR/RR (smooth HR lightly)
        if len(hr_ts) > 2:
            k = 5
            hr_arr = np.array(hr_bpm)
            if hr_arr.size >= k:
                hr_smooth = np.convolve(hr_arr, np.ones(k)/k, mode='same')
            else:
                hr_smooth = hr_arr
            ln_hr.set_data(hr_ts, hr_smooth)
            ln_rr.set_data(hr_ts, rr_s)
            ax2.relim(); ax2.autoscale_view()

        return lines + [ln_hr, ln_rr]

    ani = FuncAnimation(fig, update, interval=100, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
