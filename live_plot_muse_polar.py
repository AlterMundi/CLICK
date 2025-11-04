# live_plot_muse_polar.py
import threading, queue, time, collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylsl import resolve_byprop, StreamInlet

def inlet_thread(selector, out_q):
    streams = selector()
    if not streams: return
    inlet = StreamInlet(streams[0], max_buflen=30)
    while True:
        s, ts = inlet.pull_sample()
        out_q.put((ts, s))

def main():
    # Buffers
    win_sec = 10
    eeg_q, hr_q = queue.Queue(), queue.Queue()
    t0 = time.time()
    times = collections.deque(maxlen=256*win_sec)
    eeg0 = collections.deque(maxlen=256*win_sec)   # first EEG channel
    hr_ts = collections.deque(maxlen=60*win_sec)
    hr_bpm = collections.deque(maxlen=60*win_sec)
    rr_s = collections.deque(maxlen=60*win_sec)

    # Threads
    threading.Thread(target=inlet_thread, args=(lambda: resolve_byprop('type','EEG',timeout=5), eeg_q), daemon=True).start()
    threading.Thread(target=inlet_thread, args=(lambda: resolve_byprop('name','PolarH10_HR',timeout=5), hr_q), daemon=True).start()

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6), sharex=False)
    ln_eeg, = ax1.plot([], [], lw=1)
    ln_hr, = ax2.plot([], [], label='HR (bpm)')
    ln_rr, = ax2.plot([], [], label='RR (s)')
    ax1.set_title('EEG ch0'); ax1.set_ylabel('uV (raw)')
    ax2.set_title('Polar HR/RR'); ax2.set_ylabel('bpm / s'); ax2.legend(loc='upper right')

    def update(_):
        # Drain EEG queue
        drained = 0
        while True:
            try:
                ts, s = eeg_q.get_nowait()
                times.append(ts - t0)
                eeg0.append(s[0])
                drained += 1
            except queue.Empty:
                break
        # Drain HR queue
        while True:
            try:
                ts, s = hr_q.get_nowait()
                hr_ts.append(ts - t0)
                hr_bpm.append(s[0])
                rr_s.append(s[1])
            except queue.Empty:
                break

        if len(times) > 2:
            tmin = max(0, times[-1] - win_sec)
            ln_eeg.set_data([t for t in times if t >= tmin],
                            [v for t,v in zip(times, eeg0) if t >= tmin])
            ax1.set_xlim(max(0, times[-1]-win_sec), times[-1])
            if len(eeg0) > 0:
                ymin, ymax = min(eeg0), max(eeg0)
                if ymin != ymax:
                    ax1.set_ylim(ymin, ymax)

        if len(hr_ts) > 0:
            ln_hr.set_data(hr_ts, hr_bpm)
            ln_rr.set_data(hr_ts, rr_s)
            ax2.relim(); ax2.autoscale_view()

        return ln_eeg, ln_hr, ln_rr

    ani = FuncAnimation(fig, update, interval=100, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
