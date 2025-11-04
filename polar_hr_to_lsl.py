# polar_hr_to_lsl.py
import struct
from bleak import BleakClient
from pylsl import StreamInfo, StreamOutlet

HR_SERVICE = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEAS    = "00002a37-0000-1000-8000-00805f9b34fb"

def parse_hr_measurement(data: bytes):
    flags = data[0]
    idx = 1
    hr_16bit = flags & 0x01
    rr_present = (flags & 0x10) != 0

    if hr_16bit:
        hr = struct.unpack_from("<H", data, idx)[0]; idx += 2
    else:
        hr = data[idx]; idx += 1

    rr_intervals = []
    if rr_present:
        while idx + 1 < len(data):
            rr = struct.unpack_from("<H", data, idx)[0]; idx += 2
            rr_intervals.append(rr / 1024.0)  # seconds

    return hr, rr_intervals

async def main(mac: str):
    info = StreamInfo("PolarH10_HR", "HEART_RATE", 2, 0, "float32", "polar_hr_"+mac.replace(":",""))
    info.desc().append_child_value("units", "hr_bpm_and_rr_s")
    outlet = StreamOutlet(info)

    async with BleakClient(mac) as client:
        await client.start_notify(HR_MEAS, lambda _, d: outlet.push_sample([parse_hr_measurement(d)[0],
                                                                          (parse_hr_measurement(d)[1][0] if parse_hr_measurement(d)[1] else 0.0)]))
        print("Streaming HR+RR to LSL. Press Ctrl+C to stop.")
        import asyncio; await asyncio.Event().wait()

if __name__ == "__main__":
    import asyncio, sys
    asyncio.run(main(sys.argv[1]))
