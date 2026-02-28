import multiprocessing
import math
import os


def cpu_worker():
    sin = math.sin
    cos = math.cos
    tan = math.tan
    exp = math.exp
    log = math.log
    sqrt = math.sqrt
    isfinite = math.isfinite

    x = 0.123456789
    y = 0.987654321
    z = 1.111111111

    # 16K bit mask
    mask = (1 << 16384) - 1
    a = 0x123456789ABCDEF123456789ABCDEF123456789ABCDEF
    b = 0xFEDCBA987654321FEDCBA987654321FEDCBA987654321

    while True:
        # ===== Heavy transcendental load =====
        for _ in range(300):
            x = sin(x) * cos(y) + sqrt(abs(z)) * exp((x % 5))
            y = log(abs(x) + 1.0000001) * tan(z)

            # ---- CRITICAL FIX ----
            # Compress magnitude BEFORE exponentiation
            cx = math.tanh(x) * 10
            cy = math.tanh(y) * 10

            z = (abs(cx) ** 2.7) + (abs(cy) ** 2.3)

        # Safety net (rarely triggered now)
        if not isfinite(x): x = 0.1
        if not isfinite(y): y = 0.1
        if not isfinite(z): z = 0.1

        # ===== Massive big-int stress =====
        for _ in range(200):
            a = (a * b + (a << 7) ^ (b >> 3)) & mask
            b = (b * a + (b << 5) ^ (a >> 2)) & mask

        x += (a & 0xFFFFFFFF) / 1e6


if __name__ == "__main__":
    cpu_quota = os.cpu_count()

    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0:
            cpu_quota = max(1, int(quota / period))
    except FileNotFoundError:
        pass

    process_count = cpu_quota * 2  # oversubscribe

    procs = []
    for _ in range(process_count):
        p = multiprocessing.Process(target=cpu_worker)
        p.start()
        procs.append(p)

    for p in procs:
        p.join()