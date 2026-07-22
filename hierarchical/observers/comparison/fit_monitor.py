"""
fit_monitor.py — cross-job health/performance dashboard for hierarchical_online fits.

Samples, every INTERVAL seconds, into one rolling log:
  * machine health: 1/5/15-min load, free memory, core count
  * per-job progress: latest heartbeat (eval N/maxiter, best negLL, s/eval, gauge)
  * completion: which subjects have written their result JSON
  * STALL DETECTION: flags any job whose log file hasn't grown since the last sample

It only READS logs and result files + OS stats; it never touches a fit. Point it
at the job logs to watch.

    python -m observers.comparison.fit_monitor \
        --logs "results/logs/jobs/02*_fit[s*]_hier_online.log" \
        --interval 60 --samples 240

Writes results/logs/monitor/hier_online_monitor_<ts>.log (and echoes to stdout).
"""
import argparse, glob, os, re, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
RESULT_DIR = HERE / "results" / "fits" / "comparison" / "hierarchical_online"

# last heartbeat line, e.g.
# [hier_online start 2/4] eval  275/400  best=  37066.5  Δbest=  1.61  rej= 16%  3.8s/eval  t= 832s  improving  rss= 640MB load= 8.5
_HB = re.compile(
    r"eval\s+(\d+)/(\d+).*?best=\s*([\d.]+).*?rej=\s*(\d+)%\s*([\d.]+)s/eval\s*"
    r"t=\s*(\d+)s\s*(\w+)")
_START = re.compile(r"start\s+(\d+)/(\d+)")
_DONE = re.compile(r"start\s+(\d+)\s+DONE\s+negLL=([\d.]+)\s+evals=(\d+)")


def _mem_free_mb():
    """Best-effort free memory (MB). Linux /proc, else vm_stat on macOS, else nan."""
    try:
        with open("/proc/meminfo") as fh:
            for ln in fh:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) / 1024
    except OSError:
        pass
    try:
        import subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = 4096
        free = re.search(r"Pages free:\s+(\d+)", out)
        spec = re.search(r"Pages speculative:\s+(\d+)", out)
        n = (int(free.group(1)) if free else 0) + (int(spec.group(1)) if spec else 0)
        return n * page / (1024**2)
    except Exception:
        return float("nan")


def _subject_of(logpath):
    m = re.search(r"\[s(\d+)\]", os.path.basename(logpath))
    return int(m.group(1)) if m else None


def snapshot(log_paths, prev_sizes):
    lines = []
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        l1, l5, l15 = os.getloadavg()
    except OSError:
        l1 = l5 = l15 = float("nan")
    ncpu = os.cpu_count() or 0
    free = _mem_free_mb()
    lines.append(f"===== {ts}  load {l1:.1f}/{l5:.1f}/{l15:.1f} ({ncpu} cores)  "
                 f"free {free:.0f}MB =====")
    for lp in sorted(log_paths):
        sid = _subject_of(lp)
        done = (RESULT_DIR / f"subject{sid}.json").exists() if sid else False
        try:
            sz = os.path.getsize(lp)
            body = Path(lp).read_text(errors="replace")
        except OSError:
            lines.append(f"  s{sid}: (no log yet)")
            continue
        # last heartbeat
        hbs = list(_HB.finditer(body))
        starts = list(_START.finditer(body))
        dones = list(_DONE.finditer(body))
        seen_before = lp in prev_sizes
        grew = (not seen_before) or (sz != prev_sizes[lp])  # 1st sample: assume ok
        prev_sizes[lp] = sz
        if done:
            lines.append(f"  s{sid}: DONE (result JSON written), {len(dones)}/"
                         f"{starts[-1].group(2) if starts else '?'} starts finished")
        elif hbs:
            m = hbs[-1]
            ev, mx, best, rej, spe, t, gauge = (m.group(1), m.group(2), float(m.group(3)),
                                                m.group(4), m.group(5), m.group(6), m.group(7))
            startpos = f"start {starts[-1].group(1)}/{starts[-1].group(2)}" if starts else "start ?"
            stall = "" if grew else "  ⚠STALL(no log growth this interval)"
            lines.append(f"  s{sid}: {startpos} eval {ev}/{mx}  best={best:.1f}  "
                         f"{spe}s/eval  {gauge}  {len(dones)} starts done{stall}")
        else:
            lines.append(f"  s{sid}: warming up (no heartbeat yet)"
                         + ("" if grew else "  ⚠no log growth"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="glob for job logs to watch")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--samples", type=int, default=240)
    a = ap.parse_args()

    mon_dir = HERE / "results" / "logs" / "monitor"
    mon_dir.mkdir(parents=True, exist_ok=True)
    outp = mon_dir / f"hier_online_monitor_{datetime.now():%Y%m%d_%H%M%S}.log"
    prev_sizes = {}
    with open(outp, "w") as out:
        hdr = f"[monitor] watching '{a.logs}' every {a.interval}s -> {outp}"
        print(hdr, flush=True); out.write(hdr + "\n"); out.flush()
        for i in range(a.samples):
            paths = glob.glob(a.logs)
            snap = snapshot(paths, prev_sizes) if paths else f"===== (no logs match {a.logs}) ====="
            print(snap, flush=True); out.write(snap + "\n"); out.flush()
            # stop early if every watched subject is done
            if paths and all((RESULT_DIR / f"subject{_subject_of(p)}.json").exists()
                             for p in paths if _subject_of(p)):
                msg = "[monitor] all watched jobs complete — exiting"
                print(msg, flush=True); out.write(msg + "\n"); break
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
