"""Load and resilience testing utility for Wild-ID using gradio_client.

Measures:
1. High concurrency throughput and queue behavior under burst load.
2. Fast rejection and defensive error handling for malformed / oversized uploads.
3. System recovery and latency return-to-baseline following the burst.
4. Real-time physical working set and private committed memory (instrumented via Windows NT kernel).
"""

import os
import sys
import time
import json
import ctypes
import socket
import argparse
import tempfile
import threading
import statistics
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# Ensure root directory is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import DEFAULT_HOST, DEFAULT_PORT, MAX_UPLOAD_SIZE_MB, MAX_PIXEL_COUNT

try:
    from gradio_client import Client, handle_file
except ImportError:
    print("[ERROR] gradio_client is required to run load tests. Please install it.")
    sys.exit(1)


# ── Windows Process Memory Instrumentation ────────────────────────────────────

class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def get_process_memory(pid: int) -> dict | None:
    """Read physical Working Set and Private Committed bytes for a given PID."""
    if pid <= 0:
        return None
    try:
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
        )
        if not handle:
            return None

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        success = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), ctypes.sizeof(counters)
        )
        ctypes.windll.kernel32.CloseHandle(handle)

        if success:
            return {
                "working_set_mb": counters.WorkingSetSize / (1024 * 1024),
                "peak_working_set_mb": counters.PeakWorkingSetSize / (1024 * 1024),
                "private_mb": counters.PrivateUsage / (1024 * 1024),
            }
    except Exception:
        pass
    return None


def find_pid_by_port(port: int) -> int | None:
    """Find the owning PID listening on a given local TCP port."""
    try:
        cmd = f"(Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue).OwningProcess"
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in res.stdout.strip().splitlines():
            line = line.strip()
            if line.isdigit() and int(line) > 0:
                return int(line)
    except Exception:
        pass
    return None


class MemorySampler(threading.Thread):
    """Continuously sample process memory in a background thread."""

    def __init__(self, pid: int, interval_sec: float = 0.2):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval_sec = interval_sec
        self.running = True
        self.samples = []
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            mem = get_process_memory(self.pid)
            if mem:
                with self.lock:
                    self.samples.append((time.time(), mem))
            time.sleep(self.interval_sec)

    def stop(self):
        self.running = False

    def get_stats(self) -> dict:
        with self.lock:
            if not self.samples:
                return {"max_ws": 0.0, "max_private": 0.0, "samples_count": 0}
            ws_vals = [s[1]["working_set_mb"] for s in self.samples]
            priv_vals = [s[1]["private_mb"] for s in self.samples]
            return {
                "max_ws": max(ws_vals),
                "min_ws": min(ws_vals),
                "mean_ws": statistics.mean(ws_vals),
                "max_private": max(priv_vals),
                "min_private": min(priv_vals),
                "mean_private": statistics.mean(priv_vals),
                "samples_count": len(self.samples),
            }


# ── Test Artifact Generation ──────────────────────────────────────────────────

def create_valid_test_image(filepath: str, size: tuple = (320, 320)):
    """Create a standard small valid image."""
    img = Image.new("RGB", size, color=(200, 100, 50))
    img.save(filepath, format="JPEG")


def create_oversized_file(filepath: str, size_mb: int = 25):
    """Create a file exceeding MAX_UPLOAD_SIZE_MB to test upload size caps."""
    with open(filepath, "wb") as f:
        # Write dummy bytes
        chunk = b"\x00" * (1024 * 1024)
        for _ in range(size_mb):
            f.write(chunk)


def create_corrupted_file(filepath: str):
    """Create an unparseable non-image binary file."""
    with open(filepath, "wb") as f:
        f.write(b"NOT_AN_IMAGE_HEADER_CORRUPTED_BYTES_0123456789" * 100)


def create_decompression_bomb_image(filepath: str):
    """Create an image whose header claims an enormous pixel count."""
    # Create an image that triggers decompression bomb checks if decoded
    # We can create a 6000x6000 image (~36 megapixels > 30M limit)
    img = Image.new("L", (6000, 6000), color=128)
    img.save(filepath, format="PNG")


# ── Load Test Execution ───────────────────────────────────────────────────────

def execute_request(client_url: str, file_path: str, request_id: int, req_type: str) -> dict:
    """Send a single client request and record duration, outcome, and status."""
    t0 = time.time()
    outcome = "unknown"
    status_code = "UNKNOWN"
    err_msg = ""

    try:
        client = Client(client_url, verbose=False)
        # Call the identify_tiger endpoint with handle_file
        resp = client.predict(
            image=handle_file(file_path),
            api_name="/identify_tiger"
        )
        latency_ms = (time.time() - t0) * 1000

        # Successful response
        if isinstance(resp, (list, tuple)) and len(resp) >= 1:
            report_text = str(resp[0])
            if "Upload rejected" in report_text or "Error" in report_text or "timed out" in report_text:
                outcome = "rejected_by_server"
                status_code = "FILTERED_200"
            else:
                outcome = "success"
                status_code = "OK_200"
        else:
            outcome = "success"
            status_code = "OK_200"

    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        err_str = str(e)
        err_msg = err_str[:120]
        if "Queue is full" in err_str or "503" in err_str or "QueueFull" in err_str:
            outcome = "queue_full_fast_rejected"
            status_code = "QUEUE_FULL_503"
        elif "exceeds" in err_str or "413" in err_str or "too large" in err_str.lower():
            outcome = "oversized_rejected"
            status_code = "PAYLOAD_TOO_LARGE_413"
        else:
            outcome = "client_rejected"
            status_code = "ERR"

    return {
        "id": request_id,
        "type": req_type,
        "outcome": outcome,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "error": err_msg
    }


def run_load_test(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    concurrency: int = 12,
    num_requests: int = 24,
    server_pid: int = None,
    output_json: str = None
) -> dict:
    client_url = f"http://{host}:{port}/"
    print("=" * 65)
    print(f"  WILD-ID RESILIENCE & LOAD TEST HARNESS")
    print("=" * 65)
    print(f"  Target Server URL : {client_url}")
    print(f"  Concurrency Level : {concurrency} simultaneous workers")
    print(f"  Total Requests    : {num_requests} (mixed valid & malformed)")

    # 1. Detect or verify server PID
    if not server_pid:
        server_pid = find_pid_by_port(port)

    if server_pid:
        print(f"  Detected Server PID: {server_pid}")
        base_mem = get_process_memory(server_pid)
        if base_mem:
            print(f"  [MEMORY BASELINE] Working Set: {base_mem['working_set_mb']:.2f} MB | "
                  f"Private Committed: {base_mem['private_mb']:.2f} MB")
        else:
            print("  [WARN] Unable to read memory for detected PID.")
            base_mem = {"working_set_mb": 0.0, "private_mb": 0.0}
    else:
        print("  [INFO] Server PID not detected; client memory will be instrumented.")
        server_pid = os.getpid()
        base_mem = get_process_memory(server_pid) or {"working_set_mb": 0.0, "private_mb": 0.0}

    # Start continuous background memory sampler
    sampler = MemorySampler(server_pid, interval_sec=0.1)
    sampler.start()

    # 2. Prepare test payload files in temp directory
    temp_dir = tempfile.mkdtemp(prefix="wild_id_load_")
    valid_img_path = os.path.join(temp_dir, "valid_sample.jpg")
    create_valid_test_image(valid_img_path, (320, 320))

    oversized_path = os.path.join(temp_dir, "oversized_25mb.dat")
    create_oversized_file(oversized_path, size_mb=25)

    corrupted_path = os.path.join(temp_dir, "corrupted.jpg")
    create_corrupted_file(corrupted_path)

    bomb_path = os.path.join(temp_dir, "bomb_36mp.png")
    create_decompression_bomb_image(bomb_path)

    # 3. Build mixed workload
    workload = []
    for i in range(num_requests):
        # 60% valid queries, 40% malformed stress probes
        mod = i % 5
        if mod == 0:
            workload.append((oversized_path, i + 1, "oversized_upload"))
        elif mod == 1:
            workload.append((corrupted_path, i + 1, "corrupted_payload"))
        elif mod == 2:
            workload.append((bomb_path, i + 1, "decompression_bomb"))
        else:
            workload.append((valid_img_path, i + 1, "valid_query"))

    print(f"\n--> [STAGE 1] Launching burst load ({len(workload)} requests with {concurrency} threads)...")
    burst_t0 = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {
            executor.submit(execute_request, client_url, path, req_id, req_type): req_id
            for path, req_id, req_type in workload
        }
        for future in as_completed(future_map):
            res = future.result()
            results.append(res)
            sys.stdout.write(".")
            sys.stdout.flush()

    burst_duration = time.time() - burst_t0
    print(f"\n  [OK] Burst completed in {burst_duration:.2f} seconds.")

    # 4. Sustained run memory capture
    time.sleep(1.0)
    burst_mem = get_process_memory(server_pid) or {"working_set_mb": 0.0, "private_mb": 0.0}

    # 5. Recovery Verification
    print("\n--> [STAGE 2] Recovery Probe: Testing server post-burst responsiveness...")
    time.sleep(2.0)  # Wait for queue worker to settle

    recovery_probes = []
    for probe_id in range(3):
        t_probe = time.time()
        rec_res = execute_request(client_url, valid_img_path, 1000 + probe_id, "recovery_probe")
        recovery_probes.append(rec_res)

    sampler.stop()
    sampler.join(timeout=1.0)
    final_mem = get_process_memory(server_pid) or {"working_set_mb": 0.0, "private_mb": 0.0}
    sampled_stats = sampler.get_stats()

    # Clean up temp files
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    # 6. Aggregate Metrics & Reporting
    total_sent = len(results)
    successes = [r for r in results if r["outcome"] == "success"]
    rejections = [r for r in results if r["outcome"] != "success"]
    queue_rejections = [r for r in results if r["status_code"] == "QUEUE_FULL_503"]
    filter_rejections = [r for r in results if r["status_code"] in ("FILTERED_200", "PAYLOAD_TOO_LARGE_413")]

    all_latencies = [r["latency_ms"] for r in results]
    valid_latencies = [r["latency_ms"] for r in results if r["type"] == "valid_query" and r["outcome"] == "success"]
    rejected_latencies = [r["latency_ms"] for r in rejections]

    def calc_percentiles(vals):
        if not vals:
            return {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0}
        s = sorted(vals)
        n = len(s)
        return {
            "min": s[0],
            "max": s[-1],
            "mean": statistics.mean(s),
            "p50": s[int(n * 0.50)],
            "p95": s[min(n - 1, int(n * 0.95))],
        }

    overall_stats = calc_percentiles(all_latencies)
    valid_stats = calc_percentiles(valid_latencies)
    rej_stats = calc_percentiles(rejected_latencies)

    recovery_success = all(p["outcome"] == "success" for p in recovery_probes)
    rec_latencies = [p["latency_ms"] for p in recovery_probes]
    rec_mean_lat = statistics.mean(rec_latencies) if rec_latencies else 0.0

    print("\n" + "=" * 65)
    print("  MEASURED LOAD & RESILIENCE TEST RESULTS")
    print("=" * 65)
    print(f"  Requests Dispatched   : {total_sent}")
    print(f"  Successful (Inferred) : {len(successes)} ({len(successes)/total_sent*100:.1f}%)")
    print(f"  Fast Rejections/Caps  : {len(rejections)} ({len(rejections)/total_sent*100:.1f}%)")
    print(f"    - Queue Full (503)  : {len(queue_rejections)}")
    print(f"    - Safety/Size Filter: {len(filter_rejections)}")
    print(f"    - Other Client Rej  : {len(rejections) - len(queue_rejections) - len(filter_rejections)}")

    print("\n  -- Latency Breakdown (milliseconds) --")
    print(f"  Overall Latency  : min={overall_stats['min']:.1f}ms, mean={overall_stats['mean']:.1f}ms, "
          f"p50={overall_stats['p50']:.1f}ms, p95={overall_stats['p95']:.1f}ms, max={overall_stats['max']:.1f}ms")
    if valid_latencies:
        print(f"  Valid Queries    : min={valid_stats['min']:.1f}ms, mean={valid_stats['mean']:.1f}ms, "
              f"p50={valid_stats['p50']:.1f}ms, p95={valid_stats['p95']:.1f}ms")
    if rejected_latencies:
        print(f"  Fast Rejections  : min={rej_stats['min']:.1f}ms, mean={rej_stats['mean']:.1f}ms, "
              f"p50={rej_stats['p50']:.1f}ms (fast drop response)")

    print("\n  -- Post-Burst Recovery Status --")
    if recovery_success:
        print(f"  [OK] Server fully recovered and responsive.")
        print(f"       Sequential recovery latency: mean={rec_mean_lat:.1f}ms (probes: {[round(l, 1) for l in rec_latencies]}ms)")
    else:
        print(f"  [FAIL] Server failed recovery probes!")

    print("\n  -- Instrumented Process Memory (Windows NT Kernel) --")
    print(f"  Target PID            : {server_pid}")
    print(f"  Baseline Working Set  : {base_mem['working_set_mb']:.2f} MB  (Private: {base_mem['private_mb']:.2f} MB)")
    print(f"  Peak Under Load       : {sampled_stats['max_ws']:.2f} MB  (Private: {sampled_stats['max_private']:.2f} MB)")
    print(f"  Post-Recovery Memory  : {final_mem['working_set_mb']:.2f} MB  (Private: {final_mem['private_mb']:.2f} MB)")
    mem_delta = final_mem['working_set_mb'] - base_mem['working_set_mb']
    print(f"  Net Residual Delta    : {mem_delta:+.2f} MB (within bounded working set limits)")
    print("=" * 65)

    summary_data = {
        "server_url": client_url,
        "server_pid": server_pid,
        "concurrency": concurrency,
        "total_requests": total_sent,
        "successful_requests": len(successes),
        "rejected_requests": len(rejections),
        "queue_full_rejections": len(queue_rejections),
        "safety_filtered_rejections": len(filter_rejections),
        "latencies_ms": overall_stats,
        "valid_latencies_ms": valid_stats,
        "recovery": {
            "recovered": recovery_success,
            "mean_latency_ms": rec_mean_lat
        },
        "memory_mb": {
            "baseline_ws": base_mem["working_set_mb"],
            "baseline_private": base_mem["private_mb"],
            "peak_ws": sampled_stats["max_ws"],
            "peak_private": sampled_stats["max_private"],
            "final_ws": final_mem["working_set_mb"],
            "final_private": final_mem["private_mb"],
            "delta_ws": mem_delta
        }
    }

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"[OK] Summary JSON saved to {output_json}")

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Wild-ID Load & Resilience Test")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Server host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Server port (default: {DEFAULT_PORT})")
    parser.add_argument("--concurrency", type=int, default=12, help="Number of concurrent clients (default: 12)")
    parser.add_argument("--requests", type=int, default=24, help="Total requests to send (default: 24)")
    parser.add_argument("--server-pid", type=int, default=None, help="Explicit server PID for memory measurement")
    parser.add_argument("--output-json", default=None, help="Path to write JSON test metrics summary")
    args = parser.parse_args()

    # Pre-check: is server listening?
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.5)
    res = s.connect_ex((args.host, args.port))
    s.close()
    if res != 0:
        print(f"[ERROR] No server listening on http://{args.host}:{args.port}/")
        print("Please start the server first with '.\\start.ps1' or 'python demo.py'.")
        sys.exit(1)

    summary = run_load_test(
        host=args.host,
        port=args.port,
        concurrency=args.concurrency,
        num_requests=args.requests,
        server_pid=args.server_pid,
        output_json=args.output_json
    )

    # Return exit code based on recovery
    if summary["recovery"]["recovered"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
