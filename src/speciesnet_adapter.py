import base64
import io
import json
import os
import queue
import subprocess
import threading
from pathlib import Path

from PIL import Image


class SpeciesNetAdapter:
    """Optional, isolated SpeciesNet worker. Disabled unless explicitly enabled."""

    def __init__(self, project_dir=None, startup_timeout=60, request_timeout=15):
        self.project_dir = Path(project_dir or Path(__file__).resolve().parent.parent)
        self.model_dir = Path(
            os.environ.get(
                "WILDID_SPECIESNET_MODEL_DIR",
                str(
                    Path.home()
                    / ".cache" / "kagglehub" / "models" / "google"
                    / "speciesnet" / "pyTorch" / "v4.0.3a" / "1"
                ),
            )
        )
        self.python = Path(
            os.environ.get(
                "WILDID_SPECIESNET_PYTHON",
                str(self.project_dir / "venv-speciesnet" / "Scripts" / "python.exe"),
            )
        )
        self.worker_script = self.project_dir / "scripts" / "speciesnet_worker.py"
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._process = None
        self._responses = None
        self._lock = threading.Lock()
        self._reader = None

    @staticmethod
    def enabled():
        return os.environ.get("WILDID_SPECIESNET_ENABLED", "0").strip() == "1"

    def _stop_worker(self):
        process = self._process
        self._process = None
        self._responses = None

        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=3)
            except Exception:
                pass
            for stream in (process.stdin, process.stdout):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass

    def _read_responses(self, process, responses):
        try:
            for line in process.stdout:
                try:
                    message = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    message = {"error": "Invalid worker response"}
                responses.put(message)
        except Exception:
            pass
        finally:
            try:
                responses.put_nowait({"error": "SpeciesNet worker stopped"})
            except queue.Full:
                pass

    def _start_worker(self):
        if self._process is not None and self._process.poll() is None:
            return

        if not self.python.is_file():
            raise RuntimeError("Configured SpeciesNet Python environment was not found")
        if not self.worker_script.is_file() or not self.model_dir.is_dir():
            raise RuntimeError("SpeciesNet worker or cached model files were not found")

        process = subprocess.Popen(
            [
                str(self.python),
                "-u",
                str(self.worker_script),
                str(self.model_dir),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        responses = queue.Queue(maxsize=2)
        self._process = process
        self._responses = responses
        self._reader = threading.Thread(
            target=self._read_responses,
            args=(process, responses),
            daemon=True,
        )
        self._reader.start()

        try:
            ready = responses.get(timeout=self.startup_timeout)
        except queue.Empty:
            self._stop_worker()
            raise TimeoutError("SpeciesNet worker startup timed out")

        if not ready.get("ready"):
            self._stop_worker()
            raise RuntimeError(ready.get("error", "SpeciesNet worker failed to start"))

    def classify_crop(self, crop):
        """Classify an in-memory PIL crop. Returns None when disabled or unavailable."""
        if not self.enabled():
            return None

        with self._lock:
            try:
                self._start_worker()

                buffer = io.BytesIO()
                crop.convert("RGB").save(buffer, format="JPEG", quality=85, optimize=False)
                payload = buffer.getvalue()

                if not payload or len(payload) > 8 * 1024 * 1024:
                    return {"error": "Crop exceeds SpeciesNet input limit"}

                request = json.dumps({
                    "image": base64.b64encode(payload).decode("ascii")
                }, separators=(",", ":"))

                self._process.stdin.write(request + "\n")
                self._process.stdin.flush()

                try:
                    response = self._responses.get(timeout=self.request_timeout)
                except queue.Empty:
                    self._stop_worker()
                    return {"error": "SpeciesNet inference timed out"}

                if response.get("error"):
                    return response

                if response.get("is_blank") or not response.get("label"):
                    return {
                        "label": None,
                        "scientific_name": None,
                        "score": response.get("score"),
                        "is_blank": bool(response.get("is_blank")),
                    }

                return response

            except Exception:
                self._stop_worker()
                return {"error": "SpeciesNet unavailable"}

    def close(self):
        with self._lock:
            self._stop_worker()
