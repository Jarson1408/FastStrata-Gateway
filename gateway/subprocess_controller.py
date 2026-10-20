import sys
import os
import subprocess
import threading
import logging
from PySide6.QtCore import QObject, Signal

_log = logging.getLogger("gateway.subprocess_controller")

class SubprocessController(QObject):
    log_received = Signal(str)
    status_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.reader_thread = None
        self.stop_event = threading.Event()
        self._stopping = False  # True while stop() drives shutdown

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, config_path=None) -> bool:
        """Start the gateway subprocess."""
        if self.is_running():
            return False  # Already running

        self.stop_event = threading.Event()
        self._stopping = False

        # Command to run the gateway (unbuffered for real-time GUI logs)
        cmd = [sys.executable, "-u", "-m", "gateway"]
        if config_path:
            # __main__.py has no config-path CLI arg, so pass it via env for
            # the child's GatewayConfig.from_env() to pick up.
            os.environ["LOCAL_LLM_GATEWAY_CONFIG_PATH"] = str(config_path)

        try:
            # Start process, redirecting stdout and stderr
            # Use CREATE_NO_WINDOW on Windows to prevent spawning cmd console window
            startupinfo = None
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0  # SW_HIDE
                
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                startupinfo=startupinfo,
                bufsize=0  # unbuffered
            )
            self.process = proc
            
            _log.info("Spawned gateway subprocess PID=%d", proc.pid)
            self.status_changed.emit(True)

            # Start thread to read stdout bound to this process instance
            self.reader_thread = threading.Thread(
                target=self._read_stdout,
                args=(proc, self.stop_event),
                daemon=True
            )
            self.reader_thread.start()
            return True
        except Exception as e:
            _log.error("Failed to start gateway subprocess: %s", e)
            self.status_changed.emit(False)
            return False

    def _read_stdout(self, proc: subprocess.Popen, stop_event: threading.Event):
        """Thread worker to read stdout lines from the pipe for a specific process."""
        pipe = proc.stdout
        if not pipe:
            return

        while not stop_event.is_set():
            try:
                line = pipe.readline()
            except Exception as e:
                _log.error("Error reading from pipe: %s", e)
                break
            if not line:
                break

            # Decode line and strip trailing whitespace
            try:
                text = line.decode("utf-8", errors="replace").rstrip()
                self.log_received.emit(text)
            except Exception as e:
                _log.error("Error decoding log line: %s", e)

        # The process exited on its own (crash / external kill).
        # Only mutate state if stop() was not requested for this process AND
        # this process is still the active process tracked by the controller.
        if not stop_event.is_set() and self.process is proc:
            self.process = None
            self.status_changed.emit(False)

    def stop(self):
        """Stop the gateway subprocess gracefully, falling back to kill."""
        proc = self.process
        reader = self.reader_thread
        stop_evt = self.stop_event

        if proc is None:
            self._cleanup_polystrata()
            return

        _log.info("Stopping gateway subprocess PID=%d", proc.pid)
        self._stopping = True
        if stop_evt is not None:
            stop_evt.set()

        try:
            # On Windows, taskkill is the most reliable way to kill child process trees
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                proc.terminate()
        except Exception as e:
            _log.warning("Graceful stop signal failed, killing subprocess: %s", e)

        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass

        # Wait for the reader thread to finish reading and exit cleanly
        if reader and reader.is_alive() and threading.current_thread() != reader:
            reader.join(timeout=2.0)

        # Only clear process and emit if self.process is still this proc
        if self.process is proc:
            self.process = None
            self.status_changed.emit(False)

        # Also ensure PolyStrata server and child strata-poly are stopped
        self._cleanup_polystrata()

    def _cleanup_polystrata(self):
        try:
            from gateway.app import _stop_polystrata_server
            _stop_polystrata_server()
        except Exception as e:
            _log.debug("PolyStrata cleanup failed: %s", e)

    def restart(self, config_path=None) -> bool:
        """Restart the gateway subprocess."""
        self.stop()
        return self.start(config_path)

    def wait_for_logs(self):
        """Wait for the reader thread to finish reading. Useful in tests."""
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2.0)

