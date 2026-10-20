import sys
import os
import html
import json
import logging
import subprocess
import threading
from collections import deque
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QTextEdit, QLabel, QPushButton, QLineEdit,
    QCheckBox, QFileDialog, QComboBox, QMessageBox, QSystemTrayIcon, QMenu,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QInputDialog, QDialog, QDialogButtonBox, QFormLayout
)
from PySide6.QtCore import Slot, Signal, QObject, Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PySide6.QtNetwork import QLocalServer, QLocalSocket

from gateway.config import SAMPLING_PRESETS, GatewayConfig
from gateway.config_file import (
    load_settings_from_file, save_settings_to_file, active_models,
    get_presets, get_active_preset_name, save_preset, delete_preset
)
from gateway.model_registry import (
    ENGINE_POLYSTRATA, ENGINE_REMOTE, KNOWN_ENGINES,
    ModelRegistry, ModelSpec, ModelSpecError,
)
from gateway import opencode_export
from gateway.subprocess_controller import SubprocessController
from gateway.gui_helpers import (
    deploy_lodestar_template, set_autostart, get_startup_shortcut_path,
    classify_log_line, describe_save_outcome, describe_health_failure,
    audit_log_path, format_upstream_chip,
)
from gateway import memory_admin
from gateway.session_report import (
    build_sessions, format_session, read_events,
)

#: Where the app writes what it would print, when it has no console.
GUI_LOG = Path(__file__).resolve().parent.parent / "gui.log"
GUI_LOG_MAX_BYTES = 5 * 1024 * 1024


def log_to_file_when_windowless(path: Path = GUI_LOG) -> bool:
    """Under pythonw -- the desktop shortcut FastStrata's setup makes -- stdout and stderr are None, so a log line
    and the traceback of an error that ends the app go nowhere: on 2026-10-20 the app started the model and was gone
    a minute later, and nothing on disk said why. Then both go to gui.log beside the gateway (the one before it kept
    as gui.log.1 past 5 MB), and an uncaught exception is logged with its traceback. True when it did that."""
    if sys.stdout is not None and sys.stderr is not None:
        return False
    try:
        if path.is_file() and path.stat().st_size > GUI_LOG_MAX_BYTES:
            os.replace(path, path.with_name(path.name + ".1"))
        f = open(path, "a", encoding="utf-8", buffering=1)
    except OSError:
        return False
    sys.stdout = sys.stderr = f

    def log_uncaught(kind, value, tb):
        logging.getLogger("gateway.gui").critical("uncaught error", exc_info=(kind, value, tb))

    sys.excepthook = log_uncaught
    threading.excepthook = lambda args: log_uncaught(args.exc_type, args.exc_value, args.exc_traceback)
    return True


log_to_file_when_windowless()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
_log = logging.getLogger("gateway.gui")

# Obsidian Flux Premium Stylesheet
DARK_STYLESHEET = """
QMainWindow {
    background-color: #131314;
}
QWidget {
    background-color: #131314;
    color: #e5e2e3;
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
}
/* Sidebar Style */
QFrame#sidebar {
    background-color: #0e0e0f;
    border-right: 1px solid #1f1e20;
    max-width: 220px;
    min-width: 220px;
}
QLabel#sidebar-title {
    font-size: 16px;
    font-weight: 900;
    color: #7c4dff;
    letter-spacing: -0.02em;
    padding: 10px 0px;
}
QPushButton#nav-btn {
    background-color: transparent;
    color: #cac3d8;
    border: none;
    border-radius: 6px;
    padding: 10px 16px;
    text-align: left;
    font-weight: 500;
}
QPushButton#nav-btn:hover {
    background-color: #1f1e20;
    color: #e5e2e3;
}
QPushButton#nav-btn:checked {
    background-color: rgba(124, 77, 255, 0.12);
    color: #7c4dff;
    border-left: 3px solid #7c4dff;
    font-weight: bold;
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
}

/* Central View Style */
QStackedWidget {
    background-color: #131314;
}
QFrame#glass-panel {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 16px;
}
QFrame#terminal-panel {
    background-color: #09090b;
    border: 1px solid #201f20;
    border-radius: 8px;
}

/* Form Fields */
QLineEdit {
    background-color: transparent;
    color: #e5e2e3;
    border: none;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    padding: 6px 0px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 14px;
}
QLineEdit:focus {
    border-bottom: 1px solid #7c4dff;
}

/* Preset Combobox */
QComboBox {
    background-color: #1a1a1d;
    color: #e5e2e3;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
}
QComboBox:focus {
    border: 1px solid #7c4dff;
}
QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #1a1a1d;
    color: #e5e2e3;
    selection-background-color: #7c4dff;
    border: 1px solid rgba(255, 255, 255, 0.15);
}

/* Custom Text area for Config JSON */
QTextEdit#config-json {
    background-color: #09090b;
    color: #cac3d8;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}
QTextEdit#config-json:focus {
    border: 1px solid #7c4dff;
}

/* Text area read-only for Logs */
QTextEdit#logs-view {
    background-color: transparent;
    border: none;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}

/* Toggles & Checkboxes */
QCheckBox {
    spacing: 10px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    background-color: #0e0e0f;
}
QCheckBox::indicator:checked {
    background-color: #7c4dff;
    border-color: #7c4dff;
}

/* Buttons */
QPushButton#btn-primary {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #7c4dff, stop:1 #9e79ff);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: bold;
}
QPushButton#btn-primary:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #8b61ff, stop:1 #ab8cff);
}
QPushButton#btn-primary:pressed {
    background-color: #6230ea;
}
QPushButton#btn-secondary {
    background-color: transparent;
    color: #cac3d8;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 6px;
    padding: 8px 16px;
}
QPushButton#btn-secondary:hover {
    background-color: rgba(255, 255, 255, 0.05);
    color: #e5e2e3;
    border-color: rgba(255, 255, 255, 0.4);
}

/* Memory table */
QTableWidget {
    background-color: #09090b;
    color: #e5e2e3;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    gridline-color: #201f20;
    font-size: 12px;
}
QTableWidget::item {
    padding: 4px 8px;
    border-bottom: 1px solid #1a191b;
}
QTableWidget::item:selected {
    background-color: rgba(124, 77, 255, 0.22);
    color: #ffffff;
}
QHeaderView::section {
    background-color: #0e0e0f;
    color: #cac3d8;
    border: none;
    border-bottom: 1px solid #201f20;
    padding: 6px 8px;
    font-weight: bold;
}

/* LODESTAR badge in the sidebar status panel */
QLabel#lodestar-badge-on {
    color: #00e676;
    font-weight: bold;
    font-size: 11px;
    padding: 2px 0px;
}
QLabel#lodestar-badge-off {
    color: #ffb020;
    font-weight: bold;
    font-size: 11px;
    padding: 2px 0px;
}

/* Helper labels */
QLabel#lbl-section-title {
    font-size: 20px;
    font-weight: 700;
    color: #e5e2e3;
    letter-spacing: -0.02em;
}
QLabel#lbl-section-subtitle {
    font-size: 12px;
    color: #cac3d8;
    opacity: 0.7;
}
"""

class HealthPoller(QObject):
    """Polls the gateway's /health endpoint off the GUI thread.

    Emits health_received with the parsed JSON dict, or {} when the gateway
    (or its upstream llama-server) is unreachable. The Mac Studio going to
    sleep is a recurring real-world failure mode — surfacing upstream health
    in the dashboard beats discovering it via a stalled OpenCode request.
    """
    health_received = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._busy = False

    def poll(self, port: int):
        if self._busy:
            return
        self._busy = True

        def work():
            data = {}
            try:
                import urllib.request
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=4) as r:
                    parsed = json.loads(r.read().decode("utf-8"))
                    if isinstance(parsed, dict):
                        data = parsed
            except Exception as exc:
                # Collapsing every failure into {} made a stopped gateway, a
                # busy port, a timeout and a malformed response light the same
                # red chip — the exact information missing while two gateways
                # raced for port 18000 (O-2026-05).
                data = describe_health_failure(exc)
                _log.debug("Health poll failed: %s", data.get("detail"))
            finally:
                self._busy = False
            self.health_received.emit(data)

        threading.Thread(target=work, daemon=True).start()


def create_status_pixmap(color_name: str) -> QPixmap:
    """Create a QPixmap of a status dot (Green, Yellow, Red) programmatically."""
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    color = QColor(color_name)
    
    # Draw outer glow
    glow = QColor(color)
    glow.setAlpha(80)
    painter.setBrush(glow)
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(1, 1, 14, 14)
    
    # Draw core dot
    painter.setBrush(color)
    painter.drawEllipse(3, 3, 10, 10)
    painter.end()
    
    return pixmap


class ModelDialog(QDialog):
    """Settings for one model of one preset.

    Deliberately thin: every field maps to a key of `ModelSpec`, and the only
    logic here is which fields make sense for which engine. Validation is
    `ModelSpec.from_dict`'s job — one definition of what a usable spec is,
    shared with the gateway that has to serve it.
    """

    def __init__(self, parent=None, spec: dict | None = None,
                 default_upstream: str = "", inherited: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Model Settings")
        self.setMinimumWidth(600)
        spec = dict(spec or {})
        #: What the PRESET would give a field left blank. Shown as placeholder
        #: text so the dialog answers "what happens if I leave this empty"
        #: without sending the user to another tab to find out.
        inherited = dict(inherited or {})

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(10)

        self.txt_id = QLineEdit(str(spec.get("id", "")))
        self.txt_id.setToolTip("What the client asks for. This is the only "
                               "name OpenCode ever sees.")
        form.addRow("Model ID", self.txt_id)

        self.txt_label = QLineEdit(str(spec.get("label", "")))
        form.addRow("Display name", self.txt_label)

        self.cmb_engine = QComboBox()
        self.cmb_engine.addItems([ENGINE_POLYSTRATA, ENGINE_REMOTE])
        engine = str(spec.get("engine") or ENGINE_POLYSTRATA)
        if engine in KNOWN_ENGINES:
            self.cmb_engine.setCurrentText(engine)
        form.addRow("Engine", self.cmb_engine)

        self.txt_upstream = QLineEdit(str(spec.get("upstream") or default_upstream))
        form.addRow("Upstream", self.txt_upstream)

        self.txt_target = QLineEdit(str(spec.get("target", "")))
        self.txt_target.setToolTip("What the ENGINE calls this model: the run "
                                   "config FastStrata's setup wrote "
                                   "(strata-*.json in the FastStrata folder), "
                                   "or the remote server's model name.")
        form.addRow("Engine's own name", self.txt_target)

        self.cmb_family = QComboBox()
        self.cmb_family.addItems(["", "qwen3", "qwen25", "gemma"])
        self.cmb_family.setCurrentText(str(spec.get("family", "")))
        self.cmb_family.setToolTip("Which tool-call parser to use. Blank "
                                   "guesses from the model name, which is "
                                   "weaker than saying so.")
        form.addRow("Parser family", self.cmb_family)

        self.txt_n_ctx = QLineEdit(str(spec.get("n_ctx", "") or ""))
        form.addRow("Context window (tokens)", self.txt_n_ctx)

        self.txt_max_output = QLineEdit(str(spec.get("max_output", 32768)))
        form.addRow("Advertised output cap", self.txt_max_output)

        self.chk_vision = QCheckBox("This model accepts images")
        self.chk_vision.setChecked(bool(spec.get("vision", False)))
        form.addRow("", self.chk_vision)

        # Limits that scale with the window. Blank means "use the preset's",
        # and the placeholder shows what that currently is -- so the dialog
        # answers "what happens if I leave this empty" without the user having
        # to go and look. A value right for a 229k window is wrong for a 32k
        # one, which is the whole reason these are here.
        self.txt_reserve = QLineEdit(str(spec.get("context_reserve", "") or ""))
        self.txt_reserve.setPlaceholderText(
            f"inherit from preset ({inherited.get('context_reserve', 16384)})")
        self.txt_reserve.setToolTip(
            "Tokens held back for the response. prompt + output = the window, "
            "so this is subtracted from n_ctx to get the prune limit.")
        form.addRow("Context reserve", self.txt_reserve)

        self.txt_n_predict = QLineEdit(str(spec.get("max_n_predict", "") or ""))
        self.txt_n_predict.setPlaceholderText(
            f"inherit from preset ({inherited.get('max_n_predict', 16384)})")
        self.txt_n_predict.setToolTip(
            "Hard cap on generated tokens. The reasoning budget is DERIVED "
            "from this, so that it still fires before the ceiling does.")
        form.addRow("Max output tokens", self.txt_n_predict)

        self.txt_spill = QLineEdit(str(spec.get("spill_max_inline_bytes", "") or ""))
        self.txt_spill.setPlaceholderText(
            f"inherit from preset ({inherited.get('spill_max_inline_bytes', 65536)})")
        self.txt_spill.setToolTip(
            "Bytes past which a tool result goes to disk. 65536 bytes is "
            "about 21k tokens: fine in a 229k window, two thirds of a 32k one.")
        form.addRow("Spill threshold (bytes)", self.txt_spill)

        self.cmb_sampling_preset = QComboBox()
        self.cmb_sampling_preset.addItem("")
        self.cmb_sampling_preset.addItems(sorted(SAMPLING_PRESETS))
        self.cmb_sampling_preset.setCurrentText(str(spec.get("sampling_preset", "")))
        self.cmb_sampling_preset.setToolTip(
            "Blank uses the preset's sampling. Different model families want "
            "different parameters.")
        form.addRow("Sampling preset", self.cmb_sampling_preset)

        self.txt_sampling = QLineEdit(
            json.dumps(spec.get("sampling")) if spec.get("sampling") else "")
        self.txt_sampling.setPlaceholderText('{"temperature": 0.15}')
        self.txt_sampling.setToolTip(
            "Per-parameter overrides, JSON. These beat what the client sends "
            "- OpenCode hardcodes temperature for any model id containing "
            "'qwen', which is a guess keyed on a name.")
        form.addRow("Sampling overrides", self.txt_sampling)

        self.txt_notes = QLineEdit(str(spec.get("notes", "")))
        form.addRow("Notes", self.txt_notes)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def to_dict(self) -> dict:
        data: dict = {
            "id": self.txt_id.text().strip(),
            "engine": self.cmb_engine.currentText().strip(),
            "upstream": self.txt_upstream.text().strip(),
            "label": self.txt_label.text().strip(),
            "target": self.txt_target.text().strip(),
            "family": self.cmb_family.currentText().strip(),
            "vision": self.chk_vision.isChecked(),
            "notes": self.txt_notes.text().strip(),
        }
        for key, field in (("n_ctx", self.txt_n_ctx),
                           ("max_output", self.txt_max_output),
                           ("context_reserve", self.txt_reserve),
                           ("max_n_predict", self.txt_n_predict),
                           ("spill_max_inline_bytes", self.txt_spill)):
            raw = field.text().strip()
            if raw:
                data[key] = int(raw)
        preset_name = self.cmb_sampling_preset.currentText().strip()
        if preset_name:
            data["sampling_preset"] = preset_name
        sampling_raw = self.txt_sampling.text().strip()
        if sampling_raw:
            # Parsed here so `_accept_if_valid` refuses malformed JSON through
            # the same path as everything else: the dialog must not be able to
            # store a value the request path cannot read.
            data["sampling"] = json.loads(sampling_raw)
        return data

    def _accept_if_valid(self):
        """Refuse a spec the gateway could not serve, here rather than later.

        `ModelSpec.from_dict` is the one definition of that, so the dialog
        cannot drift from what the request path accepts.
        """
        try:
            ModelSpec.from_dict(self.to_dict())
        except json.JSONDecodeError as exc:
            QMessageBox.warning(
                self, "Sampling overrides are not JSON",
                'Expected something like {"temperature": 0.15}.\n\n' + str(exc))
            return
        except (ModelSpecError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Not a usable model", str(exc))
            return
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self, controller: SubprocessController):
        super().__init__()
        self.controller = controller
        self.config = GatewayConfig.from_env()
        # Reference to the tray's LODESTAR action, set by main() after the
        # tray is built. Kept in sync with the settings checkbox through
        # set_lodestar_enabled() so both surfaces agree at all times.
        self.tray_lodestar_action = None
        # Tray icon reference (set by main()) — used for live tooltip updates.
        self.tray_icon = None
        self._is_running = False
        self._upstream_ok: bool | None = None
        # The raw model dicts of the preset on screen. Raw rather than parsed,
        # because the table is an EDITOR: a spec that does not parse still has
        # to be visible so it can be fixed. Assigned here rather than as a
        # class attribute, which would be one list shared by every instance and
        # mutated in place by the add/remove handlers.
        self._models: list[dict] = []

        self.setWindowTitle("FastStrata Gateway")
        self.resize(920, 560)
        self.setStyleSheet(DARK_STYLESHEET)

        # Main Layout (Horizontal)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Initialize Sidebar
        self.sidebar_frame = QFrame(objectName="sidebar")
        self._init_sidebar(self.sidebar_frame)
        main_layout.addWidget(self.sidebar_frame)

        # Initialize Stacked Views (Center Area)
        self.view_stack = QStackedWidget()
        main_layout.addWidget(self.view_stack)

        # Setup Views
        self._init_logs_view()
        self._init_settings_view()
        self._init_lodestar_view()
        self._init_memory_view()
        self._init_sessions_view()

        # Connect Subprocess Controller
        self.controller.log_received.connect(self.append_log)
        self.controller.status_changed.connect(self.on_subprocess_status_changed)

        # Upstream health polling (only while the gateway is running)
        self.health_poller = HealthPoller(self)
        self.health_poller.health_received.connect(self.on_health_received)
        self.health_timer = QTimer(self)
        self.health_timer.setInterval(15000)
        self.health_timer.timeout.connect(self._poll_health)
        self.health_timer.start()
        
        # Load config to UI components
        self.load_config_to_ui()

        # Set default active page (Logs)
        self.btn_nav_logs.setChecked(True)
        self.view_stack.setCurrentIndex(0)

        # Start gateway automatically on launch
        self.start_gateway()

    def _init_sidebar(self, frame: QFrame):
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 24, 16, 24)
        layout.setSpacing(8)

        # Brand Title
        title = QLabel("LOCAL GATEWAY", objectName="sidebar-title")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(title)

        subtitle = QLabel("Obsidian Node 01")
        subtitle.setStyleSheet("color: #cac3d8; font-size: 11px; opacity: 0.5; margin-bottom: 24px;")
        layout.addWidget(subtitle)

        # Navigation Buttons (Checkable, Mutually Exclusive)
        self.btn_nav_logs = QPushButton("🎛️  Logs Stream", objectName="nav-btn")
        self.btn_nav_logs.setCheckable(True)
        self.btn_nav_logs.setChecked(True)
        self.btn_nav_logs.clicked.connect(lambda: self.switch_view(0))
        layout.addWidget(self.btn_nav_logs)

        self.btn_nav_settings = QPushButton("⚙️  Configuration", objectName="nav-btn")
        self.btn_nav_settings.setCheckable(True)
        self.btn_nav_settings.clicked.connect(lambda: self.switch_view(1))
        layout.addWidget(self.btn_nav_settings)

        self.btn_nav_lodestar = QPushButton("🧠  Lodestar Deployer", objectName="nav-btn")
        self.btn_nav_lodestar.setCheckable(True)
        self.btn_nav_lodestar.clicked.connect(lambda: self.switch_view(2))
        layout.addWidget(self.btn_nav_lodestar)

        self.btn_nav_memory = QPushButton("💾  Memory Manager", objectName="nav-btn")
        self.btn_nav_memory.setCheckable(True)
        self.btn_nav_memory.clicked.connect(lambda: self.switch_view(3))
        layout.addWidget(self.btn_nav_memory)

        self.btn_nav_sessions = QPushButton("🧭  Sessions", objectName="nav-btn")
        self.btn_nav_sessions.setCheckable(True)
        self.btn_nav_sessions.clicked.connect(lambda: self.switch_view(4))
        layout.addWidget(self.btn_nav_sessions)

        # Group them for exclusive selection
        self.nav_buttons = [self.btn_nav_logs, self.btn_nav_settings,
                            self.btn_nav_lodestar, self.btn_nav_memory,
                            self.btn_nav_sessions]

        layout.addStretch()

        # Quick Control Panel in Sidebar
        status_panel = QFrame()
        status_panel.setStyleSheet("background-color: #131314; border-radius: 8px; padding: 10px; border: 1px solid #1f1e20;")
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(8, 8, 8, 8)

        self.lbl_sidebar_status = QLabel("Status: Offline")
        self.lbl_sidebar_status.setStyleSheet("color: #ff1744; font-weight: bold; font-size: 11px;")
        status_layout.addWidget(self.lbl_sidebar_status)

        # LODESTAR protocol badge — reflects the persisted toggle
        self.lbl_lodestar_badge = QLabel("LODESTAR: on")
        self.lbl_lodestar_badge.setObjectName("lodestar-badge-on")
        status_layout.addWidget(self.lbl_lodestar_badge)

        self.btn_sidebar_restart = QPushButton("🔄 Restart Node")
        self.btn_sidebar_restart.setStyleSheet("background-color: #7c4dff; color: white; border: none; border-radius: 4px; padding: 6px; font-size: 11px;")
        self.btn_sidebar_restart.clicked.connect(self.restart_gateway)
        status_layout.addWidget(self.btn_sidebar_restart)

        layout.addWidget(status_panel)

    def switch_view(self, index: int):
        # Update Nav Checkboxes
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        self.view_stack.setCurrentIndex(index)

    def _init_logs_view(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header Title
        header_layout = QHBoxLayout()
        v_title = QVBoxLayout()
        v_title.addWidget(QLabel("Live Gateway Stream", objectName="lbl-section-title"))
        
        meta_layout = QHBoxLayout()
        self.lbl_pid = QLabel("PID: --")
        self.lbl_pid.setStyleSheet("color: #cac3d8; font-family: monospace; font-size: 11px; background-color: #201f20; padding: 2px 6px; border-radius: 4px;")
        meta_layout.addWidget(self.lbl_pid)
        
        self.lbl_port_indicator = QLabel("PORT: 18000")
        self.lbl_port_indicator.setStyleSheet("color: #cac3d8; font-family: monospace; font-size: 11px; background-color: #201f20; padding: 2px 6px; border-radius: 4px;")
        meta_layout.addWidget(self.lbl_port_indicator)

        # Upstream (llama-server) health chip, fed by HealthPoller
        self.lbl_upstream = QLabel("UPSTREAM: --")
        self.lbl_upstream.setStyleSheet("color: #cac3d8; font-family: monospace; font-size: 11px; background-color: #201f20; padding: 2px 6px; border-radius: 4px;")
        meta_layout.addWidget(self.lbl_upstream)
        meta_layout.addStretch()
        v_title.addLayout(meta_layout)
        
        header_layout.addLayout(v_title)
        header_layout.addStretch()

        # Status Circle Indicator
        self.lbl_status_indicator = QLabel()
        self.lbl_status_indicator.setPixmap(create_status_pixmap("#ff1744"))
        header_layout.addWidget(self.lbl_status_indicator)

        self.lbl_status_text = QLabel("Offline")
        self.lbl_status_text.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_status_text.setStyleSheet("color: #ff1744; margin-right: 8px;")
        header_layout.addWidget(self.lbl_status_text)
        layout.addLayout(header_layout)

        # Terminal Panel
        terminal_panel = QFrame(objectName="terminal-panel")
        terminal_layout = QVBoxLayout(terminal_panel)
        terminal_layout.setContentsMargins(0, 0, 0, 0)

        # Fake terminal titlebar
        titlebar = QFrame()
        titlebar.setStyleSheet("background-color: #131314; border-bottom: 1px solid #201f20; border-top-left-radius: 8px; border-top-right-radius: 8px;")
        titlebar.setFixedHeight(30)
        titlebar_layout = QHBoxLayout(titlebar)
        titlebar_layout.setContentsMargins(12, 0, 12, 0)
        
        dots = QLabel("● ● ●")
        dots.setStyleSheet("color: rgba(255,255,255,0.2); font-size: 10px;")
        titlebar_layout.addWidget(dots)
        
        titlebar_layout.addStretch()
        titlebar_layout.addWidget(QLabel("gateway-stdout -- 127.0.0.1", styleSheet="color: #cac3d8; font-size: 10px; opacity: 0.5;"))
        terminal_layout.addWidget(titlebar)

        self.log_text = QTextEdit(objectName="logs-view")
        self.log_text.setReadOnly(True)
        # Cap the in-memory log buffer: gateway sessions run for hours and an
        # unbounded QTextEdit slowly eats RAM and repaint time.
        self.log_text.document().setMaximumBlockCount(5000)
        terminal_layout.addWidget(self.log_text)

        # Error-like lines (ERROR/WARNING) are additionally kept in their own
        # bounded buffer so routine traffic can never scroll a problem out of
        # reach before the user looks at it.
        self._live_buffer: deque[str] = deque(maxlen=5000)
        self._error_buffer: deque[str] = deque(maxlen=1000)
        self._errors_only = False
        layout.addWidget(terminal_panel)

        # Footer Actions
        footer_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("🚀 Run Proxy", objectName="btn-primary")
        self.btn_start.clicked.connect(self.start_gateway)
        footer_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("🛑 Stop Node", objectName="btn-secondary")
        self.btn_stop.clicked.connect(self.stop_gateway)
        footer_layout.addWidget(self.btn_stop)

        self.btn_errors = QPushButton("⚠ Errors (0)", objectName="btn-secondary")
        self.btn_errors.setCheckable(True)
        self.btn_errors.setToolTip(
            "Show only ERROR/WARNING lines. Kept in a separate buffer, so they "
            "survive even when routine traffic fills the live view.")
        self.btn_errors.toggled.connect(self._set_errors_only)
        footer_layout.addWidget(self.btn_errors)

        btn_clear = QPushButton("Clear logs", objectName="btn-secondary")
        btn_clear.clicked.connect(self.clear_logs)
        footer_layout.addWidget(btn_clear)

        btn_open_file = QPushButton("Open Log in Notepad", objectName="btn-secondary")
        btn_open_file.clicked.connect(self.open_log_in_notepad)
        footer_layout.addWidget(btn_open_file)

        footer_layout.addStretch()
        layout.addLayout(footer_layout)

        self.view_stack.addWidget(page)

    def _init_settings_view(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(QLabel("Gateway Configuration", objectName="lbl-section-title"))
        layout.addWidget(QLabel("Modify local LLM gateway proxy rules, upstream servers, and aliases", objectName="lbl-section-subtitle"))

        # Form Glass Panel
        form_panel = QFrame(objectName="glass-panel")
        form_layout = QVBoxLayout(form_panel)
        form_layout.setSpacing(16)

        # Configuration Presets Bar
        form_layout.addWidget(QLabel("CONFIGURATION PRESET"))
        preset_row = QHBoxLayout()
        self.cmb_presets = QComboBox()
        self.cmb_presets.setMinimumHeight(32)
        self.cmb_presets.currentIndexChanged.connect(self._on_preset_selected)
        preset_row.addWidget(self.cmb_presets, stretch=1)

        self.btn_save_preset = QPushButton("💾 Save as Preset...", objectName="btn-secondary")
        self.btn_save_preset.clicked.connect(self._on_save_as_preset)
        preset_row.addWidget(self.btn_save_preset)

        self.btn_delete_preset = QPushButton("🗑️ Delete", objectName="btn-secondary")
        self.btn_delete_preset.clicked.connect(self._on_delete_preset)
        preset_row.addWidget(self.btn_delete_preset)

        form_layout.addLayout(preset_row)

        # Upstream Address Input
        form_layout.addWidget(QLabel("UPSTREAM LLM HOST ADDRESS"))
        self.txt_upstream = QLineEdit()
        self.txt_upstream.setPlaceholderText("e.g. http://127.0.0.1:8080")
        form_layout.addWidget(self.txt_upstream)

        # Port, Reserve & Spill Threshold (Horizontal)
        h_layout = QHBoxLayout()
        
        v_port = QVBoxLayout()
        v_port.addWidget(QLabel("GATEWAY LOCAL PORT"))
        self.txt_port = QLineEdit()
        v_port.addWidget(self.txt_port)
        h_layout.addLayout(v_port)

        v_reserve = QVBoxLayout()
        v_reserve.addWidget(QLabel("CONTEXT RESERVE TOKENS"))
        self.txt_reserve = QLineEdit()
        v_reserve.addWidget(self.txt_reserve)
        h_layout.addLayout(v_reserve)

        v_spill = QVBoxLayout()
        v_spill.addWidget(QLabel("SPILL THRESHOLD (BYTES)"))
        self.txt_spill = QLineEdit()
        self.txt_spill.setPlaceholderText("8192")
        v_spill.addWidget(self.txt_spill)
        h_layout.addLayout(v_spill)

        v_ctx = QVBoxLayout()
        v_ctx.addWidget(QLabel("FALLBACK N_CTX (TOKENS)"))
        self.txt_fallback_n_ctx = QLineEdit()
        self.txt_fallback_n_ctx.setPlaceholderText("32768")
        v_ctx.addWidget(self.txt_fallback_n_ctx)
        h_layout.addLayout(v_ctx)
        
        form_layout.addLayout(h_layout)

        # Queue Timeout & Stream Max Duration (Horizontal)
        h_timeouts = QHBoxLayout()

        v_queue = QVBoxLayout()
        v_queue.addWidget(QLabel("QUEUE TIMEOUT (SECONDS)"))
        self.txt_queue_timeout = QLineEdit()
        self.txt_queue_timeout.setPlaceholderText("7200")
        v_queue.addWidget(self.txt_queue_timeout)
        h_timeouts.addLayout(v_queue)

        v_stream = QVBoxLayout()
        v_stream.addWidget(QLabel("STREAM MAX DURATION (SECONDS)"))
        self.txt_stream_duration = QLineEdit()
        self.txt_stream_duration.setPlaceholderText("7200")
        v_stream.addWidget(self.txt_stream_duration)
        h_timeouts.addLayout(v_stream)

        form_layout.addLayout(h_timeouts)

        # Model Aliases JSON Box
        form_layout.addWidget(QLabel("MODEL ALIASES MAPPING (JSON)"))
        self.txt_aliases = QTextEdit(objectName="config-json")
        self.txt_aliases.setMaximumHeight(80)
        self.txt_aliases.setAcceptRichText(False)
        form_layout.addWidget(self.txt_aliases)

        # ------------------------------------------------------------------
        # The models this preset serves, and what runs each one.
        #
        # A preset used to be one upstream plus a flat alias map. With the
        # FastStrata models taking turns on the same card, the model IS the
        # routing decision — so the list has to be visible and editable where
        # the preset is chosen, rather than being a JSON blob somebody edits in
        # Notepad.
        # ------------------------------------------------------------------
        form_layout.addWidget(QLabel("MODELS IN THIS PRESET"))
        self.tbl_models = QTableWidget(0, 5)
        self.tbl_models.setHorizontalHeaderLabels(
            ["MODEL ID", "ENGINE", "CONTEXT", "VISION", "STATUS"])
        self.tbl_models.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_models.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_models.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_models.verticalHeader().setVisible(False)
        self.tbl_models.setMinimumHeight(150)
        self.tbl_models.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        for col in range(1, 5):
            self.tbl_models.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents)
        self.tbl_models.doubleClicked.connect(self._on_edit_model)
        form_layout.addWidget(self.tbl_models)

        models_row = QHBoxLayout()
        self.btn_model_add = QPushButton("➕ Add Model...", objectName="btn-secondary")
        self.btn_model_add.clicked.connect(self._on_add_model)
        models_row.addWidget(self.btn_model_add)

        self.btn_model_edit = QPushButton("⚙️ Model Settings...", objectName="btn-secondary")
        self.btn_model_edit.clicked.connect(self._on_edit_model)
        models_row.addWidget(self.btn_model_edit)

        self.btn_model_remove = QPushButton("🗑️ Remove", objectName="btn-secondary")
        self.btn_model_remove.clicked.connect(self._on_remove_model)
        models_row.addWidget(self.btn_model_remove)

        models_row.addStretch()

        self.btn_export_opencode = QPushButton("📤 Export to OpenCode",
                                               objectName="btn-secondary")
        self.btn_export_opencode.setToolTip(
            "Write these models into OpenCode's own config. OpenCode does not "
            "discover a custom provider's models — the list in its config is "
            "the list. A .bak is kept and nothing else in the file is touched.")
        self.btn_export_opencode.clicked.connect(self._on_export_opencode)
        models_row.addWidget(self.btn_export_opencode)

        form_layout.addLayout(models_row)

        # Checkboxes (Toggles)
        self.chk_lodestar = QCheckBox("Enforce LODESTAR Memory Protocol (default: enabled)")
        form_layout.addWidget(self.chk_lodestar)

        self.chk_autostart = QCheckBox("Start automatically on Windows boot (adds Startup shortcut)")
        form_layout.addWidget(self.chk_autostart)

        layout.addWidget(form_panel)
        layout.addStretch()

        # Save Button
        save_layout = QHBoxLayout()
        save_layout.addStretch()
        
        btn_save = QPushButton("💾 Save & Apply Changes", objectName="btn-primary")
        btn_save.clicked.connect(self.save_config)
        save_layout.addWidget(btn_save)
        layout.addLayout(save_layout)

        self.view_stack.addWidget(page)

    def _init_lodestar_view(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(QLabel("Lodestar Template Deployer", objectName="lbl-section-title"))
        layout.addWidget(QLabel("Bootstraps a local workspace directory with the Lodestar framework template", objectName="lbl-section-subtitle"))

        # Info Box (Source Template)
        src_panel = QFrame(objectName="glass-panel")
        src_layout = QVBoxLayout(src_panel)
        src_layout.addWidget(QLabel("SOURCE TEMPLATE LOCATION:"))
        
        gateway_root = Path(__file__).resolve().parent.parent
        template_dir = gateway_root / "LODESTAR_template"
        
        lbl_tmpl = QLabel(f"<code>{template_dir}</code>")
        lbl_tmpl.setTextFormat(Qt.RichText)
        lbl_tmpl.setStyleSheet("color: #7c4dff; font-family: monospace; font-size: 11px;")
        src_layout.addWidget(lbl_tmpl)
        layout.addWidget(src_panel)
        layout.addSpacing(16)

        # Target folder selector
        layout.addWidget(QLabel("TARGET WORKSPACE DIRECTORY"))
        target_layout = QHBoxLayout()
        self.txt_target_dir = QLineEdit()
        self.txt_target_dir.setPlaceholderText("Select project directory path...")
        target_layout.addWidget(self.txt_target_dir)

        btn_browse = QPushButton("Browse...", objectName="btn-secondary")
        btn_browse.clicked.connect(self.browse_target_directory)
        target_layout.addWidget(btn_browse)
        layout.addLayout(target_layout)

        layout.addStretch()

        # Deploy Action
        deploy_layout = QHBoxLayout()
        deploy_layout.addStretch()
        
        btn_deploy = QPushButton("⚡ Deploy Template", objectName="btn-primary")
        btn_deploy.clicked.connect(self.deploy_template)
        deploy_layout.addWidget(btn_deploy)
        layout.addLayout(deploy_layout)

        self.view_stack.addWidget(page)

    def _init_memory_view(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(QLabel("Semantic Memory Manager", objectName="lbl-section-title"))
        layout.addWidget(QLabel(
            "Browse, prune and back up the lessons stored in a project's "
            "semantic-memory database", objectName="lbl-section-subtitle"))

        # Project selector
        sel_layout = QHBoxLayout()
        self.txt_memory_dir = QLineEdit()
        self.txt_memory_dir.setPlaceholderText("Select a LODESTAR project directory...")
        sel_layout.addWidget(self.txt_memory_dir)

        btn_mem_browse = QPushButton("Browse...", objectName="btn-secondary")
        btn_mem_browse.clicked.connect(self.browse_memory_directory)
        sel_layout.addWidget(btn_mem_browse)

        btn_mem_load = QPushButton("Load", objectName="btn-secondary")
        btn_mem_load.clicked.connect(self.refresh_memory_table)
        sel_layout.addWidget(btn_mem_load)
        layout.addLayout(sel_layout)

        self.lbl_memory_count = QLabel("No project loaded.")
        self.lbl_memory_count.setStyleSheet("color: #cac3d8; font-size: 11px; opacity: 0.7;")
        layout.addWidget(self.lbl_memory_count)

        # Memory table
        self.memory_table = QTableWidget(0, 4)
        self.memory_table.setHorizontalHeaderLabels(["ID", "Category", "When", "Lesson"])
        self.memory_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.memory_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.memory_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.memory_table.verticalHeader().setVisible(False)
        header = self.memory_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        layout.addWidget(self.memory_table)

        # Actions
        action_layout = QHBoxLayout()
        btn_mem_delete = QPushButton("🗑 Delete Selected", objectName="btn-secondary")
        btn_mem_delete.clicked.connect(self.delete_selected_memory)
        action_layout.addWidget(btn_mem_delete)

        btn_mem_export = QPushButton("📤 Export to JSON", objectName="btn-secondary")
        btn_mem_export.clicked.connect(self.export_memory)
        action_layout.addWidget(btn_mem_export)

        action_layout.addStretch()
        hint = QLabel("Import a backup with:  python .context/semantic-memory/embed.py import <file.json>")
        hint.setStyleSheet("color: #cac3d8; font-size: 10px; opacity: 0.5;")
        action_layout.addWidget(hint)
        layout.addLayout(action_layout)

        self.view_stack.addWidget(page)

    def _init_sessions_view(self):
        """Replay a past session from the audit log.

        Every decision the gateway made — the state it derived, the prompt
        digest, the calls it blocked or made itself — is already in
        gateway_events.jsonl. Reading it as a timeline is the difference
        between debugging the model and debugging the gateway.
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(QLabel("Sessions", objectName="lbl-section-title"))
        layout.addWidget(QLabel(
            "Replay what the gateway showed the model, turn by turn",
            objectName="lbl-section-subtitle"))

        sel_layout = QHBoxLayout()
        self.cmb_session = QComboBox()
        self.cmb_session.setMinimumWidth(360)
        self.cmb_session.currentIndexChanged.connect(self.show_selected_session)
        sel_layout.addWidget(self.cmb_session)

        btn_reload = QPushButton("Reload", objectName="btn-secondary")
        btn_reload.clicked.connect(self.refresh_sessions)
        sel_layout.addWidget(btn_reload)
        sel_layout.addStretch()
        layout.addLayout(sel_layout)

        self.lbl_session_hint = QLabel("Press Reload to read the audit log.")
        self.lbl_session_hint.setStyleSheet("color: #cac3d8; font-size: 11px; opacity: 0.7;")
        layout.addWidget(self.lbl_session_hint)

        self.txt_session = QTextEdit()
        self.txt_session.setReadOnly(True)
        self.txt_session.setLineWrapMode(QTextEdit.NoWrap)
        self.txt_session.setFont(QFont("Consolas", 9))
        layout.addWidget(self.txt_session)

        self._sessions: list = []
        self.view_stack.addWidget(page)

    def refresh_sessions(self):
        """Re-read the audit log and repopulate the session picker."""
        log_path = audit_log_path(self.config)
        if not log_path.is_file():
            self.lbl_session_hint.setText(f"No audit log at {log_path}")
            return
        try:
            self._sessions = build_sessions(read_events(log_path))
        except OSError as e:
            self.lbl_session_hint.setText(f"Could not read the audit log: {e}")
            return

        # Newest first: the session you want is almost always the last one.
        self._sessions.sort(key=lambda s: s.ended_at, reverse=True)

        self.cmb_session.blockSignals(True)
        self.cmb_session.clear()
        for s in self._sessions:
            self.cmb_session.addItem(f"{len(s.turns):>4} tur · {s.session_id}")
        self.cmb_session.blockSignals(False)

        self.lbl_session_hint.setText(f"{len(self._sessions)} sessions in the log.")
        if self._sessions:
            self.cmb_session.setCurrentIndex(0)
            self.show_selected_session()
        else:
            self.txt_session.clear()

    def show_selected_session(self):
        index = self.cmb_session.currentIndex()
        if 0 <= index < len(self._sessions):
            # Plain text, not HTML: the timeline contains tool arguments and
            # raw model markup, and setPlainText renders them verbatim.
            self.txt_session.setPlainText(format_session(self._sessions[index]))

    def browse_memory_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if directory:
            self.txt_memory_dir.setText(directory)
            self.refresh_memory_table()

    def refresh_memory_table(self):
        project = self.txt_memory_dir.text().strip()
        self.memory_table.setRowCount(0)
        if not project:
            self.lbl_memory_count.setText("No project loaded.")
            return
        db_path = memory_admin.resolve_db(project)
        # describe_database also answers "is this schema readable at all" —
        # a question this tab never asked, so a database in a newer schema
        # showed up as an empty one (R-2026-01).
        message, rows = memory_admin.describe_database(db_path)
        self.memory_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            id_item = QTableWidgetItem(str(r.id))
            id_item.setData(Qt.UserRole, r.id)
            self.memory_table.setItem(i, 0, id_item)
            self.memory_table.setItem(i, 1, QTableWidgetItem(r.category))
            self.memory_table.setItem(i, 2, QTableWidgetItem((r.created_at or "").split(".")[0]))
            self.memory_table.setItem(i, 3, QTableWidgetItem(r.text))
        self.lbl_memory_count.setText(message)

    def _selected_memory_id(self) -> int | None:
        row = self.memory_table.currentRow()
        if row < 0:
            return None
        item = self.memory_table.item(row, 0)
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def delete_selected_memory(self):
        project = self.txt_memory_dir.text().strip()
        mem_id = self._selected_memory_id()
        if not project or mem_id is None:
            QMessageBox.warning(self, "Warning", "Select a lesson row to delete first.")
            return
        confirm = QMessageBox.question(
            self, "Delete lesson",
            f"Permanently delete lesson id={mem_id} from semantic memory?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return
        if memory_admin.delete_memory(memory_admin.resolve_db(project), mem_id):
            self.refresh_memory_table()
        else:
            QMessageBox.critical(self, "Error", f"Could not delete lesson id={mem_id}.")

    def export_memory(self):
        project = self.txt_memory_dir.text().strip()
        if not project:
            QMessageBox.warning(self, "Warning", "Load a project first.")
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "Export semantic memory", "memory-export.json", "JSON (*.json)")
        if not out:
            return
        try:
            n = memory_admin.export_memories(memory_admin.resolve_db(project), out)
            QMessageBox.information(self, "Export complete", f"Exported {n} lesson(s) to {out}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")

    def load_config_to_ui(self):
        """Read config settings to form components."""
        self._populate_presets_ui()
        settings = load_settings_from_file()

        self.txt_upstream.setText(settings.get("upstream", self.config.upstream))
        self.txt_port.setText(str(settings.get("port", self.config.port)))
        self.txt_reserve.setText(str(settings.get("context_reserve", self.config.context_reserve)))
        self.txt_spill.setText(str(settings.get("spill_max_inline_bytes", self.config.spill_max_inline_bytes)))
        self.txt_fallback_n_ctx.setText(str(settings.get("fallback_n_ctx", self.config.fallback_n_ctx)))
        self.txt_queue_timeout.setText(str(settings.get("semaphore_timeout", self.config.semaphore_timeout)))
        self.txt_stream_duration.setText(str(settings.get("stream_max_duration", self.config.stream_max_duration)))
        aliases = settings.get("model_aliases", self.config.model_aliases)
        self.txt_aliases.setPlainText(json.dumps(aliases, indent=2))

        # Top-level `models` is the active configuration and what the gateway
        # actually reads. It is ABSENT in every settings.json written before
        # this existed, so the active preset is the fallback -- otherwise the
        # dashboard opens with an empty table on exactly the machine the
        # feature was built for, and the user concludes it did not ship.
        self._models = active_models()
        self._refresh_models_table()

        self._sync_lodestar_ui(settings.get("lodestar_enabled", True))
        self.chk_autostart.setChecked(get_startup_shortcut_path().is_file())

        self.lbl_port_indicator.setText(f"PORT: {self.txt_port.text()}")

    def _populate_presets_ui(self, target_preset: str | None = None):
        """Populate the preset combobox from settings.json."""
        self.cmb_presets.blockSignals(True)
        self.cmb_presets.clear()
        presets = get_presets()
        active_name = target_preset or get_active_preset_name()

        for name in presets.keys():
            self.cmb_presets.addItem(name)

        idx = self.cmb_presets.findText(active_name)
        if idx >= 0:
            self.cmb_presets.setCurrentIndex(idx)
        elif self.cmb_presets.count() > 0:
            self.cmb_presets.setCurrentIndex(0)
        self.cmb_presets.blockSignals(False)

    def _on_preset_selected(self, index: int):
        """Handle preset selection from the dropdown."""
        if index < 0:
            return
        preset_name = self.cmb_presets.currentText()
        presets = get_presets()
        if preset_name not in presets:
            return
        preset = presets[preset_name]
        self._load_preset_values_to_ui(preset)

    def _load_preset_values_to_ui(self, data: dict):
        """Fill form inputs from a preset dict."""
        self.txt_upstream.setText(str(data.get("upstream", "")))
        self.txt_port.setText(str(data.get("port", 18000)))
        self.txt_reserve.setText(str(data.get("context_reserve", 16384)))
        self.txt_spill.setText(str(data.get("spill_max_inline_bytes", 8192)))
        self.txt_fallback_n_ctx.setText(str(data.get("fallback_n_ctx", 32768)))
        self.txt_queue_timeout.setText(str(data.get("semaphore_timeout", 7200.0)))
        self.txt_stream_duration.setText(str(data.get("stream_max_duration", 7200)))
        aliases = data.get("model_aliases", {})
        self.txt_aliases.setPlainText(json.dumps(aliases, indent=2))
        self._models = [dict(m) for m in (data.get("models") or [])
                        if isinstance(m, dict)]
        self._refresh_models_table()
        self._sync_lodestar_ui(bool(data.get("lodestar_enabled", True)))
        self.lbl_port_indicator.setText(f"PORT: {self.txt_port.text()}")

    def _on_save_as_preset(self):
        """Prompt the user for a new preset name and save current form fields."""
        current_name = self.cmb_presets.currentText()
        name, ok = QInputDialog.getText(
            self, "Save Preset", "Enter a name for this configuration preset:",
            QLineEdit.Normal, current_name
        )
        if not ok or not name or not name.strip():
            return
        clean_name = name.strip()

        try:
            aliases_str = self.txt_aliases.toPlainText().strip()
            aliases = json.loads(aliases_str) if aliases_str else {}
            if not isinstance(aliases, dict):
                raise ValueError("Model aliases must be a valid JSON object.")

            preset_data = {
                "upstream": self.txt_upstream.text().strip(),
                "port": int(self.txt_port.text().strip()),
                "context_reserve": int(self.txt_reserve.text().strip()),
                "spill_max_inline_bytes": int(self.txt_spill.text().strip() or "8192"),
                "fallback_n_ctx": int(self.txt_fallback_n_ctx.text().strip() or "32768"),
                "semaphore_timeout": float(self.txt_queue_timeout.text().strip() or "7200.0"),
                "stream_max_duration": int(self.txt_stream_duration.text().strip() or "7200"),
                "model_aliases": aliases,
                "lodestar_enabled": self.chk_lodestar.isChecked(),
                "models": [dict(m) for m in self._models],
            }
            if save_preset(clean_name, preset_data, set_active=True):
                self._populate_presets_ui(target_preset=clean_name)
                QMessageBox.information(self, "Preset Saved", f"Preset '{clean_name}' has been saved successfully.")
            else:
                QMessageBox.warning(self, "Error", f"Failed to save preset '{clean_name}'.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid configuration values: {e}")

    def _on_delete_preset(self):
        """Delete currently selected preset with confirmation."""
        preset_name = self.cmb_presets.currentText()
        presets = get_presets()
        if len(presets) <= 1:
            QMessageBox.warning(self, "Cannot Delete", "At least one preset must remain in configuration.")
            return

        confirm = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete preset '{preset_name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return

        if delete_preset(preset_name):
            self._populate_presets_ui()
            active = get_active_preset_name()
            presets = get_presets()
            if active in presets:
                self._load_preset_values_to_ui(presets[active])
            QMessageBox.information(self, "Preset Deleted", f"Preset '{preset_name}' was removed.")
        else:
            QMessageBox.warning(self, "Error", f"Could not delete preset '{preset_name}'.")

    # -- Models ------------------------------------------------------------

    def _refresh_models_table(self):
        """Redraw the table from `self._models`.

        `self._models` is the raw list of dicts, kept rather than a
        ModelRegistry, because the table is an EDITOR: a spec that does not
        parse still has to be visible and fixable, and a registry drops it.
        The registry is built alongside, only to read back what parsed.
        """
        rows = list(getattr(self, "_models", []) or [])
        registry = ModelRegistry.from_list(rows)
        by_id = {s.id: s for s in registry}

        self.tbl_models.setRowCount(len(rows))
        for row, raw in enumerate(rows):
            model_id = str(raw.get("id") or "?")
            spec = by_id.get(model_id)
            if spec is None:
                cells = [model_id, str(raw.get("engine") or "?"), "-", "-",
                         "invalid — see Model Settings"]
            else:
                missing = spec.missing_files(self.config.polystrata_home())
                cells = [
                    spec.id,
                    spec.engine,
                    f"{spec.n_ctx:,}" if spec.n_ctx else "-",
                    "yes" if spec.vision else "",
                    "not installed" if missing else "ready",
                ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if spec is not None and col == 4 and missing:
                    item.setToolTip("Missing: " + ", ".join(missing))
                self.tbl_models.setItem(row, col, item)

        for err in registry.errors:
            _log.warning("Model registry: %s", err)

    def _inherited_limits(self) -> dict:
        """What a model that overrides nothing would get, from the form.

        Read off the FORM rather than from settings.json, so an edit the user
        has made but not yet saved is what the placeholders describe. A
        placeholder that disagrees with the box above it is worse than none.
        """
        def _int(widget, fallback):
            try:
                return int(widget.text().strip())
            except (ValueError, AttributeError):
                return fallback

        return {
            "context_reserve": _int(self.txt_reserve, 16384),
            "spill_max_inline_bytes": _int(self.txt_spill, 65536),
            "max_n_predict": getattr(self.config, "max_n_predict", 16384),
        }

    def _selected_model_row(self) -> int:
        rows = self.tbl_models.selectionModel().selectedRows() \
            if self.tbl_models.selectionModel() else []
        if rows:
            return rows[0].row()
        return self.tbl_models.currentRow()

    def _on_add_model(self):
        dialog = ModelDialog(self, None,
                             default_upstream=self.txt_upstream.text().strip(),
                             inherited=self._inherited_limits())
        if dialog.exec() != QDialog.Accepted:
            return
        data = dialog.to_dict()
        if any(str(m.get("id", "")).lower() == data["id"].lower()
               for m in self._models):
            QMessageBox.warning(self, "Already there",
                                f"A model called '{data['id']}' is already in "
                                "this preset.")
            return
        self._models.append(data)
        self._refresh_models_table()

    def _on_edit_model(self):
        row = self._selected_model_row()
        if row < 0 or row >= len(self._models):
            QMessageBox.information(self, "No model selected",
                                    "Select a model in the table first.")
            return
        dialog = ModelDialog(self, self._models[row],
                             default_upstream=self.txt_upstream.text().strip(),
                             inherited=self._inherited_limits())
        if dialog.exec() != QDialog.Accepted:
            return
        self._models[row] = dialog.to_dict()
        self._refresh_models_table()

    def _on_remove_model(self):
        row = self._selected_model_row()
        if row < 0 or row >= len(self._models):
            return
        model_id = str(self._models[row].get("id", "?"))
        confirm = QMessageBox.question(
            self, "Remove model",
            f"Remove '{model_id}' from this preset?\n\nNothing is deleted from "
            "disk — only this preset stops offering the model.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return
        del self._models[row]
        self._refresh_models_table()

    def _on_export_opencode(self):
        """Write this preset's models into OpenCode's config.

        Asks first, and the question carries the two things that are easy to
        get wrong: which models would DISAPPEAR from the client, and how many
        comments a JSONC round trip would delete. Both are irreversible from
        the user's point of view, and both are invisible in a success message.
        """
        registry = ModelRegistry.from_list(self._models)
        if not len(registry):
            QMessageBox.information(self, "Nothing to export",
                                    "This preset declares no models.")
            return

        plan = opencode_export.plan_export(registry.specs)
        if not plan.ok:
            QMessageBox.warning(self, "Cannot export", plan.error)
            return

        lines = [f"Write {len(plan.models)} model(s) into:", str(plan.path), ""]
        lines.append("Models: " + ", ".join(plan.models))
        if plan.removed:
            lines += ["", "These will be REMOVED from OpenCode (they are not in "
                          "this preset):", "  " + ", ".join(plan.removed)]
        if plan.comments:
            lines += ["", f"WARNING: the file has {plan.comments} comment(s). "
                          "Writing JSON drops them. A .bak is kept."]
        lines += ["", "Everything else in the file — other providers, MCP "
                      "servers, plugins — is preserved."]

        confirm = QMessageBox.question(
            self, "Export to OpenCode", "\n".join(lines),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return

        try:
            port = int(self.txt_port.text().strip() or "18000")
        except ValueError:
            port = 18000
        result = opencode_export.write_export(registry.specs, port, path=plan.path)
        if not result.ok:
            QMessageBox.critical(self, "Export failed", result.error)
            return
        QMessageBox.information(
            self, "Exported",
            f"Wrote {len(result.models)} model(s) to {result.path}.\n"
            f"Backup: {result.backup}\n\nRestart OpenCode to pick them up.")

    def _sync_lodestar_ui(self, enabled: bool):
        """Reflect the LODESTAR toggle across checkbox, sidebar badge and tray.

        Single source of truth: every surface is updated here with signals
        blocked, so toggling in one place (settings save or tray menu) keeps
        the others consistent without feedback loops.
        """
        self.chk_lodestar.blockSignals(True)
        self.chk_lodestar.setChecked(enabled)
        self.chk_lodestar.blockSignals(False)

        if enabled:
            self.lbl_lodestar_badge.setText("LODESTAR: on")
            self.lbl_lodestar_badge.setObjectName("lodestar-badge-on")
        else:
            self.lbl_lodestar_badge.setText("LODESTAR: OFF")
            self.lbl_lodestar_badge.setObjectName("lodestar-badge-off")
        # Re-polish so the objectName-based stylesheet re-applies
        self.lbl_lodestar_badge.style().unpolish(self.lbl_lodestar_badge)
        self.lbl_lodestar_badge.style().polish(self.lbl_lodestar_badge)

        if self.tray_lodestar_action is not None:
            self.tray_lodestar_action.blockSignals(True)
            self.tray_lodestar_action.setChecked(enabled)
            self.tray_lodestar_action.blockSignals(False)
        self._update_tray_tooltip()

    def set_lodestar_enabled(self, enabled: bool):
        """Persist the LODESTAR toggle, sync every surface and restart."""
        settings = load_settings_from_file()
        settings["lodestar_enabled"] = enabled
        if not save_settings_to_file(settings):
            # Without this the tray switch lies in the other direction: it
            # shows the new position while the restarted gateway reads the old
            # value out of settings.json (D-2026-01).
            QMessageBox.warning(
                self, "Not saved",
                "The LODESTAR switch could not be written to settings.json, so "
                "the gateway would restart with the previous setting. Nothing "
                "was changed — see gui.log.")
            self._sync_lodestar_ui(not enabled)
            return
        self._sync_lodestar_ui(enabled)
        self.restart_gateway()

    def save_config(self):
        """Validate settings and save them to settings.json."""
        try:
            upstream = self.txt_upstream.text().strip()
            port = int(self.txt_port.text().strip())
            reserve = int(self.txt_reserve.text().strip())
            spill_bytes = int(self.txt_spill.text().strip() or "8192")
            fallback_n_ctx = int(self.txt_fallback_n_ctx.text().strip() or "32768")
            queue_timeout = float(self.txt_queue_timeout.text().strip() or "7200.0")
            stream_duration = int(self.txt_stream_duration.text().strip() or "7200")
            aliases_str = self.txt_aliases.toPlainText().strip()
            lodestar_enabled = self.chk_lodestar.isChecked()
            autostart_enabled = self.chk_autostart.isChecked()

            # Parse aliases JSON
            aliases = json.loads(aliases_str) if aliases_str else {}
            if not isinstance(aliases, dict):
                raise ValueError("Model aliases must be a valid JSON object.")

            active_preset_name = self.cmb_presets.currentText().strip() or "Custom"
            presets = get_presets()
            presets[active_preset_name] = {
                "upstream": upstream,
                "port": port,
                "context_reserve": reserve,
                "spill_max_inline_bytes": spill_bytes,
                "fallback_n_ctx": fallback_n_ctx,
                "semaphore_timeout": queue_timeout,
                "stream_max_duration": stream_duration,
                "model_aliases": aliases,
                "lodestar_enabled": lodestar_enabled,
                "models": [dict(m) for m in self._models],
            }

            data = {
                "upstream": upstream,
                "port": port,
                "context_reserve": reserve,
                "spill_max_inline_bytes": spill_bytes,
                "fallback_n_ctx": fallback_n_ctx,
                "semaphore_timeout": queue_timeout,
                "stream_max_duration": stream_duration,
                "model_aliases": aliases,
                "lodestar_enabled": lodestar_enabled,
                # Into the ACTIVE settings too, not only into the preset: the
                # gateway reads top-level keys, and a preset it never looks at
                # is a list that edits nothing. The same is true of every other
                # key in this dict, which is why they are all written twice.
                "models": [dict(m) for m in self._models],
                "active_preset": active_preset_name,
                "presets": presets,
            }

            # Save file (merges with existing settings.json). Both results are
            # READ: showing "saved" for a write that failed sends the gateway
            # back up with the previous upstream and hides why (D-2026-01).
            settings_saved = save_settings_to_file(data)

            # Configure Autostart — this already returned a bool; nothing used it.
            autostart_ok = set_autostart(autostart_enabled)

            # Keep the LODESTAR badge and tray toggle in sync
            self._sync_lodestar_ui(lodestar_enabled)
            self.lbl_port_indicator.setText(f"PORT: {port}")

            # Restart subprocess to apply changes
            self.restart_gateway()

            level, message = describe_save_outcome(
                settings_saved=settings_saved,
                autostart_ok=autostart_ok,
                autostart_requested=autostart_enabled,
            )
            if level == "success":
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.warning(self, "Saved with problems", message)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")

    def browse_target_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Target Workspace Directory")
        if directory:
            self.txt_target_dir.setText(directory)

    def deploy_template(self):
        target = self.txt_target_dir.text().strip()
        if not target:
            QMessageBox.warning(self, "Warning", "Please specify a target directory.")
            return

        result = deploy_lodestar_template(target)
        if "Successfully" in result:
            QMessageBox.information(self, "Success", result)
        else:
            QMessageBox.critical(self, "Error", result)

    def start_gateway(self):
        """Start the gateway subprocess."""
        self.lbl_status_text.setText("Starting...")
        self.lbl_status_text.setStyleSheet("color: #ffb3b3;")
        self.lbl_status_indicator.setPixmap(create_status_pixmap("#ffb3b3"))
        self.controller.start()

    def stop_gateway(self):
        """Stop the gateway subprocess."""
        self.lbl_status_text.setText("Stopping...")
        self.lbl_status_text.setStyleSheet("color: #ffb3b3;")
        self.lbl_status_indicator.setPixmap(create_status_pixmap("#ffb3b3"))
        self.controller.stop()

    def restart_gateway(self):
        """Restart the gateway subprocess."""
        self.lbl_status_text.setText("Restarting...")
        self.lbl_status_text.setStyleSheet("color: #ffb3b3;")
        self.lbl_status_indicator.setPixmap(create_status_pixmap("#ffb3b3"))
        self.controller.restart()

    @Slot(str)
    def append_log(self, text: str):
        """Append log line to the live view; error-like lines also go to the
        persistent error buffer behind the "⚠ Errors" toggle."""
        color, is_error = classify_log_line(text)

        # Gateway logs are full of XML/markup (<think>, <tool_call>, <|...|>);
        # QTextEdit.append interprets HTML, so unescaped lines get swallowed
        # or render broken. Escape the payload, style via the wrapper only.
        safe = html.escape(text)
        line = f'<span style="color: {color};">{safe}</span>'

        self._live_buffer.append(line)
        if is_error:
            self._error_buffer.append(line)
            self._refresh_error_button()

        if not self._errors_only or is_error:
            self.log_text.append(line)
            self.log_text.ensureCursorVisible()

    def _refresh_error_button(self):
        n = len(self._error_buffer)
        label = f"⚠ Errors ({n})"
        if self.btn_errors.text() != label:
            self.btn_errors.setText(label)
        # Red accent as soon as there is anything worth looking at
        if n and not self.btn_errors.property("alerting"):
            self.btn_errors.setProperty("alerting", True)
            self.btn_errors.setStyleSheet(
                "color: #ff1744; border: 1px solid #ff1744; border-radius: 6px; "
                "padding: 8px 16px; background-color: transparent;")

    def _set_errors_only(self, checked: bool):
        """Switch the log view between the live stream and the error buffer."""
        self._errors_only = checked
        self.log_text.clear()
        source = self._error_buffer if checked else self._live_buffer
        for line in source:
            self.log_text.append(line)
        self.log_text.ensureCursorVisible()

    def clear_logs(self):
        """Clear the view and BOTH buffers (explicit user action)."""
        self.log_text.clear()
        self._live_buffer.clear()
        self._error_buffer.clear()
        self.btn_errors.setText("⚠ Errors (0)")
        self.btn_errors.setProperty("alerting", False)
        self.btn_errors.setStyleSheet("")

    def _current_port(self) -> int:
        try:
            return int(self.txt_port.text().strip())
        except (ValueError, AttributeError):
            return self.config.port

    def _poll_health(self):
        if self.controller.is_running():
            self.health_poller.poll(self._current_port())

    @Slot(dict)
    def on_health_received(self, data: dict):
        """Update the upstream chip and tray tooltip from a /health response."""
        if not self.controller.is_running():
            return
        upstream = (data or {}).get("upstream") or {}
        if not upstream and (data or {}).get("reason"):
            upstream = data  # Poll failure payload from describe_health_failure

        label, color, tip = format_upstream_chip(upstream)
        self.lbl_upstream.setText(label)
        self.lbl_upstream.setToolTip(tip)
        self.lbl_upstream.setStyleSheet(
            f"color: {color}; font-family: monospace; font-size: 11px; "
            f"background-color: #201f20; padding: 2px 6px; border-radius: 4px;"
        )
        self._upstream_ok = upstream.get("status") in ("ok", "sleeping")
        self._update_tray_tooltip()

    def _update_tray_tooltip(self):
        if self.tray_icon is None:
            return
        if self._is_running:
            pid = self.controller.process.pid if self.controller.process else "?"
            state = f"Running (PID {pid})"
            if self._upstream_ok is False:
                state += " · upstream UNREACHABLE"
        else:
            state = "Offline"
        lodestar = "on" if self.chk_lodestar.isChecked() else "OFF"
        self.tray_icon.setToolTip(
            f"FastStrata Gateway — {state} · LODESTAR {lodestar}")

    @Slot(bool)
    def on_subprocess_status_changed(self, is_running: bool):
        self._is_running = is_running
        if is_running:
            self.lbl_status_text.setText("Running")
            self.lbl_status_text.setStyleSheet("color: #00e676;")
            self.lbl_status_indicator.setPixmap(create_status_pixmap("#00e676"))
            
            self.lbl_sidebar_status.setText("Status: Online")
            self.lbl_sidebar_status.setStyleSheet("color: #00e676; font-weight: bold; font-size: 11px;")
            
            if self.controller.process:
                self.lbl_pid.setText(f"PID: {self.controller.process.pid}")
            
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
        else:
            self.lbl_status_text.setText("Offline")
            self.lbl_status_text.setStyleSheet("color: #ff1744;")
            self.lbl_status_indicator.setPixmap(create_status_pixmap("#ff1744"))
            
            self.lbl_sidebar_status.setText("Status: Offline")
            self.lbl_sidebar_status.setStyleSheet("color: #ff1744; font-weight: bold; font-size: 11px;")
            
            self.lbl_pid.setText("PID: --")
            self._upstream_ok = None
            self.lbl_upstream.setText("UPSTREAM: --")
            self.lbl_upstream.setStyleSheet(
                "color: #cac3d8; font-family: monospace; font-size: 11px; "
                "background-color: #201f20; padding: 2px 6px; border-radius: 4px;")

            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
        self._update_tray_tooltip()
        if is_running:
            # First health reading right away instead of waiting 15 s
            QTimer.singleShot(2000, self._poll_health)

    def open_log_in_notepad(self):
        """Open the active log file in the OS default editor (cross-platform)."""
        root = Path(__file__).resolve().parent.parent
        log_path = root / "gateway.log"
        if not log_path.is_file():
            log_path = root / "proxy.log"

        if not log_path.is_file():
            QMessageBox.warning(self, "Error", f"Log file not found at {log_path}")
            return
        try:
            if os.name == "nt":
                os.startfile(str(log_path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(log_path)])
            else:
                subprocess.Popen(["xdg-open", str(log_path)])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open log file: {e}")

    def closeEvent(self, event):
        """Close event intercepts closing window to minimize to tray instead."""
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.hide()
            event.ignore()
        else:
            self.controller.stop()
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # Its own name: Local-LLM-Gateway's app used the same one, so with that app running the shortcut found it,
    # asked IT to show itself and quit.
    server_name = "faststrata_gateway_gui_single_instance"
    socket = QLocalSocket()
    socket.connectToServer(server_name)
    if socket.waitForConnected(500):
        # Already running: tell existing instance to activate and exit
        socket.write(b"show\n")
        socket.waitForBytesWritten(1000)
        socket.disconnectFromServer()
        sys.exit(0)

    # Primary instance: create local server to receive activation signals
    local_server = QLocalServer()
    local_server.removeServer(server_name)
    local_server.listen(server_name)

    # Initialize subprocess controller
    controller = SubprocessController()
    
    # Initialize main window
    window = MainWindow(controller)
    window.show()

    # System Tray Icon Setup
    tray_icon = QSystemTrayIcon(app)
    
    # Use programmatically drawn icon matching Obsidian Flux
    active_pixmap = create_status_pixmap("#00e676")
    tray_icon.setIcon(QIcon(active_pixmap))
    tray_icon.setToolTip("FastStrata Gateway")

    # Tray Menu
    menu = QMenu()

    def show_dashboard():
        # Ensure window is visible (in case it was hidden to tray)
        window.show()
        window.showNormal()
        window.raise_()
        window.activateWindow()
        if os.name == "nt":
            try:
                import ctypes
                hwnd = int(window.winId())
                ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                ctypes.windll.user32.SetForegroundWindow(hwnd)
            except Exception:
                pass

    def on_single_instance_connection():
        client = local_server.nextPendingConnection()
        if client:
            show_dashboard()
            client.disconnectFromServer()

    local_server.newConnection.connect(on_single_instance_connection)

    action_open = menu.addAction("🎛️ Open Dashboard")
    action_open.triggered.connect(show_dashboard)

    action_restart = menu.addAction("🔄 Restart Gateway")
    action_restart.triggered.connect(window.restart_gateway)

    def tray_deploy_template():
        directory = QFileDialog.getExistingDirectory(
            None, "Deploy LODESTAR template to project directory")
        if not directory:
            return
        result = deploy_lodestar_template(directory)
        if "Successfully" in result:
            QMessageBox.information(None, "LODESTAR template", result)
        else:
            QMessageBox.critical(None, "LODESTAR template", result)

    action_deploy = menu.addAction("⚡ Deploy LODESTAR Template...")
    action_deploy.triggered.connect(tray_deploy_template)

    menu.addSeparator()

    # Dynamic checkable LODESTAR protocol toggle. The window owns the state
    # (set_lodestar_enabled persists + syncs every surface + restarts), so
    # the tray action is registered on the window and merely forwards clicks.
    action_lodestar = menu.addAction("🧠 LODESTAR Protocol")
    action_lodestar.setCheckable(True)
    settings = load_settings_from_file()
    action_lodestar.setChecked(settings.get("lodestar_enabled", True))
    window.tray_lodestar_action = action_lodestar
    action_lodestar.triggered.connect(
        lambda checked: window.set_lodestar_enabled(checked))

    # Autostart checkable toggle
    action_autostart = menu.addAction("🚀 Start on Windows Boot")
    action_autostart.setCheckable(True)
    action_autostart.setChecked(get_startup_shortcut_path().is_file())
    
    def toggle_autostart():
        enabled = action_autostart.isChecked()
        set_autostart(enabled)
        window.load_config_to_ui()
    action_autostart.triggered.connect(toggle_autostart)

    menu.addSeparator()

    action_exit = menu.addAction("❌ Exit")
    
    def on_exit():
        local_server.close()
        controller.stop()
        app.quit()
    action_exit.triggered.connect(on_exit)

    tray_icon.setContextMenu(menu)
    tray_icon.show()

    # Register the tray icon on the window for live tooltip updates
    window.tray_icon = tray_icon
    window._update_tray_tooltip()

    # Double click on tray icon restores main window
    def on_tray_activated(reason):
        if reason == QSystemTrayIcon.DoubleClick:
            show_dashboard()
    tray_icon.activated.connect(on_tray_activated)

    # Status icon listener
    def update_tray_icon(is_running):
        color = "#00e676" if is_running else "#ff1744"
        pix = create_status_pixmap(color)
        tray_icon.setIcon(QIcon(pix))
    controller.status_changed.connect(update_tray_icon)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
