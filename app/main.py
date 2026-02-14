# helper_tray.py
from __future__ import annotations

import json
import os
import sys
import subprocess
import threading
import tempfile
import time
import re
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests
import mss
import pystray
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import scrolledtext

# Local audio playback (MP3) on Windows.
# If this import fails, we will fall back to opening in the browser.
try:
    from playsound import playsound  # type: ignore
except Exception:
    playsound = None  # type: ignore

try:
    import pygetwindow as gw
except Exception:
    gw = None  # type: ignore

# Global hotkeys (no admin required in most cases)
try:
    from pynput import keyboard as pynput_keyboard  # type: ignore
except Exception:
    pynput_keyboard = None  # type: ignore


APP_NAME = "TradeTalkerAI™"
DEFAULT_WINDOW_QUERY = "Webull"
DEFAULT_DELAY_MS = 1000
DEFAULT_MODE = "brief"
DEFAULT_VOICE = 1
DEFAULT_VOICE_SPEED = 1.0  # 0.5 (slow) .. 1.5 (fast)
DEFAULT_SAVE_CHART = False
# Default: play audio directly on the PC (no browser).
DEFAULT_OPEN_AUDIO = False
DEFAULT_PLAY_LOCAL = True
DEFAULT_SAVE_IMAGES_LOCAL = True  # Save captured frames to ./captures

# Default hotkeys
DEFAULT_HK_FULL = "F8"
DEFAULT_HK_BRIEF = "F9"
DEFAULT_HK_MOMO = "F10"


def _normalize_hotkey_spec(spec: str | None, fallback: str) -> str:
    s = (spec or "").strip()
    return s if s else fallback


def _parse_hotkey_spec(spec: str) -> tuple[set[str], str]:
    """Parses a hotkey like 'F8' or 'Ctrl+Shift+F8' into (mods, keyname)."""
    parts = [p.strip().lower() for p in spec.replace(" ", "").split("+") if p.strip()]
    mods: set[str] = set()
    keyname = ""
    for p in parts:
        if p in ("ctrl", "control"):
            mods.add("ctrl")
        elif p in ("alt", "option"):
            mods.add("alt")
        elif p == "shift":
            mods.add("shift")
        else:
            keyname = p
    if not keyname:
        raise ValueError("Missing key")
    # Normalize F-keys
    if keyname.startswith("f") and keyname[1:].isdigit():
        n = int(keyname[1:])
        if not (1 <= n <= 24):
            raise ValueError("F-key out of range")
        keyname = f"f{n}"
    return mods, keyname


def _pynput_key_from_name(keyname: str):
    if pynput_keyboard is None:
        return None
    # F-keys
    if keyname.startswith("f") and keyname[1:].isdigit():
        return getattr(pynput_keyboard.Key, keyname, None)
    # Single character key
    if len(keyname) == 1:
        return pynput_keyboard.KeyCode.from_char(keyname)
    # Common named keys
    mapping = {
        "space": pynput_keyboard.Key.space,
        "enter": pynput_keyboard.Key.enter,
        "tab": pynput_keyboard.Key.tab,
        "esc": pynput_keyboard.Key.esc,
        "escape": pynput_keyboard.Key.esc,
    }
    return mapping.get(keyname)

VOICE_MAP = {
    1: "oscar",
    2: "alan",
    3: "victor",
    4: "edward",
    5: "nancy",
    6: "marla",
}
MODE_OPTIONS = ["brief", "momentum", "full"]

BASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / "settings.json"
CAPTURES_DIR = BASE_DIR / "captures"
STARTUP_CHIME_PATH = BASE_DIR / "openbell.mp3"
ANALYZING_FULL_PATH = BASE_DIR / "analizing_full.mp3"
ANALYZING_BRIEF_PATH = BASE_DIR / "analizing_brief.mp3"
ANALYZING_MOMO_PATH = BASE_DIR / "analizing_momentum.mp3"
WAITING_PATH = BASE_DIR / "waiting.mp3"

# Persistent device identifier (stored next to the app)
DEVICE_ID_FILE = BASE_DIR / "device_id.txt"


APP_LOGO_PATH = BASE_DIR / "TradeTalkerAI.ico"


@dataclass
class LogEntry:
    ts: float
    mode: str
    transcript: str

    def format_line(self) -> str:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts))
        mode = (self.mode or "").strip()
        return f"[{t}] ({mode}) {self.transcript}".strip()


@dataclass
class Settings:
    window_query: str = DEFAULT_WINDOW_QUERY
    window_title: str = ""
    delay_ms: int = DEFAULT_DELAY_MS
    mode: str = DEFAULT_MODE
    voice: int = DEFAULT_VOICE
    voice_speed: float = DEFAULT_VOICE_SPEED
    save_chart: bool = DEFAULT_SAVE_CHART
    save_images_local: bool = DEFAULT_SAVE_IMAGES_LOCAL
    open_audio: bool = DEFAULT_OPEN_AUDIO
    play_local_audio: bool = DEFAULT_PLAY_LOCAL
    api_url: str = "https://tradetalkerai-api-production.up.railway.app"

    # Subscriber/license id (sent with each request)
    subscriber_id: str = ""

    # Global hotkeys (defaults: F8/F9/F10)
    hk_full: str = DEFAULT_HK_FULL
    hk_brief: str = DEFAULT_HK_BRIEF
    hk_momo: str = DEFAULT_HK_MOMO

    def normalized_api_url(self) -> str:
        return (self.api_url or "").strip().rstrip("/")


def load_settings() -> Settings:
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            s = Settings(**{k: data.get(k) for k in Settings.__dataclass_fields__.keys() if k in data})  # type: ignore
            if s.mode not in MODE_OPTIONS:
                s.mode = DEFAULT_MODE
            if s.voice not in VOICE_MAP:
                s.voice = DEFAULT_VOICE
            if not isinstance(s.delay_ms, int) or s.delay_ms < 0:
                s.delay_ms = DEFAULT_DELAY_MS
            try:
                vs = float(s.voice_speed)
            except Exception:
                vs = DEFAULT_VOICE_SPEED
            if vs < 0.5: vs = 0.5
            if vs > 1.5: vs = 1.5
            s.voice_speed = vs
            return s
        except Exception:
            pass
    return Settings()


def save_settings(s: Settings) -> None:
    SETTINGS_PATH.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")


def ensure_captures_dir() -> Path:
    try:
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return CAPTURES_DIR


def open_folder(path: Path) -> None:
    """Open a folder in the OS file explorer."""
    try:
        ensure_captures_dir()
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def delete_all_captures_dialog(parent: tk.Misc | None = None) -> None:
    """Confirm and delete all files inside the captures folder."""
    try:
        ensure_captures_dir()
        # Confirm
        folder = CAPTURES_DIR
        if not folder.exists():
            messagebox.showinfo("Delete Screen Captures", "No captures folder was found.", parent=parent)
            return

        files = [p for p in folder.glob("*") if p.is_file()]
        if not files:
            messagebox.showinfo("Delete Screen Captures", "No screen captures to delete.", parent=parent)
            return

        ok = messagebox.askyesno(
            "Delete Screen Captures",
            f"Delete all screen captures and messages in:\n{folder}\n\nThis cannot be undone.",
            parent=parent,
        )
        if not ok:
            return

        deleted = 0
        for p in files:
            try:
                p.unlink()
                deleted += 1
            except Exception:
                pass

        messagebox.showinfo("Delete Screen Captures", f"Deleted {deleted} file(s).", parent=parent)
    except Exception:
        pass


def save_captures(img1: Image.Image, img2: Optional[Image.Image], mode: str) -> list[Path]:
    """Persist captured frames to ./captures and return written paths."""
    out_dir = ensure_captures_dir()
    # Include milliseconds to avoid collisions when hotkeys are pressed quickly.
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    ms = int((time.time() * 1000) % 1000)
    safe_mode = re.sub(r"[^A-Za-z0-9_-]+", "", (mode or "mode").strip().lower()) or "mode"
    base = f"{ts}_{ms:03d}_{safe_mode}"
    paths: list[Path] = []

    p1 = out_dir / f"{base}_1.png"
    img1.save(p1, format="PNG")
    paths.append(p1)

    if img2 is not None:
        p2 = out_dir / f"{base}_2.png"
        img2.save(p2, format="PNG")
        paths.append(p2)

    return paths


def message_txt_path_from_image_path(img_path: Path) -> Path:
    """Given a capture image path, return a per-capture-set message file path.

    Example:
      20260210_193520_094_brief_2.png -> 20260210_193520_094_message.txt

    Note: this intentionally drops the mode and image index so each capture-set
    (timestamp millisecond prefix) gets exactly one message file.
    """
    name = img_path.name
    m = re.match(r"^(\d{8}_\d{6}_\d{3})_", name)
    if m:
        prefix = m.group(1)
    else:
        # Fallback: first 3 underscore-delimited chunks (YYYYMMDD_HHMMSS_mmm)
        parts = img_path.stem.split("_")
        prefix = "_".join(parts[:3]) if len(parts) >= 3 else img_path.stem
    return img_path.with_name(f"{prefix}_message.txt")


def write_message_log(msg_path: Path, line: str) -> None:
    """Write a single message line to msg_path (UTF-8)."""
    msg_path.parent.mkdir(parents=True, exist_ok=True)
    msg_path.write_text((line or "").rstrip() + "\n", encoding="utf-8")



def get_or_create_device_id() -> str:
    """Return a stable per-device id for this installation.

    Stored in ./device_id.txt next to the app so portable ZIP installs remain stable.
    """
    try:
        if DEVICE_ID_FILE.exists():
            v = DEVICE_ID_FILE.read_text(encoding="utf-8").strip()
            if v:
                return v
    except Exception:
        pass
    v = str(uuid.uuid4())
    try:
        DEVICE_ID_FILE.write_text(v, encoding="utf-8")
    except Exception:
        pass
    return v



def list_open_windows() -> list[str]:
    if gw is None:
        return []
    titles: list[str] = []
    for t in gw.getAllTitles():
        t = (t or "").strip()
        if t:
            titles.append(t)
    seen = set()
    out: list[str] = []
    for t in titles:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def find_window_by_query(query: str) -> Optional[str]:
    q = (query or "").strip().lower()
    if not q or gw is None:
        return None
    for t in list_open_windows():
        if q in t.lower():
            return t
    return None


def capture_window(title: str) -> Optional[Image.Image]:
    if gw is None:
        return None
    try:
        w = gw.getWindowsWithTitle(title)[0]
    except Exception:
        return None

    try:
        if getattr(w, "isMinimized", False):
            w.restore()
            time.sleep(0.15)
    except Exception:
        pass

    left, top, right, bottom = w.left, w.top, w.right, w.bottom
    width, height = max(0, right - left), max(0, bottom - top)
    if width < 50 or height < 50:
        return None

    with mss.mss() as sct:
        monitor = {"left": left, "top": top, "width": width, "height": height}
        shot = sct.grab(monitor)
        return Image.frombytes("RGB", shot.size, shot.rgb)



def post_analyze(api_url: str, img1: Image.Image, img2: Optional[Image.Image], s: Settings) -> dict:
    endpoint = f"{api_url.rstrip('/')}/v1/analyze"

    from io import BytesIO

    def to_png_bytes(im: Image.Image) -> bytes:
        bio = BytesIO()
        im.save(bio, format="PNG")
        return bio.getvalue()

    img1_bytes = to_png_bytes(img1)
    img2_bytes = to_png_bytes(img2) if img2 is not None else b""

    files = {"image": ("chart.png", img1_bytes, "image/png")}
    if img2 is not None:
        files["image2"] = ("chart2.png", img2_bytes, "image/png")

    data = {
        "mode": s.mode,
        "voice": str(int(s.voice)),
        # Speech speed for the API (snapped to 0.1 increments: 0.5 .. 1.5)
        "SPEED": f"{float(getattr(s, 'voice_speed', DEFAULT_VOICE_SPEED)):.1f}",
        "speed": f"{float(getattr(s, 'voice_speed', DEFAULT_VOICE_SPEED)):.1f}",
        "voice_speed": f"{float(getattr(s, 'voice_speed', DEFAULT_VOICE_SPEED)):.1f}",
        "save_chart": "true" if s.save_chart else "false",
        "frame_delay_ms": str(int(s.delay_ms)),
    }

    subscriber_id = (getattr(s, "subscriber_id", "") or "").strip()
    device_id = get_or_create_device_id()

    num_images = 1 + (1 if img2 is not None else 0)
    image_bytes = len(img1_bytes) + (len(img2_bytes) if img2 is not None else 0)

    headers = {
        "X-Subscriber-Id": subscriber_id,
        "X-Device-Id": device_id,
        "X-Num-Images": str(num_images),
        "X-Image-Bytes": str(image_bytes),
    }

    # Prepare the request so we can log payload size (server-side can also recompute).
    req = requests.Request("POST", endpoint, files=files, data=data, headers=headers)
    prepped = req.prepare()
    payload_bytes = 0
    try:
        if prepped.body is not None:
            payload_bytes = len(prepped.body) if isinstance(prepped.body, (bytes, bytearray)) else len(str(prepped.body).encode("utf-8"))
    except Exception:
        payload_bytes = 0
    prepped.headers["X-Payload-Bytes"] = str(payload_bytes)

    with requests.Session() as sess:
        r = sess.send(prepped, timeout=60)

    # Return JSON if present; otherwise return a minimal dict.
    if r.status_code >= 400:
        # Try to surface a helpful message to the tray app.
        try:
            j = r.json()
        except Exception:
            j = {"detail": (r.text or "").strip()}
        # Raise after capturing content.
        try:
            r.raise_for_status()
        except Exception:
            # Attach parsed JSON for the caller to display.
            j["_status_code"] = r.status_code
            return j

    try:
        return r.json()
    except Exception:
        return {"transcript": (r.text or "").strip(), "_status_code": r.status_code}


def open_in_browser(url: str) -> None:
    import webbrowser
    try:
        webbrowser.open(url)
    except Exception:
        pass


def play_audio_locally(audio_url: str) -> bool:
    """Download MP3 and play it on the local speakers.

    Returns True if playback was started, else False.
    """
    if playsound is None:
        return False
    try:
        r = requests.get(audio_url, timeout=30)
        r.raise_for_status()
        fd, path = tempfile.mkstemp(suffix=".mp3", prefix="tradetalker_")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(r.content)

        # Play in a background thread so the tray UI stays responsive.
        def _play_and_cleanup() -> None:
            try:
                playsound(path)
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass

        threading.Thread(target=_play_and_cleanup, daemon=True).start()
        return True
    except Exception:
        return False

def play_file_locally(path: str | Path) -> bool:
    """Play a local MP3 file on the local speakers (non-blocking).

    Returns True if playback was started, else False.
    """
    if playsound is None:
        return False
    try:
        p = str(path)
        if not os.path.exists(p):
            return False

        def _play() -> None:
            try:
                playsound(p)
            except Exception:
                pass

        threading.Thread(target=_play, daemon=True).start()
        return True
    except Exception:
        return False



class SettingsUI:
    def __init__(self, settings: Settings, on_save_cb, on_quit_cb=None, existing_logs: Optional[list[LogEntry]] = None):
        self.settings = settings
        self.on_save_cb = on_save_cb
        self.on_quit_cb = on_quit_cb

        self.root = tk.Tk()
        self.root.title("TradeTalkerAI™")
        self.root.geometry("780x600")
        self.root.minsize(680, 520)
        self.root.resizable(True, True)

        # ---- Top branding (logo + link ABOVE tabs, centered) ----
        top_brand = ttk.Frame(self.root)
        top_brand.pack(fill="x", pady=(8, 0))

        top_brand.columnconfigure(0, weight=1)
        top_brand.columnconfigure(1, weight=0)
        top_brand.columnconfigure(2, weight=1)

        brand_inner = ttk.Frame(top_brand)
        brand_inner.grid(row=0, column=1)

        self._logo_img = None
        try:
            if APP_LOGO_PATH.exists():
                _im = Image.open(APP_LOGO_PATH)
                _im = _im.convert("RGBA").resize((36, 36))
                self._logo_img = ImageTk.PhotoImage(_im)
                ttk.Label(brand_inner, image=self._logo_img).pack(side="left", padx=(0, 8))
        except Exception:
            pass

        site_link = tk.Label(brand_inner, text="TradeTalkerAI.com", fg="blue", cursor="hand2")
        site_link.pack(side="left")

        def _open_site_top(_event=None):
            import webbrowser
            webbrowser.open("https://tradetalkerai.com")

        site_link.bind("<Button-1>", _open_site_top)

        # Tabs: Messages + Settings
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.tab_messages = ttk.Frame(self.notebook, padding=10)
        self.tab_settings = ttk.Frame(self.notebook)
        self.tab_about = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.tab_messages, text="Messages")
        self.notebook.add(self.tab_settings, text="Settings")
        self.notebook.add(self.tab_about, text="About")

        # Messages tab
        header_row = ttk.Frame(self.tab_messages)
        header_row.pack(fill="x")
        self.txt_messages = scrolledtext.ScrolledText(self.tab_messages, height=20, wrap="word")
        self.txt_messages.pack(fill="both", expand=True, pady=(6, 10))
        self.txt_messages.configure(state="disabled")

        btns_msg = ttk.Frame(self.tab_messages)
        btns_msg.pack(fill="x")
        ttk.Button(btns_msg, text="Clear", command=self._clear_messages).pack(side="left")
        ttk.Button(btns_msg, text="Quit", command=self._quit_app).pack(side="right", padx=(0, 8))
        ttk.Button(btns_msg, text="Close", command=self.root.destroy).pack(side="right")


        # About tab
        ttk.Label(self.tab_about, text="TradeTalkerAI™", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 6))
        ttk.Label(self.tab_about, text="AI-powered chart analysis assistant", foreground="#555").pack(anchor="w", pady=(0, 12))

        about_text = (
            "TradeTalkerAI™ is an AI-powered market analysis assistant designed to help traders evaluate charts, price action, "
            "and momentum through automated visual capture and spoken insights.\n\n"
            "Disclaimer: TradeTalkerAI™ does not provide financial advice. Outputs are generated by artificial intelligence and "
            "are for informational and educational purposes only. Trading involves risk; you are solely responsible for your decisions.\n\n"
            "This software, including its design, interface, source code, audio, and branding assets, is protected by United States "
            "and international copyright laws.\n\n"
            "Developed by Kevin Dooley\n"
            "Powered by AI vision + speech technologies.\n\n"
            "© 2026 TradeTalkerAI™. All rights reserved.\n"
            "Unauthorized copying, redistribution, reverse engineering, or resale is prohibited."
        )

        ttk.Label(self.tab_about, text=about_text, wraplength=680, justify="left").pack(anchor="w", fill="x")

        about_links = ttk.Frame(self.tab_about)
        about_links.pack(fill="x", pady=(14, 0))

        ttk.Label(about_links, text="Website:").pack(side="left")

        site_link = tk.Label(about_links, text="TradeTalkerAI.com", fg="blue", cursor="hand2")
        site_link.pack(side="left", padx=(6, 0))

        def _open_site_about(_event=None):
            import webbrowser
            webbrowser.open("https://tradetalkerai.com")

        site_link.bind("<Button-1>", _open_site_about)

        # Settings tab (scrollable)
        self.canvas = tk.Canvas(self.tab_settings, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.tab_settings, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.frame = ttk.Frame(self.canvas, padding=14)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.frame, anchor="nw")

        self.frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

        self.var_query = tk.StringVar(value=settings.window_query)
        self.var_delay = tk.StringVar(value=str(settings.delay_ms))
        self.var_mode = tk.StringVar(value=settings.mode)
        self.var_voice = tk.StringVar(value=str(settings.voice))
        self.var_voice_speed = tk.DoubleVar(value=float(getattr(settings, 'voice_speed', DEFAULT_VOICE_SPEED)))
        self.var_hk_full = tk.StringVar(value=settings.hk_full)
        self.var_hk_brief = tk.StringVar(value=settings.hk_brief)
        self.var_hk_momo = tk.StringVar(value=settings.hk_momo)
        self.var_save = tk.BooleanVar(value=bool(settings.save_chart))
        self.var_save_images_local = tk.BooleanVar(value=bool(getattr(settings, 'save_images_local', DEFAULT_SAVE_IMAGES_LOCAL)))
        self.var_open = tk.BooleanVar(value=bool(settings.open_audio))
        self.var_play_local = tk.BooleanVar(value=bool(settings.play_local_audio))
        self.var_api = tk.StringVar(value=settings.api_url)
        self.var_subscriber = tk.StringVar(value=getattr(settings, 'subscriber_id', ''))
        self._device_id = get_or_create_device_id()

        self._build()

        # Load any prior log entries
        self.set_logs(existing_logs or [])

    def _on_frame_configure(self, _evt=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def _build(self):
        ttk.Label(self.frame, text="Select chart window (title contains):").pack(anchor="w")
        ttk.Entry(self.frame, textvariable=self.var_query).pack(fill="x", pady=(2, 10))

        ttk.Label(self.frame, text="Or pick from open windows:").pack(anchor="w")
        self.cmb_windows = ttk.Combobox(self.frame, state="readonly")
        self.cmb_windows.pack(fill="x", pady=(2, 6))
        self._refresh_windows()

        btn_row = ttk.Frame(self.frame)
        btn_row.pack(fill="x", pady=(0, 10))
        ttk.Button(btn_row, text="Refresh list", command=self._refresh_windows).pack(side="left")
        ttk.Button(btn_row, text="Use selected window title", command=self._use_selected_title).pack(side="left", padx=8)

        ttk.Label(self.frame, text="Delay between frames (ms):").pack(anchor="w")
        ttk.Entry(self.frame, textvariable=self.var_delay).pack(anchor="w", pady=(2, 10))

        ttk.Label(self.frame, text="Voice (1..6): 1=Oscar 2=Alan 3=Victor 4=Edward 5=Nancy 6=Marla").pack(anchor="w")
        ttk.Entry(self.frame, textvariable=self.var_voice).pack(anchor="w", pady=(2, 10))

        ttk.Label(self.frame, text="Voice Speed:").pack(anchor="w")
        vs_row = ttk.Frame(self.frame)
        vs_row.pack(fill='x', pady=(2, 2))

        ttk.Label(vs_row, text='Slow').pack(side='left')

        # Value readout (updates while sliding)
        lbl_voice_speed_value = ttk.Label(self.frame, text=f"{float(self.var_voice_speed.get() or 1.0):.1f}")

        # ttk.Scale is continuous; we "snap" to 0.1 increments so values are:
        # .5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5
        def _snap_voice_speed(_val: str) -> None:
            try:
                v = float(_val)
            except Exception:
                v = float(self.var_voice_speed.get() or 1.0)
            v = max(0.5, min(1.5, v))
            v = round(v * 10.0) / 10.0
            # Avoid feedback loops by only setting when changed.
            if abs(self.var_voice_speed.get() - v) > 1e-9:
                self.var_voice_speed.set(v)
            lbl_voice_speed_value.configure(text=f"{v:.1f}")

        vs_scale = ttk.Scale(
            vs_row,
            from_=0.5,
            to=1.5,
            orient='horizontal',
            variable=self.var_voice_speed,
            command=_snap_voice_speed,
        )
        vs_scale.pack(side='left', fill='x', expand=True, padx=10)

        ttk.Label(vs_row, text='Fast').pack(side='left')

        lbl_voice_speed_value.pack(anchor="center", pady=(0, 10))


        ttk.Label(self.frame, text="Hotkeys (global):").pack(anchor="w")
        hk_grid = ttk.Frame(self.frame)
        hk_grid.pack(fill="x", pady=(2, 10))


        ttk.Label(self.frame, text="Subscriber ID (license):").pack(anchor="w")
        ttk.Entry(self.frame, textvariable=self.var_subscriber).pack(fill="x", pady=(2, 10))

        # Device ID (read-only). Used to detect abuse / multiple installs.
        dev_row = ttk.Frame(self.frame)
        dev_row.pack(fill="x", pady=(0, 10))
        ttk.Label(dev_row, text="Device ID:").pack(side="left")
        dev_val = ttk.Entry(dev_row)
        dev_val.pack(side="left", fill="x", expand=True, padx=(6, 6))
        dev_val.insert(0, self._device_id)
        dev_val.configure(state="readonly")
        def _copy_dev():
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(self._device_id)
            except Exception:
                pass
        ttk.Button(dev_row, text="Copy", command=_copy_dev).pack(side="right")
        ttk.Label(hk_grid, text="Full").grid(row=0, column=0, sticky="w")
        ttk.Entry(hk_grid, textvariable=self.var_hk_full, width=16).grid(row=0, column=1, sticky="w", padx=(8, 18))
        ttk.Label(hk_grid, text="Brief").grid(row=0, column=2, sticky="w")
        ttk.Entry(hk_grid, textvariable=self.var_hk_brief, width=16).grid(row=0, column=3, sticky="w", padx=(8, 18))
        ttk.Label(hk_grid, text="Momo").grid(row=0, column=4, sticky="w")
        ttk.Entry(hk_grid, textvariable=self.var_hk_momo, width=16).grid(row=0, column=5, sticky="w", padx=(8, 0))

        # Examples line hidden (kept for future use)
        # ttk.Label(
        #     self.frame,
        #     text="Examples: F8 or Ctrl+Shift+F8. Defaults: Full=F8, Brief=F9, Momo=F10",
        #     foreground="#666",
        # ).pack(anchor="w", pady=(0, 10))
        row_caps = ttk.Frame(self.frame)
        row_caps.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(
            row_caps,
            text="Capture charts and messages",
            variable=self.var_save_images_local,
        ).pack(side="left")

        # Right-side buttons (stacked)
        caps_btns = ttk.Frame(row_caps)
        caps_btns.pack(side="right")
        ttk.Button(
            caps_btns,
            text="Open Captures Folder",
            command=lambda: open_folder(CAPTURES_DIR),
        ).pack(fill="x")
        ttk.Button(
            caps_btns,
            text="Delete captures",
            command=lambda: delete_all_captures_dialog(self.root),
        ).pack(fill="x", pady=(6, 0))


        # Audio behavior
        if playsound is None:
            ttk.Label(
                self.frame,
                text="Local audio playback unavailable (install 'playsound'); browser fallback will be used.",
                foreground="#a33",
            ).pack(anchor="w", pady=(0, 6))

        # Prefer local playback; browser is optional as a fallback.
        cb_local = ttk.Checkbutton(self.frame, text="Spoken Analysis", variable=self.var_play_local)
        cb_local.pack(anchor="w", pady=(0, 6))

        if playsound is None:
            cb_local.state(["disabled"])
            ttk.Label(
                self.frame,
                text="(Local audio disabled — install playsound:  python -m pip install playsound)",
                foreground="#666",
            ).pack(anchor="w", pady=(0, 6))

        # --- API URL hidden for production (kept for future use) ---
        # ttk.Label(self.frame, text="API URL:").pack(anchor="w")
        # ttk.Entry(self.frame, textvariable=self.var_api).pack(fill="x", pady=(2, 10))

        actions = ttk.Frame(self.frame)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Button(actions, text="Save settings", command=self._save).pack(side="left")
        ttk.Button(actions, text="Close", command=self.root.destroy).pack(side="right")
        ttk.Button(actions, text="Quit", command=self._quit_app).pack(side="right", padx=(0, 8))

        ttk.Label(self.frame, text=f"Settings file: {SETTINGS_PATH}", foreground="#666").pack(anchor="w", pady=(14, 0))

    def _refresh_windows(self):
        titles = list_open_windows()
        # Include a blank option so selection is not required.
        values = [""] + titles
        self.cmb_windows["values"] = values
        # Default to blank.
        self.cmb_windows.current(0)

    def _use_selected_title(self):
        t = self.cmb_windows.get().strip()
        if t:
            # Put something useful into query if empty
            if not self.var_query.get().strip():
                self.var_query.set(t[:16])

    def _save(self):
        # --- API URL validation disabled (hidden from UI) ---
        # api_url = self.var_api.get().strip()
        # if not (api_url.startswith("http://") or api_url.startswith("https://")):
        #     messagebox.showerror("API URL", "API URL must start with http:// or https://")
        #     return

        try:
            delay = int(self.var_delay.get().strip() or "0")
            if delay < 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Delay", "Delay must be an integer >= 0")
            return

        try:
            voice = int(self.var_voice.get().strip() or str(DEFAULT_VOICE))
            if voice not in VOICE_MAP:
                raise ValueError
        except Exception:
            messagebox.showerror("Voice", "Voice must be an integer 1..6")
            return

        mode = self.var_mode.get().strip()
        if mode not in MODE_OPTIONS:
            messagebox.showerror("Mode", f"Mode must be one of {MODE_OPTIONS}")
            return

        self.settings.window_query = self.var_query.get().strip()
        self.settings.delay_ms = delay
        self.settings.mode = mode
        self.settings.voice = voice
        # Voice speed (0.5..1.5)
        try:
            vs = float(self.var_voice_speed.get())
        except Exception:
            vs = DEFAULT_VOICE_SPEED
        if vs < 0.5: vs = 0.5
        if vs > 1.5: vs = 1.5
        self.settings.voice_speed = vs
        self.settings.subscriber_id = self.var_subscriber.get().strip()
        self.settings.save_chart = bool(self.var_save.get())
        self.settings.save_images_local = bool(self.var_save_images_local.get())
        self.settings.open_audio = bool(self.var_open.get())
        self.settings.play_local_audio = bool(self.var_play_local.get())
        # self.settings.api_url = api_url  # disabled (API hidden)

        # If local playback isn't available, fall back to browser.
        if self.settings.play_local_audio and playsound is None:
            self.settings.play_local_audio = False
            self.settings.open_audio = True

        # Hotkeys (global). Accept simple keys like F8 or combos like Ctrl+Shift+F8.
        hk_full = _normalize_hotkey_spec(self.var_hk_full.get(), DEFAULT_HK_FULL)
        hk_brief = _normalize_hotkey_spec(self.var_hk_brief.get(), DEFAULT_HK_BRIEF)
        hk_momo = _normalize_hotkey_spec(self.var_hk_momo.get(), DEFAULT_HK_MOMO)

        # If pynput is installed, validate that we can parse these.
        if pynput_keyboard is not None:
            try:
                _parse_hotkey_spec(hk_full)
                _parse_hotkey_spec(hk_brief)
                _parse_hotkey_spec(hk_momo)
            except Exception:
                messagebox.showerror(
                    "Hotkeys",
                    "Hotkeys must look like 'F8' or 'Ctrl+Shift+F8'.\nExamples: F8, F9, F10",
                )
                return

        self.settings.hk_full = hk_full
        self.settings.hk_brief = hk_brief
        self.settings.hk_momo = hk_momo

        try:
            save_settings(self.settings)
        except Exception as e:
            messagebox.showerror("Save", f"Could not save settings.json:\n{e}")
            return

        try:
            self.on_save_cb(self.settings)
        except Exception:
            pass

        # Confirm to user
        try:
            messagebox.showinfo("Saved", "Settings saved.", parent=self.root)
        except Exception:
            pass


    # -------- Messages tab helpers --------

    def _quit_app(self):
        # Play the exit bell, then wait long enough for the MP3 to be heard.
        try:
            play_file_locally(STARTUP_CHIME_PATH)
            time.sleep(3.1)  # openbell.mp3 is ~2.9s
        except Exception:
            pass

        """Quit and close the helper completely (tray icon + hotkeys)."""
        try:
            if callable(self.on_quit_cb):
                self.on_quit_cb()
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass

    def _clear_messages(self):
        self.txt_messages.configure(state="normal")
        self.txt_messages.delete("1.0", "end")
        self.txt_messages.configure(state="disabled")

    def set_logs(self, logs: list[LogEntry]) -> None:
        """Replace the message box contents with formatted log entries."""
        self._clear_messages()
        if not logs:
            return
        self.txt_messages.configure(state="normal")
        for e in logs:
            try:
                self.txt_messages.insert("end", e.format_line() + "\n")
            except Exception:
                pass
        self.txt_messages.see("end")
        self.txt_messages.configure(state="disabled")

    def append_log(self, entry: LogEntry) -> None:
        """Thread-safe append. Can be called from non-UI threads."""
        def _do():
            try:
                self.txt_messages.configure(state="normal")
                self.txt_messages.insert("end", entry.format_line() + "\n")
                self.txt_messages.see("end")
                self.txt_messages.configure(state="disabled")
            except Exception:
                pass
        try:
            self.root.after(0, _do)
        except Exception:
            pass

        # Suppressed blocking dialog to avoid delaying TTS playback.
        # messagebox.showinfo("Saved", "Settings saved.")

    def show(self):
        self.root.mainloop()


class HelperApp:
    def __init__(self):
        self.settings = load_settings()

        # Message log (time, mode, transcript)
        self._logs: list[LogEntry] = []
        self._ui: Optional[SettingsUI] = None

        # Prevent overlapping captures
        self._capture_lock = threading.Lock()
        self._capturing = False

        # Hotkey state (global)
        self._hk_listener = None
        self._hk_down_mods: set[str] = set()
        self._hk_fired: set[str] = set()

        self.icon = pystray.Icon(APP_NAME, self._make_icon(), APP_NAME, self._build_menu())

        # Start listening immediately
        self._start_hotkeys()

        # Startup sound once we're ready.
        self._startup_chime()

    def _make_icon(self) -> Image.Image:
        img = Image.new("RGB", (64, 64), "black")
        d = ImageDraw.Draw(img)
        d.rectangle((10, 10, 54, 54), outline="white", width=3)
        d.line((18, 42, 30, 30), fill="white", width=4)
        d.line((30, 30, 46, 22), fill="white", width=4)
        return img

    def _build_menu(self):
        return pystray.Menu(
            pystray.MenuItem("Messages", self._open_messages),
            pystray.MenuItem("Settings", self._open_settings),
            pystray.MenuItem("Quit", self._quit),
        )

    def _stop_hotkeys(self) -> None:
        if self._hk_listener is not None:
            try:
                self._hk_listener.stop()
            except Exception:
                pass
            self._hk_listener = None
        self._hk_down_mods.clear()
        self._hk_fired.clear()

    def _start_hotkeys(self) -> None:
        """Start/refresh global hotkeys (F8/F9/F10 by default)."""
        self._stop_hotkeys()
        if pynput_keyboard is None:
            return

        # Parse configured hotkeys
        hk_map: list[tuple[str, set[str], object, str]] = []
        specs = [
            ("full", self.settings.hk_full, "full"),
            ("brief", self.settings.hk_brief, "brief"),
            ("momo", self.settings.hk_momo, "momentum"),
        ]
        for label, spec, mode in specs:
            spec = _normalize_hotkey_spec(spec, {"full": DEFAULT_HK_FULL, "brief": DEFAULT_HK_BRIEF, "momo": DEFAULT_HK_MOMO}[label])
            mods, key_name = _parse_hotkey_spec(spec)
            key_obj = _pynput_key_from_name(key_name)
            if key_obj is None:
                continue
            hk_map.append((label, mods, key_obj, mode))

        if not hk_map:
            return

        def on_press(key):
            # Track modifiers
            try:
                if key in (pynput_keyboard.Key.ctrl, pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r):
                    self._hk_down_mods.add("ctrl")
                    return
                if key in (pynput_keyboard.Key.shift, pynput_keyboard.Key.shift_l, pynput_keyboard.Key.shift_r):
                    self._hk_down_mods.add("shift")
                    return
                if key in (pynput_keyboard.Key.alt, pynput_keyboard.Key.alt_l, pynput_keyboard.Key.alt_r):
                    self._hk_down_mods.add("alt")
                    return
            except Exception:
                pass

            for label, req_mods, key_obj, mode in hk_map:
                if key != key_obj:
                    continue
                if not req_mods.issubset(self._hk_down_mods):
                    continue
                # de-bounce: only fire once per key press
                if label in self._hk_fired:
                    continue
                self._hk_fired.add(label)
                # Play analysis cue immediately
                if mode == "full":
                    play_file_locally(ANALYZING_FULL_PATH)
                elif mode == "brief":
                    play_file_locally(ANALYZING_BRIEF_PATH)
                elif mode == "momentum":
                    play_file_locally(ANALYZING_MOMO_PATH)
                self._capture_with_mode(mode)

        def on_release(key):
            try:
                if key in (pynput_keyboard.Key.ctrl, pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r):
                    self._hk_down_mods.discard("ctrl")
                    return
                if key in (pynput_keyboard.Key.shift, pynput_keyboard.Key.shift_l, pynput_keyboard.Key.shift_r):
                    self._hk_down_mods.discard("shift")
                    return
                if key in (pynput_keyboard.Key.alt, pynput_keyboard.Key.alt_l, pynput_keyboard.Key.alt_r):
                    self._hk_down_mods.discard("alt")
                    return
            except Exception:
                pass

            # If any of our hotkeys were released, allow firing again
            for label, _req_mods, key_obj, _mode in hk_map:
                if key == key_obj:
                    self._hk_fired.discard(label)

        self._hk_listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
        self._hk_listener.daemon = True
        self._hk_listener.start()


    def _startup_chime(self) -> None:
        """Play startup bell, then waiting sound once ready."""
        try:
            # Small delay ensures the tray icon is fully initialized.
            def _delayed():
                time.sleep(0.15)
                # play openbell (blocking via temp thread sleep)
                play_file_locally(STARTUP_CHIME_PATH)
                # openbell is ~2.9s
                time.sleep(3.1)
                # then play waiting.mp3
                play_file_locally(WAITING_PATH)

            threading.Thread(target=_delayed, daemon=True).start()
        except Exception:
            pass


    def _open_messages(self, _icon=None, _item=None):
        def on_save(new_settings: Settings):
            self.settings = new_settings
            self._start_hotkeys()

        def run_ui():
            try:
                if self._ui is not None and self._ui.root.winfo_exists():
                    self._ui.root.after(0, lambda: self._ui.root.lift())
                    return
            except Exception:
                pass

            ui = SettingsUI(self.settings, on_save, on_quit_cb=self._quit, existing_logs=list(self._logs))
            # Force Settings tab
            ui.root.after(0, lambda: ui.notebook.select(ui.tab_settings))
            self._ui = ui
            try:
                # Force Messages tab
                ui.root.after(0, lambda: ui.notebook.select(ui.tab_messages))
                ui.show()
            finally:
                self._ui = None

        threading.Thread(target=run_ui, daemon=True).start()

    def _open_settings(self, _icon=None, _item=None):
        def on_save(new_settings: Settings):
            self.settings = new_settings
            self._start_hotkeys()

        def run_ui():
            # If already open, just bring it forward.
            try:
                if self._ui is not None and self._ui.root.winfo_exists():
                    self._ui.root.after(0, lambda: self._ui.root.lift())
                    return
            except Exception:
                pass

            ui = SettingsUI(self.settings, on_save, on_quit_cb=self._quit, existing_logs=list(self._logs))
            # Force Settings tab
            ui.root.after(0, lambda: ui.notebook.select(ui.tab_settings))
            self._ui = ui
            try:
                ui.show()
            finally:
                self._ui = None

        threading.Thread(target=run_ui, daemon=True).start()

    def _capture(self, _icon=None, _item=None):
        self._capture_with_mode(None)

    def _capture_with_mode(self, mode_override: str | None) -> None:
        # Avoid overlapping requests when the user spams hotkeys
        with self._capture_lock:
            if self._capturing:
                return
            self._capturing = True
        threading.Thread(target=self._capture_worker, args=(mode_override,), daemon=True).start()

    def _capture_worker(self, mode_override: str | None):
        try:
            s = self.settings
            if mode_override:
                s = Settings(**asdict(self.settings))
                s.mode = mode_override
    
            api = s.normalized_api_url()
            if not api:
                self._notify("Missing API URL", "Open Settings and paste your Railway URL.")
                return
    
            title = find_window_by_query(s.window_query) or ""
            if not title:
                self._notify("No window found", "Couldn't find a matching window. Make sure your chart app is open.")
                return
    
            img1 = capture_window(title)
            if img1 is None:
                self._notify("Capture failed", "Couldn't capture the window (maybe minimized).")
                return
    
            time.sleep(max(0, int(s.delay_ms)) / 1000.0)
            img2 = capture_window(title)
    
            # Optionally save captured frames locally for debugging/audit.
            capture_paths = None
            if getattr(s, 'save_images_local', False):
                try:
                    capture_paths = save_captures(img1, img2, s.mode)
                except Exception:
                    capture_paths = None
    
            result = post_analyze(api, img1, img2, s)
    
            # Handle subscription gate responses (401/403) returned as dict.
            try:
                sc = int(result.get("_status_code", 0) or 0)
            except Exception:
                sc = 0
            if sc in (401, 403):
                msg = (result.get("detail") or result.get("message") or "You need an active subscription to use TradeTalkerAI.").strip()
                self._record_log(s.mode, msg)
                self._notify("Subscription", msg[:120] + ("…" if len(msg) > 120 else ""))
                return
    
            audio_full = result.get("audio_full_url") or result.get("audio_url")
            transcript = result.get("transcript", "")
            self._record_log(s.mode, transcript)
    
            # If we saved captures for this hotkey press, also save the current message line.
            if capture_paths:
                try:
                    msg_path = message_txt_path_from_image_path(capture_paths[-1])
                    current_line = self._logs[-1].format_line() if self._logs else ""
                    write_message_log(msg_path, current_line)
                except Exception:
                    pass
    
            if audio_full:
                if s.play_local_audio:
                    started = play_audio_locally(audio_full)
                    # Optional fallback: open in browser if local playback isn't available.
                    if (not started) and s.open_audio:
                        open_in_browser(audio_full)
                elif s.open_audio:
                    open_in_browser(audio_full)
    
            self._notify("Analyzed", transcript[:120] + ("…" if len(transcript) > 120 else ""))
        except requests.exceptions.RequestException:
            # Network hiccup / timeout / server unreachable
            self._notify("Network down", "Network down — try again in a few minutes.")
        except Exception as e:
            self._notify("Error", str(e))
        finally:
            with self._capture_lock:
                self._capturing = False
    
    
        

    def _record_log(self, mode: str, transcript: str) -> None:
        try:
            entry = LogEntry(ts=time.time(), mode=mode or "", transcript=transcript or "")
            self._logs.append(entry)
            # Keep the log from growing without bound.
            if len(self._logs) > 500:
                self._logs = self._logs[-500:]
            if self._ui is not None:
                self._ui.append_log(entry)
        except Exception:
            pass
    
    

    def _notify(self, title: str, msg: str):
        try:
            self.icon.notify(msg, title)
        except Exception:
            pass
    
    

    def _quit(self, _icon=None, _item=None):
        self._stop_hotkeys()
        self.icon.stop()
    
    def run(self):
        self.icon.run()
def main():
    HelperApp().run()


if __name__ == "__main__":
    main()
