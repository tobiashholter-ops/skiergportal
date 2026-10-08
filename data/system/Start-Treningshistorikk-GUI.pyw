from __future__ import annotations

import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Tkinter mangler i Python-installasjonen: {exc}")


ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
REQ = ROOT / "treningshistorikk_app" / "requirements.txt"
APP = ROOT / "treningshistorikk_app" / "app.py"
SECRETS_DIR = ROOT / ".streamlit"
SECRETS_FILE = SECRETS_DIR / "secrets.toml"

LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "streamlit.log"

URL = "http://localhost:8501"

CREATE_NO_WINDOW = 0x08000000


def _center_window(win: tk.Tk, width: int, height: int) -> None:
    win.update_idletasks()
    screen_w = win.winfo_screenwidth()
    screen_h = win.winfo_screenheight()
    x = max(0, (screen_w - width) // 2)
    y = max(0, (screen_h - height) // 2)
    win.geometry(f"{width}x{height}+{x}+{y}")


def _draw_logo(canvas: tk.Canvas, *, accent: str, fg: str, bg: str) -> None:
    canvas.configure(width=56, height=56, highlightthickness=0, bd=0, bg=bg)

    # Simple "chart-in-a-circle" mark.
    pad = 6
    canvas.create_oval(pad, pad, 56 - pad, 56 - pad, fill=accent, outline=accent)

    # Trend line
    canvas.create_line(
        18,
        34,
        26,
        28,
        32,
        31,
        40,
        22,
        fill=fg,
        width=3,
        capstyle=tk.ROUND,
        joinstyle=tk.ROUND,
    )
    # Small dot
    canvas.create_oval(38.5, 20.5, 42.5, 24.5, fill=fg, outline=fg)


def read_saved_token() -> str:
    if not SECRETS_FILE.exists():
        return ""
    try:
        txt = SECRETS_FILE.read_text(encoding="utf-8")
    except Exception:
        return ""

    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("CONCEPT2_ACCESS_TOKEN") and "=" in line:
            _, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            return value
    return ""


def save_token(token: str) -> None:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    SECRETS_FILE.write_text(f'CONCEPT2_ACCESS_TOKEN="{token}"\n', encoding="utf-8")


def ensure_venv_and_deps(status_cb) -> None:
    if not VENV_PY.exists():
        status_cb("Oppretter .venv …")
        subprocess.check_call(
            [sys.executable, "-m", "venv", str(ROOT / ".venv")],
            cwd=str(ROOT),
            creationflags=CREATE_NO_WINDOW,
        )

    status_cb("Installerer avhengigheter …")
    subprocess.check_call(
        [str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=str(ROOT),
        creationflags=CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        [str(VENV_PY), "-m", "pip", "install", "-r", str(REQ)],
        cwd=str(ROOT),
        creationflags=CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def kill_orphaned_streamlit() -> None:
    """Stop any leftover Streamlit processes that still reference the app.

    This can happen after refactors/moves; then a browser may hit an old server which
    tries to load a now-missing file.
    """

    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -and $_.CommandLine -match 'streamlit run' -and $_.CommandLine -match 'treningshistorikk_app\\\\app\\.py' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(ROOT),
            creationflags=CREATE_NO_WINDOW,
            text=True,
        ).strip()
    except Exception:
        return

    if not out:
        return

    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                cwd=str(ROOT),
                creationflags=CREATE_NO_WINDOW,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            continue


class Launcher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Treningshistorikk")
        self.resizable(False, False)

        # A minimal modern palette.
        self._bg = "#0b1220"
        self._card = "#111a2e"
        self._text = "#e5e7eb"
        self._muted = "#a7b0c0"
        self._accent = "#4f7cff"

        self.configure(bg=self._bg)
        _center_window(self, 640, 360)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("App.TFrame", background=self._bg)
        style.configure("Card.TFrame", background=self._card)

        style.configure("Title.TLabel", background=self._bg, foreground=self._text, font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=self._bg, foreground=self._muted, font=("Segoe UI", 10))

        style.configure("CardTitle.TLabel", background=self._card, foreground=self._text, font=("Segoe UI", 11, "bold"))
        style.configure("CardText.TLabel", background=self._card, foreground=self._muted, font=("Segoe UI", 9))

        style.configure("TCheckbutton", background=self._card, foreground=self._muted)
        style.map("TCheckbutton", foreground=[("disabled", "#6b7280")])

        style.configure(
            "Primary.TButton",
            background=self._accent,
            foreground="#ffffff",
            padding=(12, 8),
            borderwidth=0,
            focusthickness=0,
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#3f66db"), ("disabled", "#374151")],
            foreground=[("disabled", "#9ca3af")],
        )

        style.configure(
            "Secondary.TButton",
            background="#1b2743",
            foreground=self._text,
            padding=(12, 8),
            borderwidth=0,
            focusthickness=0,
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#233255"), ("disabled", "#111827")],
            foreground=[("disabled", "#9ca3af")],
        )

        style.configure("Status.TLabel", background=self._bg, foreground=self._muted, font=("Segoe UI", 9))

        style.configure(
            "Launch.Horizontal.TProgressbar",
            troughcolor=self._card,
            background=self._accent,
            borderwidth=0,
            thickness=5,
        )

        self.proc: subprocess.Popen[str] | None = None
        self.log_fp = None

        root = ttk.Frame(self, style="App.TFrame", padding=(18, 16))
        root.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(root, style="App.TFrame")
        header.pack(fill=tk.X)

        logo = tk.Canvas(header)
        _draw_logo(logo, accent=self._accent, fg="#ffffff", bg=self._bg)
        logo.pack(side=tk.LEFT)

        header_text = ttk.Frame(header, style="App.TFrame")
        header_text.pack(side=tk.LEFT, padx=(14, 0), fill=tk.X, expand=True)

        ttk.Label(header_text, text="Treningshistorikk", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header_text,
            text="Starter din lokale treningsanalyse (CSV eller Concept2 API)",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        # Cards
        cards = ttk.Frame(root, style="App.TFrame")
        cards.pack(fill=tk.BOTH, expand=True, pady=(16, 0))
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=0)

        left = ttk.Frame(cards, style="Card.TFrame", padding=16)
        left.grid(row=0, column=0, sticky="nsew")

        ttk.Label(left, text="Tilkobling", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            left,
            text="Token er valgfritt. Uten token bruker du CSV fra system/Data/.",
            style="CardText.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 12))

        ttk.Label(left, text="Concept2 token (valgfritt)", style="CardText.TLabel").grid(row=2, column=0, sticky="w")

        self.token_var = tk.StringVar(value=os.getenv("CONCEPT2_ACCESS_TOKEN", "") or read_saved_token())
        self.token_entry = ttk.Entry(left, textvariable=self.token_var, show="*")
        self.token_entry.grid(row=3, column=0, sticky="ew", pady=(6, 10))
        left.columnconfigure(0, weight=1)

        self.remember_var = tk.BooleanVar(value=bool(read_saved_token()))
        ttk.Checkbutton(left, text="Husk token lokalt (.streamlit/secrets.toml)", variable=self.remember_var).grid(
            row=4,
            column=0,
            sticky="w",
        )

        right = ttk.Frame(cards, style="Card.TFrame", padding=16)
        right.grid(row=0, column=1, sticky="ns", padx=(14, 0))

        ttk.Label(right, text="Handlinger", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(right, text="Alt kjører lokalt på denne PC-en.", style="CardText.TLabel").pack(anchor="w", pady=(2, 12))

        self.start_btn = ttk.Button(right, text="Start", style="Primary.TButton", command=self.start)
        self.start_btn.pack(fill=tk.X)

        self.stop_btn = ttk.Button(right, text="Stopp", style="Secondary.TButton", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(right, text="Åpne i nettleser", style="Secondary.TButton", command=lambda: webbrowser.open(URL)).pack(
            fill=tk.X, pady=(10, 0)
        )
        ttk.Button(right, text="Vis logg", style="Secondary.TButton", command=self.show_log).pack(fill=tk.X, pady=(10, 0))

        # Status
        self.status_var = tk.StringVar(value="Klar")
        ttk.Label(root, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w", pady=(14, 2))

        self._progressbar = ttk.Progressbar(
            root,
            style="Launch.Horizontal.TProgressbar",
            mode="indeterminate",
            length=200,
        )
        self._progressbar.pack(fill=tk.X)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.update_idletasks()

    def _pb_start(self) -> None:
        self._progressbar.start(12)
        self.update_idletasks()

    def _pb_stop(self) -> None:
        self._progressbar.stop()
        self._progressbar["value"] = 0
        self.update_idletasks()

    def start(self) -> None:
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Treningshistorikk", "Appen kjører allerede.")
            return

        # Give immediate visual feedback before the background thread starts
        self.start_btn.configure(state=tk.DISABLED, text="Starter …")
        self.set_status("Forbereder oppstart …")
        self._pb_start()

        token = self.token_var.get().strip()
        if token and self.remember_var.get():
            try:
                save_token(token)
            except Exception as exc:
                messagebox.showwarning("Treningshistorikk", f"Klarte ikke lagre token: {exc}")

        def worker() -> None:
            try:
                kill_orphaned_streamlit()
                ensure_venv_and_deps(self.set_status)
                self.set_status("Starter UI …")

                env = os.environ.copy()
                if token:
                    env["CONCEPT2_ACCESS_TOKEN"] = token

                cmd = [
                    str(VENV_PY),
                    "-m",
                    "streamlit",
                    "run",
                    str(APP),
                    "--server.address",
                    "localhost",
                    "--server.port",
                    "8501",
                    "--server.headless",
                    "true",
                ]

                LOG_DIR.mkdir(parents=True, exist_ok=True)
                self.log_fp = open(LOG_FILE, "a", encoding="utf-8", errors="replace")
                self.log_fp.write("\n--- start ---\n")
                self.log_fp.write("cmd: " + " ".join(cmd) + "\n")
                self.log_fp.flush()

                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    creationflags=CREATE_NO_WINDOW,
                    stdout=self.log_fp,
                    stderr=self.log_fp,
                    text=True,
                )

                self.after(0, lambda: webbrowser.open(URL))
                self.after(0, self._set_running_ui)
                self.set_status(f"Kjører: {URL}")
            except subprocess.CalledProcessError as exc:
                self.after(0, lambda: messagebox.showerror("Treningshistorikk", f"Feil ved oppstart: {exc}"))
                self.after(0, self._set_stopped_ui)
                self.set_status("Feil")
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Treningshistorikk", f"Uventet feil: {exc}"))
                self.after(0, self._set_stopped_ui)
                self.set_status("Feil")

        threading.Thread(target=worker, daemon=True).start()

    def _set_running_ui(self) -> None:
        self._pb_stop()
        self.start_btn.configure(state=tk.DISABLED, text="Start")
        self.stop_btn.configure(state=tk.NORMAL)

    def _set_stopped_ui(self) -> None:
        self._pb_stop()
        self.start_btn.configure(state=tk.NORMAL, text="Start")
        self.stop_btn.configure(state=tk.DISABLED)

    def stop(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            self._set_stopped_ui()
            self.set_status("Stoppet")
            return

        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        finally:
            self.proc = None
            if self.log_fp is not None:
                try:
                    self.log_fp.write("--- stop ---\n")
                    self.log_fp.flush()
                except Exception:
                    pass
                try:
                    self.log_fp.close()
                except Exception:
                    pass
                self.log_fp = None
            self._set_stopped_ui()
            self.set_status("Stoppet")

    def show_log(self) -> None:
        if not LOG_FILE.exists():
            messagebox.showinfo("Treningshistorikk", "Ingen loggfil enda. Start appen først.")
            return
        try:
            os.startfile(str(LOG_FILE))  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("Treningshistorikk", f"Klarte ikke åpne loggfil: {exc}")

    def on_close(self) -> None:
        self.stop()
        self.destroy()


if __name__ == "__main__":
    Launcher().mainloop()
