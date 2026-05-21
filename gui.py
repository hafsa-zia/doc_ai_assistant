from __future__ import annotations

import os
import re
import sys
import time
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ================= COLORS =================
COLOR_BG = "#f4f6f8"
COLOR_HEADER = "#1f2937"


class DocAIAssistantGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Intelligent Document Assistant")
        self.geometry("1100x720")
        self.configure(bg=COLOR_BG)

        self.proc = None

        self._build_style()
        self._build_ui()

    # ============== STYLE ==============
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TButton", padding=8)
        style.configure(
            "Primary.TButton",
            foreground="white",
            background="#2563eb",
            font=("Segoe UI", 10, "bold")
        )
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))

    # ============== UI ==============
    def _build_ui(self):
        # ---------- HEADER ----------
        header = tk.Frame(self, bg=COLOR_HEADER, padx=16, pady=12)
        header.pack(fill="x")

        tk.Label(
            header,
            text="Intelligent Document Assistant",
            bg=COLOR_HEADER,
            fg="white",
            font=("Segoe UI", 14, "bold")
        ).pack(side="left")

        tk.Label(
            header,
            text="Offline Extractive & LLM-based PDF Summarisation",
            bg=COLOR_HEADER,
            fg="#d1d5db",
            font=("Segoe UI", 10)
        ).pack(side="left", padx=14)

        # ---------- MAIN ----------
        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 12))

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        # ---------- FOLDERS ----------
        folders = ttk.LabelFrame(left, text="Folders", padding=10)
        folders.pack(fill="x")

        self.input_var = tk.StringVar(value=os.path.join(os.getcwd(), "data", "input"))
        self.output_var = tk.StringVar(value=os.path.join(os.getcwd(), "data", "output"))

        ttk.Label(folders, text="Input Folder").pack(anchor="w")
        row = ttk.Frame(folders)
        row.pack(fill="x", pady=4)
        ttk.Entry(row, textvariable=self.input_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse", command=self.pick_input).pack(side="left", padx=6)

        ttk.Label(folders, text="Output Folder").pack(anchor="w")
        row2 = ttk.Frame(folders)
        row2.pack(fill="x", pady=4)
        ttk.Entry(row2, textvariable=self.output_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row2, text="Browse", command=self.pick_output).pack(side="left", padx=6)

        # ---------- MODE ----------
        mode_box = ttk.LabelFrame(left, text="Run Mode", padding=10)
        mode_box.pack(fill="x", pady=10)

        self.mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(
            mode_box, text="Single PDF",
            variable=self.mode_var, value="single",
            command=self._on_mode_change
        ).pack(anchor="w")

        ttk.Radiobutton(
            mode_box, text="Batch (All PDFs)",
            variable=self.mode_var, value="batch",
            command=self._on_mode_change
        ).pack(anchor="w")

        self.single_frame = ttk.Frame(mode_box)
        self.single_frame.pack(fill="x", pady=6)

        ttk.Label(self.single_frame, text="Select PDF").pack(anchor="w")
        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(
            self.single_frame,
            textvariable=self.file_var,
            state="readonly"
        )
        self.file_combo.pack(fill="x", pady=4)

        ttk.Button(
            self.single_frame,
            text="Refresh",
            command=self.refresh_files
        ).pack(fill="x")

        # ---------- ACTIONS ----------
        actions = ttk.LabelFrame(left, text="Actions", padding=10)
        actions.pack(fill="x", pady=10)

        self.run_btn = ttk.Button(
            actions, text="Run",
            style="Primary.TButton",
            command=self.run_selected
        )
        self.run_btn.pack(fill="x")

        self.stop_btn = ttk.Button(
            actions, text="Stop",
            command=self.stop_run,
            state="disabled"
        )
        self.stop_btn.pack(fill="x", pady=6)

        ttk.Button(
            actions, text="Open Output Folder",
            command=self.open_output_folder
        ).pack(fill="x")

        # ---------- STATUS ----------
        status_box = ttk.LabelFrame(left, text="Status", padding=10)
        status_box.pack(fill="x", pady=10)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(
            status_box,
            textvariable=self.status_var,
            wraplength=280
        ).pack(anchor="w")

        self.progress = ttk.Progressbar(status_box, maximum=100)
        self.progress.pack(fill="x", pady=6)

        # ---------- LOGS ----------
        logs = ttk.LabelFrame(right, text="Live Output", padding=10)
        logs.pack(fill="both", expand=True)

        self.log = tk.Text(
            logs,
            wrap="word",
            font=("Consolas", 10),
            fg="black"
        )
        self.log.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(logs, command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scrollbar.set)

        # All log tags → BLACK
        self.log.tag_config("INFO", foreground="black")
        self.log.tag_config("SUCCESS", foreground="black")
        self.log.tag_config("WARN", foreground="black")
        self.log.tag_config("ERROR", foreground="black")

       
        self.refresh_files()
        self._on_mode_change()

    # ============== HELPERS ==============
    def _log(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def pick_input(self):
        p = filedialog.askdirectory()
        if p:
            self.input_var.set(p)
            self.refresh_files()

    def pick_output(self):
        p = filedialog.askdirectory()
        if p:
            self.output_var.set(p)

    def open_output_folder(self):
        try:
            os.startfile(self.output_var.get())
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def refresh_files(self):
        in_dir = self.input_var.get()
        files = (
            [f for f in os.listdir(in_dir) if f.lower().endswith(".pdf")]
            if os.path.exists(in_dir)
            else []
        )
        self.file_combo["values"] = files
        if files:
            self.file_var.set(files[0])

    def _on_mode_change(self):
        if self.mode_var.get() == "single":
            self.single_frame.pack(fill="x")
        else:
            self.single_frame.pack_forget()

    # ============== RUN PROCESS ==============
    def run_selected(self):
        if self.proc:
            return

        script = "src/main.py" if self.mode_var.get() == "single" else "src/batch_run.py"
        self._start_process(script)

    def _start_process(self, script):
        self.log.delete("1.0", "end")
        self.progress["value"] = 0
        self.status_var.set("Running...")

        cmd = [sys.executable, "-u", script]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        threading.Thread(target=self._reader_thread, daemon=True).start()
        threading.Thread(target=self._wait_thread, daemon=True).start()

    def stop_run(self):
        if self.proc:
            self.proc.terminate()
            self._log("\n[Stopped by user]\n")

    def _reader_thread(self):
        for line in self.proc.stdout:
            self._parse_progress(line)
            self._log(line)
        self.proc = None

    def _wait_thread(self):
        time.sleep(0.5)
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.progress["value"] = 100
        self.status_var.set("Finished successfully")

    def _parse_progress(self, line):
        m = re.search(r"chunk\s+(\d+)/(\d+)", line)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            self.progress["value"] = int(cur / total * 90)
            self.status_var.set(f"LLM summarizing chunk {cur}/{total}...")


if __name__ == "__main__":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = DocAIAssistantGUI()
    app.mainloop()
