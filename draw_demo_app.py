#!/usr/bin/env python3
"""
Draw a digit once, then classify it on Hardware or Software each with its own 
result panel (prediction + confidence bar chart + timing), 
so you can compare the SAME input across both backends.

For the board:
    ssh petalinux@<board-ip>
    ./lenet5_host -x lenet5.bin -i some_file.txt      # should print "FPGA time : ..."
    ./lenet5_host -x lenet5.bin -s some_file.txt       # should print "ARM  time : ..."

Requirements: pip install pillow AND sudo apt install python3-tk

"""
import subprocess
import tempfile
import os
import re
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageDraw, ImageOps

# ============================== CONFIG ===============================
BOARD_USER = "petalinux"
BOARD_IP = "10.42.0.132"
REMOTE_HOME = f"/home/{BOARD_USER}"
REMOTE_BIN = f"{REMOTE_HOME}/lenet5_host"
REMOTE_BITSTREAM = f"{REMOTE_HOME}/lenet5.bin"
REMOTE_IMG = f"{REMOTE_HOME}/live_digit.txt"
SSH_TIMEOUT_SEC = 12


CANVAS_SIZE = 360
BRUSH_RADIUS = 12
MNIST_SIZE = 28


def preprocess_to_text(pil_img):
    img = pil_img.resize((MNIST_SIZE, MNIST_SIZE), Image.LANCZOS)
    img = ImageOps.invert(img)
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(path, "w") as f:
        for y in range(MNIST_SIZE):
            row = [str(img.getpixel((x, y))) for x in range(MNIST_SIZE)]
            f.write(" ".join(row) + "\n")
    return path


def parse_prediction(out):
    pred_match = re.search(r"Predicted digit:\s*(\d)", out)
    if not pred_match:
        return None, {}, None
    pred = int(pred_match.group(1))
    probs = {}
    for m in re.finditer(r"^\s*(\d)\s*:\s*([-\d.]+)\s*([\d.]+)%", out, re.MULTILINE):
        probs[int(m.group(1))] = float(m.group(3))
    time_match = re.search(r"(?:FPGA|ARM)\s+time\s*:\s*([\d.]+)\s*us", out)
    time_us = float(time_match.group(1)) if time_match else None
    return pred, probs, time_us


def run_cmd(cmd, timeout):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"exit code {result.returncode}")
    return result.stdout


def run_hardware(local_txt_path):
    run_cmd(["scp", local_txt_path, f"{BOARD_USER}@{BOARD_IP}:{REMOTE_IMG}"],
             SSH_TIMEOUT_SEC)
    remote_cmd = f"{REMOTE_BIN} -x {REMOTE_BITSTREAM} -i {REMOTE_IMG}"
    return run_cmd(["ssh", f"{BOARD_USER}@{BOARD_IP}", remote_cmd], SSH_TIMEOUT_SEC)


def run_software(local_txt_path):
    run_cmd(["scp", local_txt_path, f"{BOARD_USER}@{BOARD_IP}:{REMOTE_IMG}"],
             SSH_TIMEOUT_SEC)
    remote_cmd = f"{REMOTE_BIN} -x {REMOTE_BITSTREAM} -s {REMOTE_IMG}"
    return run_cmd(["ssh", f"{BOARD_USER}@{BOARD_IP}", remote_cmd], SSH_TIMEOUT_SEC)


class ResultPanel(tk.Frame):
    def __init__(self, parent, title, accent, fonts, **kwargs):
        super().__init__(parent, bg="#0f172a", **kwargs)
        self.accent = accent
        big_font, label_font, status_font = fonts

        tk.Label(self, text=title, font=label_font, fg=accent, bg="#0f172a").pack(pady=(0, 4))

        row = tk.Frame(self, bg="#0f172a")
        row.pack()
        self.pred_label = tk.Label(row, text="--", font=big_font, fg=accent, bg="#0f172a")
        self.pred_label.pack(side="left", padx=(0, 16))

        info = tk.Frame(row, bg="#0f172a")
        info.pack(side="left")
        self.conf_label = tk.Label(info, text="", font=status_font, fg="#94a3b8", bg="#0f172a", justify="left")
        self.conf_label.pack(anchor="w")
        self.time_label = tk.Label(info, text="", font=status_font, fg=accent, bg="#0f172a", justify="left")
        self.time_label.pack(anchor="w")

        self.bars_canvas = tk.Canvas(self, width=430, height=396, bg="#0f172a", highlightthickness=0)
        self.bars_canvas.pack(pady=(10, 0))

        self.status = tk.Label(self, text="Ready.", font=status_font, fg="#94a3b8", bg="#0f172a", wraplength=430, justify="left")
        self.status.pack(pady=(6, 0))

        self.draw_bars({})

    def draw_bars(self, probs):
        c = self.bars_canvas
        c.delete("all")
        bar_h = 30
        gap = 8
        max_w = 260
        label_x = 20
        bar_x = 45
        top_pad = 10 
        pct_gap = 14 
        best = max(probs.values()) if probs else 0
        for digit in range(10):
            y = top_pad + digit * (bar_h + gap)
            p = probs.get(digit, 0.0)
            w = int(max_w * p / 100.0)
            color = self.accent if (p == best and p > 0) else "#334155"
            c.create_text(label_x, y + bar_h / 2, text=str(digit), fill="white", font=("Helvetica", 14, "bold"))
            c.create_rectangle(bar_x, y, bar_x + max_w, y + bar_h, fill="#1e293b", outline="")
            c.create_rectangle(bar_x, y, bar_x + w, y + bar_h, fill=color, outline="")
            c.create_text(bar_x + max_w + pct_gap, y + bar_h / 2, text=f"{p:.1f}%", fill="#94a3b8", font=("Consolas", 12), anchor="w")

    def set_status(self, text, color="#94a3b8"):
        self.status.config(text=text, fg=color)
        self.update_idletasks()

    def show_result(self, pred, probs, time_us):
        self.pred_label.config(text=str(pred))
        conf = probs.get(pred, 0)
        self.conf_label.config(text=f"{conf:.1f}% confident")
        if time_us is not None:
            self.time_label.config(text=f"{time_us:.1f} us")
        else:
            self.time_label.config(text="(no timing -- patch applied?)")
        self.draw_bars(probs)
        self.set_status("Done.", "#22c55e")

    def clear(self):
        self.pred_label.config(text="--")
        self.conf_label.config(text="")
        self.time_label.config(text="")
        self.draw_bars({})
        self.set_status("Ready.", "#94a3b8")


class DemoApp:
    def __init__(self, root):
        self.root = root
        root.title("LeNet-5 on KV260")
        root.configure(bg="#0f172a")

        fonts_draw = (
            tkfont.Font(family="Helvetica", size=18, weight="bold"),
            tkfont.Font(family="Helvetica", size=18),
            tkfont.Font(family="Consolas", size=13),
        )
        fonts_result = (
            tkfont.Font(family="Helvetica", size=64, weight="bold"),
            tkfont.Font(family="Helvetica", size=17),
            tkfont.Font(family="Consolas", size=13),
        )

        # ---- Left: single drawing canvas ----
        left = tk.Frame(root, bg="#0f172a")
        left.grid(row=0, column=0, padx=20, pady=20, sticky="n")

        tk.Label(left, text="Draw a digit (0-9)", font=fonts_draw[1],
                 fg="white", bg="#0f172a").pack(pady=(0, 8))

        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", highlightthickness=2, highlightbackground="#334155")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.on_draw)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.pil_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        self.last_xy = None

        tk.Button(left, text="Classify Hardware", font=fonts_draw[1], bg="#22d3ee", fg="#0f172a", padx=14, pady=8, command=self.classify_hw).pack(fill="x", pady=(14, 4))
        tk.Button(left, text="Classify Software", font=fonts_draw[1], bg="#a78bfa", fg="#0f172a", padx=14, pady=8, command=self.classify_sw).pack(fill="x", pady=4)
        tk.Button(left, text="Clear", font=fonts_draw[1], bg="#475569", fg="white", padx=14, pady=8, command=self.clear).pack(fill="x", pady=(4, 4))

        self.status = tk.Label(left, text="Ready.", font=fonts_draw[2], fg="#94a3b8", bg="#0f172a", wraplength=CANVAS_SIZE,justify="left")
        self.status.pack(pady=(6, 0))

        # ---- Right: two result panels side by side ----
        results = tk.Frame(root, bg="#0f172a")
        results.grid(row=0, column=1, padx=10, pady=20, sticky="n")

        self.hw_result = ResultPanel(results, "HARDWARE", "#22d3ee", fonts_result, padx=16)
        self.hw_result.grid(row=0, column=0, sticky="n")

        self.sw_result = ResultPanel(results, "SOFTWARE", "#a78bfa", fonts_result, padx=16)
        self.sw_result.grid(row=0, column=1, sticky="n")

    # ---------------- drawing ----------------
    def on_draw(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        if self.last_xy:
            self.canvas.create_line(self.last_xy[0], self.last_xy[1], x, y, fill="black", width=r * 2, capstyle="round", smooth=True)
            self.pil_draw.line([self.last_xy, (x, y)], fill=0, width=r * 2)
        self.pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
        self.last_xy = (x, y)

    def on_release(self, event):
        self.last_xy = None

    def clear(self):
        self.canvas.delete("all")
        self.pil_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        self.hw_result.clear()
        self.sw_result.clear()
        self.set_status("Ready.", "#94a3b8")

    def set_status(self, text, color="#94a3b8"):
        self.status.config(text=text, fg=color)
        self.root.update_idletasks()

    # ---------------- classify ----------------
    def _classify_one(self, backend_fn, debug_name, panel):
        panel.set_status("Preprocessing...", "#facc15")
        try:
            local_txt = preprocess_to_text(self.pil_img)
        except Exception as e:
            panel.set_status(f"Preprocess failed: {e}", "#ef4444")
            return

        panel.set_status("Classifying on board...", "#facc15")
        try:
            out = backend_fn(local_txt)
        except Exception as e:
            panel.set_status(f"Failed: {e}", "#ef4444")
            return

        pred, probs, time_us = parse_prediction(out)
        if pred is None:
            panel.set_status("Could not parse output (see console).", "#ef4444")
            print(f"---- raw output from {debug_name} ----\n{out}")
            return

        panel.show_result(pred, probs, time_us)

    def classify_hw(self):
        self.set_status("Classifying (hardware)...", "#facc15")
        self._classify_one(run_hardware, "hardware", self.hw_result)
        self.set_status("Ready.", "#94a3b8")

    def classify_sw(self):
        self.set_status("Classifying (software)...", "#facc15")
        self._classify_one(run_software, "software", self.sw_result)
        self.set_status("Ready.", "#94a3b8")


if __name__ == "__main__":
    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()