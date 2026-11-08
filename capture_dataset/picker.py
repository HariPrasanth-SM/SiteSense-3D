"""Explicit capture-only pins; no map editing or CUDA initialization."""
from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import ImageTk

from .common import (FloorMap, map_identity, pixel_to_world, positions_image,
                     read_json, select_stations, write_json, world_to_pixel)


class Picker:
    def __init__(self, root, args, cfg):
        self.root, self.args, self.cfg = root, args, cfg
        self.fmap = FloorMap.load(args.map)
        self.pins, self.excluded, self.stations = [], [], []
        self.identity = map_identity(args.map)
        self.selection_path = Path(args.out) / "selection.json"
        if self.selection_path.exists():
            saved = read_json(self.selection_path)
            if saved["map_identity"] != self.identity:
                raise ValueError("Existing selection belongs to another map. Use a new --out directory.")
            self.pins = saved["pins"]
            self.excluded = saved.get("excluded_capture_disks", [])
        root.title("Valmet | Capture planning only - navigation map unchanged")
        self.allow_unknown = tk.BooleanVar(value=False)
        self.status = tk.StringVar()
        ttk.Label(root, text="Left click: pin a camera. Right click: remove nearest pin. Shift+click: exclude a 0.4 m disk.").pack(anchor="w", padx=8, pady=5)
        row = ttk.Frame(root); row.pack(fill="x", padx=8)
        ttk.Checkbutton(row, text="Allow clicked GRAY cells for capture ONLY", variable=self.allow_unknown).pack(side="left")
        ttk.Button(row, text="Undo exclusion", command=self.undo_exclusion).pack(side="left", padx=6)
        ttk.Button(row, text="Recompute spread", command=self.refresh).pack(side="left")
        ttk.Button(row, text="SAVE PLAN + CLOSE", command=self.save).pack(side="left", padx=8)
        self.canvas = tk.Canvas(root, highlightthickness=0, background="#20252e")
        self.canvas.pack(padx=8, pady=6)
        ttk.Label(root, textvariable=self.status, wraplength=980).pack(fill="x", padx=8, pady=4)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Button-3>", self.remove)
        self.canvas.bind("<Motion>", self.hover)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.saved = False
        self.refresh()

    def locate(self, event):
        # positions_image contains a 62px header above the actual map.
        if event.y < 62 or event.x < 0 or event.x >= self.size[0] or event.y >= self.size[1] + 62:
            return None
        xy = pixel_to_world(self.fmap, [event.x, event.y - 62], self.size)
        cell = self.fmap.world_to_cell(xy)
        return (xy, cell) if self.fmap.contains(cell) else None

    def click(self, event):
        loc = self.locate(event)
        if loc is None: return
        xy, cell = loc
        if event.state & 0x0001:
            if any(np.linalg.norm(xy - [p["x"], p["y"]]) <= 0.4 for p in self.pins):
                self.status.set("Remove nearby pinned cameras before adding this exclusion."); return
            self.excluded.append({"x": float(xy[0]), "y": float(xy[1]), "radius": 0.4})
            self.refresh(); return
        state = int(self.fmap.occupancy[cell])
        if state == 100:
            self.status.set("Occupied cell: not accepted. Use existing manual viewer to inspect it first."); return
        if state == -1 and not self.allow_unknown.get():
            self.status.set("Gray cell: explicitly enable the capture-only checkbox to pin it. No navigation permission is granted."); return
        xy = self.fmap.cell_to_world(cell)
        if any(self.fmap.world_to_cell([p["x"], p["y"]]) == cell for p in self.pins): return
        if len(self.pins) >= self.args.positions:
            self.status.set("Pin count equals total position budget. Remove a pin or increase --positions."); return
        self.pins.append({"x": float(xy[0]), "y": float(xy[1]), "allow_unknown": state == -1,
                          "note": "User-selected capture location; not a navigation clearance annotation."})
        self.refresh()

    def remove(self, event):
        loc = self.locate(event)
        if loc is None or not self.pins: return
        xy, _ = loc
        distance = np.array([np.linalg.norm(xy - [p["x"], p["y"]]) for p in self.pins])
        if float(distance.min()) <= 0.65:
            self.pins.pop(int(distance.argmin())); self.refresh()

    def hover(self, event):
        loc = self.locate(event)
        if loc is None: return
        xy, cell = loc
        state = {0: "known-free", -1: "unknown", 100: "occupied"}[int(self.fmap.occupancy[cell])]
        self.status.set(f"X={xy[0]:.3f}, Y={xy[1]:.3f} provisional m | {state} | "
                        f"{len(self.pins)} pinned + {len(self.stations)-len(self.pins)} auto positions. "
                        "Pin 5-8 positions along the inspected gray corridor; free-space coverage fills automatically.")

    def undo_exclusion(self):
        if self.excluded: self.excluded.pop(); self.refresh()

    def refresh(self):
        try:
            self.stations = select_stations(self.fmap, self.pins, self.excluded, self.args.positions,
                                             self.args.clearance, self.args.spacing)
            image = positions_image(self.fmap, self.stations, self.excluded,
                                    max_size=(min(1100, self.root.winfo_screenwidth()-80),
                                              max(250, self.root.winfo_screenheight()-260)))
            self.size = (image.width, image.height - 62)
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.configure(width=image.width, height=image.height)
            self.canvas.delete("all"); self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
            n = len(self.stations)
            self.status.set(f"{len(self.pins)} pinned, {n-len(self.pins)} automatic; "
                            f"{n * len(self.cfg['yaws_deg']) * len(self.cfg['pitches_deg'])} planned frames. "
                            "No 3-D collision/visibility check. Purple pins remain unknown in the navigation map.")
        except Exception as exc:
            self.status.set(str(exc))

    def save(self):
        try:
            from .common import create_plan, save_plan
            self.stations = select_stations(self.fmap, self.pins, self.excluded, self.args.positions,
                                             self.args.clearance, self.args.spacing)
            Path(self.args.out).mkdir(parents=True, exist_ok=True)
            write_json(self.selection_path, {"map_identity": self.identity, "pins": self.pins,
                                            "excluded_capture_disks": self.excluded})
            plan = create_plan(self.args.map, self.args.scene, self.stations, self.pins, self.excluded, self.cfg)
            save_plan(self.args.out, plan)
            self.saved = True
            print(f"Saved {len(self.stations)} positions, {len(plan['frames'])} frames to {self.args.out}", flush=True)
            print(f"Pilot: {sum(v['pilot'] for v in plan['frames'])} frames. Next run: render --pilot", flush=True)
            self.root.destroy()
        except Exception as exc:
            messagebox.showerror("Cannot save capture plan", str(exc))

    def close(self):
        if messagebox.askyesno("Close without saving?", "Close without committing this capture plan?"):
            self.root.destroy()


def run_picker(args, cfg):
    if (Path(args.out) / "plan.json").exists():
        raise ValueError("This dataset already has an immutable plan. Use a NEW --out for revised poses.")
    root = tk.Tk()
    Picker(root, args, cfg)
    root.mainloop()
