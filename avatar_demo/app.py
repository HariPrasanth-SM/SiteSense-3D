"""Click-to-navigate desktop UI. Run from the project root: python -m avatar_demo."""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import queue
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

from .core import FloorMap, Planner, PlanningError, Walker
from .render_worker import render_loop, save_frame
from .manual_ui import ManualControlsMixin
from .manual import point_allowed


class AvatarApp(ManualControlsMixin):
    def __init__(self, args, floor_map, planner):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.args, self.map, self.planner = args, floor_map, planner
        self.root = tk.Tk()
        self.root.title("Valmet | Manual inspection + avatar navigation + Gaussian RGB/depth")
        self.root.geometry("1440x940")
        self.root.minsize(1180, 780)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.closed = False
        self.session = Path(args.output) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session.mkdir(parents=True, exist_ok=False)
        with (self.session / "settings.json").open("w") as stream:
            json.dump(vars(args), stream, indent=2, default=str)
        self.log_file = (self.session / "trajectory.csv").open("w", newline="")
        self.csv = csv.writer(self.log_file)
        self.csv.writerow(["simulation_time_s", "map_x_m", "map_y_m", "yaw_rad", "state",
                           "camera_height_m", "pitch_deg", "manual_map_checks", "manual_body_clearance",
                           "input_map_status", "footprint_candidate_free", "speed_m_s", "turn_deg_s"])
        self.last_log, self.sim_time = -1.0, 0.0
        self.last_tick = time.monotonic()
        self.last_command_stamp = None
        self.next_command_check = 0.0
        self.last_frame = None
        self.last_signature = None
        self.sequence = 0
        self.plan_count, self.snapshot_count = 0, 0
        self.target = None
        self.worker = None
        self.walker = None
        self.base_image_tk = None
        self.background_id = None
        self.map_geometry = None

        self.message = tk.StringVar(value="Select a target; inspect the route and camera before pressing Walk.")
        self.render_message = tk.StringVar(value="Map-only mode" if args.no_render else "Renderer starting...")
        self.coordinates = tk.StringVar(value="Map coordinates use the existing provisional metre scale.")
        self.motion_text = tk.StringVar(value="")
        self.click_mode = tk.StringVar(value="target")
        self.target_mode = tk.StringVar(value="object")
        self.x_var, self.y_var = tk.StringVar(value="0"), tk.StringVar(value="0")
        self.height_var = tk.StringVar(value=str(args.camera_height))
        self.pitch_var = tk.StringVar(value=str(args.pitch_deg))
        self.standoff_var = tk.StringVar(value=str(args.standoff))
        self.max_approach_var = tk.StringVar(value=str(args.max_approach))
        self.heading_var = tk.StringVar(value=str(args.yaw_deg))
        self.camera_height, self.pitch_deg = args.camera_height, args.pitch_deg
        self._init_manual()
        self._make_controls()
        self._make_map_image()
        try:
            start = planner.set_start(args.start) if args.start is not None else planner.auto_start()
            self.walker = Walker(start, math.radians(args.yaw_deg), args.speed, math.radians(args.turn_rate))
            self.message.set("Auto-start is map-derived, NOT scene-verified. Check the RGB view or set a start manually.")
        except PlanningError as exc:
            self.message.set(str(exc))
        if not args.no_render:
            ctx = mp.get_context("spawn")
            self.shared_pose = ctx.Array("d", [-1, 0, 0, 0, args.camera_height, args.pitch_deg, 0])
            self.frames, self.statuses = ctx.Queue(maxsize=1), ctx.Queue(maxsize=16)
            self.stop_event = ctx.Event()
            config = dict(vars(args), session=str(self.session), height=args.height)
            self.worker = ctx.Process(
                target=render_loop, args=(config, self.shared_pose, self.frames, self.statuses, self.stop_event),
                daemon=True, name="valmet-gsplat-renderer",
            )
            self._send_pose(force=True)
            self.worker.start()
        self._tick_job = self.root.after(60, self.tick)
        self.root.bind("<space>", self._space)
        self.root.bind("<Escape>", lambda event: self.pause())
        self._bind_manual_controls()
        if args.gui_test_seconds:
            self.root.after(int(args.gui_test_seconds * 1000), self.close)
        print(f"Session outputs: {self.session}", flush=True)

    def _make_controls(self):
        ttk, tk = self.ttk, self.tk
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill="x")
        ttk.Radiobutton(toolbar, text="Click: target", variable=self.click_mode, value="target").pack(side="left")
        ttk.Radiobutton(toolbar, text="Click: set start", variable=self.click_mode, value="start").pack(side="left", padx=6)
        ttk.Label(toolbar, text="Task:").pack(side="left", padx=(10, 2))
        ttk.Radiobutton(toolbar, text="Inspect object", variable=self.target_mode, value="object").pack(side="left")
        ttk.Radiobutton(toolbar, text="Go to free point", variable=self.target_mode, value="point").pack(side="left")
        ttk.Button(toolbar, text="Walk / resume", command=self.walk).pack(side="left", padx=8)
        ttk.Button(toolbar, text="Pause", command=self.pause).pack(side="left")
        ttk.Button(toolbar, text="Save camera frame", command=self.snapshot).pack(side="left", padx=8)

        row = ttk.Frame(self.root, padding=(6, 2))
        row.pack(fill="x")
        for text, var, width in [("Target X", self.x_var, 8), ("Y", self.y_var, 8),
                                 ("Stand-off", self.standoff_var, 5), ("Max approach", self.max_approach_var, 5)]:
            ttk.Label(row, text=text).pack(side="left", padx=(5, 2))
            ttk.Entry(row, textvariable=var, width=width).pack(side="left")
        ttk.Button(row, text="Plan target", command=self.plan_fields).pack(side="left", padx=8)
        ttk.Label(row, text="Idle heading (deg)").pack(side="left")
        ttk.Entry(row, textvariable=self.heading_var, width=5).pack(side="left", padx=3)
        ttk.Button(row, text="Apply heading", command=self.apply_heading).pack(side="left")

        row2 = ttk.Frame(self.root, padding=(6, 2))
        row2.pack(fill="x")
        ttk.Label(row2, text="Camera height (m)").pack(side="left")
        ttk.Entry(row2, textvariable=self.height_var, width=5).pack(side="left", padx=4)
        ttk.Label(row2, text="Pitch (deg, negative = down)").pack(side="left")
        ttk.Entry(row2, textvariable=self.pitch_var, width=5).pack(side="left", padx=4)
        ttk.Button(row2, text="Apply camera", command=self.apply_camera).pack(side="left", padx=6)
        ttk.Label(row2, textvariable=self.motion_text).pack(side="left", padx=10)

        self._make_manual_controls()

        body = ttk.Frame(self.root, padding=6)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(left, background="#151b24", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.redraw_map)
        self.canvas.bind("<Button-1>", self.map_click)
        self.canvas.bind("<Motion>", self.mouse_move)
        ttk.Label(left, text="Green: strict footprint area | Amber: input-free, allowed by point-centre manual checks\n"
                            "Dark: occupied | Gray: unknown | Blue: avatar | Cyan: rendered pose | Pink trail: unchecked").grid(row=1, column=0, sticky="w")
        ttk.Label(left, textvariable=self.coordinates).grid(row=2, column=0, sticky="w")
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        rgb_frame = ttk.LabelFrame(right, text="Simulated RGB — camera follows avatar heading")
        rgb_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        depth_frame = ttk.LabelFrame(right, text=f"Expected depth — near bright, far dark, range 0–{self.args.depth_display_max:g} m")
        depth_frame.grid(row=1, column=0, sticky="nsew")
        self.rgb_label = ttk.Label(rgb_frame, text="Waiting for first camera image", anchor="center")
        self.rgb_label.pack(expand=True, fill="both")
        self.depth_label = ttk.Label(depth_frame, text="Depth is synthetic; invalid pixels are black", anchor="center")
        self.depth_label.pack(expand=True, fill="both")
        self._make_review_controls()
        ttk.Label(self.root, textvariable=self.render_message, wraplength=1380, padding=(6, 2)).pack(fill="x")
        ttk.Label(self.root, textvariable=self.message, wraplength=1180, padding=(6, 4)).pack(fill="x")
        ttk.Label(self.root, text=f"Static 2-D kinematic demo. No 3-D/head-clearance guarantee. Scale {self.args.scale:g} is provisional; maps are never edited.",
                  padding=(6, 2)).pack(fill="x")

    def _make_map_image(self):
        image = np.full((*self.map.shape, 3), [109, 116, 127], dtype=np.uint8)
        image[self.map.occupancy == 100] = [37, 43, 53]
        image[self.map.occupancy == 0] = [171, 137, 73]
        image[self.planner.walkable] = [203, 226, 194]
        self.map_image = Image.fromarray(np.flipud(image))

    def redraw_map(self, event=None):
        width, height = max(self.canvas.winfo_width(), 10), max(self.canvas.winfo_height(), 10)
        rows, cols = self.map.shape
        scale = min((width - 28) / cols, (height - 28) / rows)
        if scale <= 0:
            return
        w, h = max(1, round(cols * scale)), max(1, round(rows * scale))
        x0, y0 = (width - w) / 2, (height - h) / 2
        self.map_geometry = (x0, y0, w / cols, h / rows)
        self.base_image_tk = ImageTk.PhotoImage(self.map_image.resize((w, h), Image.Resampling.NEAREST))
        self.canvas.delete("background")
        self.canvas.create_image(x0, y0, image=self.base_image_tk, anchor="nw", tags="background")
        self.canvas.tag_lower("background")
        self.draw_overlay()

    def world_to_pixel(self, xy):
        x0, y0, sx, sy = self.map_geometry
        c, r = self.map.world_to_grid_continuous(xy)
        return x0 + c * sx, y0 + (self.map.shape[0] - r) * sy

    def pixel_to_world(self, u, v):
        if self.map_geometry is None:
            return None
        x0, y0, sx, sy = self.map_geometry
        c = (u - x0) / sx
        r = self.map.shape[0] - (v - y0) / sy
        if not (0 <= c < self.map.shape[1] and 0 <= r < self.map.shape[0]):
            return None
        return self.map.origin[:2] + self.map.rotation @ (np.array([c, r]) * self.map.resolution)

    def _circle(self, xy, radius, **kwargs):
        u, v = self.world_to_pixel(xy)
        self.canvas.create_oval(u-radius, v-radius, u+radius, v+radius, tags="overlay", **kwargs)

    def draw_overlay(self):
        if self.map_geometry is None:
            return
        self.canvas.delete("overlay")
        if self.walker is None:
            return
        w = self.walker
        self._draw_manual_overlay()
        if w.plan is not None:
            coords = [component for xy in w.plan.xy for component in self.world_to_pixel(xy)]
            if len(coords) >= 4:
                self.canvas.create_line(*coords, fill="#3a8de5", width=3, tags="overlay")
            self._circle(w.plan.endpoint, 6, outline="#f4bc43", width=3)
            self.canvas.create_line(*self.world_to_pixel(w.plan.endpoint), *self.world_to_pixel(w.plan.target),
                                    fill="#f4bc43", dash=(4, 4), tags="overlay")
        if self.target is not None:
            u, v = self.world_to_pixel(self.target)
            self.canvas.create_line(u-7, v-7, u+7, v+7, fill="#f35b60", width=3, tags="overlay")
            self.canvas.create_line(u-7, v+7, u+7, v-7, fill="#f35b60", width=3, tags="overlay")
        if self.last_frame is not None:
            self._circle(self.last_frame["pose"][:2], 5, outline="#15c7cc", width=2)
        _, _, sx, _ = self.map_geometry
        radius_px = max(5, self.args.radius / self.map.resolution * sx)
        self._circle(w.xy, radius_px, fill="#3275d3" if self.collision_var.get() else "#b73a53",
                     outline="#ffffff", width=1)
        front = w.xy + max(0.4, self.args.radius * 1.8) * np.array([math.cos(w.yaw), math.sin(w.yaw)])
        self.canvas.create_line(*self.world_to_pixel(w.xy), *self.world_to_pixel(front),
                                arrow="last", fill="#ffffff", width=2, tags="overlay")

    def mouse_move(self, event):
        xy = self.pixel_to_world(event.x, event.y)
        if xy is None:
            return
        cell = self.map.world_to_cell(xy)
        state = {-1: "unknown", 0: "candidate free", 100: "occupied"}[int(self.map.occupancy[cell])]
        self.coordinates.set(f"X={xy[0]:.3f}, Y={xy[1]:.3f} m | {state} | footprint-safe={bool(self.planner.walkable[cell])}")

    def map_click(self, event):
        self.root.focus_set()
        xy = self.pixel_to_world(event.x, event.y)
        if xy is None:
            return
        if self.click_mode.get() == "start":
            self.set_start(xy)
        else:
            self.x_var.set(f"{xy[0]:.4f}")
            self.y_var.set(f"{xy[1]:.4f}")
            self.plan_fields()

    def set_start(self, xy):
        self.pause()
        try:
            xy = np.asarray(xy, float)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError("Start coordinates must be finite X/Y values.")
            if self.collision_var.get():
                if not point_allowed(self.map, self.planner.walkable, xy, self.strict_var.get()):
                    raise PlanningError("Start violates the selected manual map check. Use a free cell or turn checks OFF for inspection.")
                point = self.map.cell_to_world(self.map.world_to_cell(xy))
            else:
                point = xy.copy()
            yaw = math.radians(float(self.heading_var.get()))
            if not math.isfinite(yaw):
                raise ValueError("Heading must be finite.")
            self.walker = Walker(point, yaw, self.args.speed, math.radians(self.args.turn_rate))
            self.target = None
            self.manual_trace.clear()
            self._remember_free_pose()
            self.message.set(f"Start set to X={point[0]:.3f}, Y={point[1]:.3f}. This is a reset/teleport, not a walked path.")
            self._send_pose(force=True)
            self.draw_overlay()
        except (ValueError, PlanningError) as exc:
            self.message.set(str(exc))

    def plan_fields(self):
        if self.walker is None:
            self.message.set("No valid start. Review the map and footprint settings first.")
            return
        # Stop manual/old automatic motion even if the new plan fails.
        self.pause()
        try:
            target = np.array([float(self.x_var.get()), float(self.y_var.get())])
            plan = self.planner.plan(self.walker.xy, target, mode=self.target_mode.get(),
                                     standoff=float(self.standoff_var.get()), max_approach=float(self.max_approach_var.get()))
            self.target = target
            self.walker.assign(plan)
            self.plan_count += 1
            with (self.session / f"plan_{self.plan_count:03d}.json").open("w") as stream:
                json.dump({"path_map_xy": plan.xy.tolist(), "target_map_xy": plan.target.tolist(),
                           "approach_map_xy": plan.endpoint.tolist(), "mode": plan.mode, "length_m": plan.length,
                           "note": plan.note}, stream, indent=2)
            self.message.set(f"Planned {plan.length:.2f} m. Stop=({plan.endpoint[0]:.2f}, {plan.endpoint[1]:.2f}). "
                             f"{plan.note} A* always uses the strict footprint map. Press Walk when ready.")
            self.draw_overlay()
        except (ValueError, PlanningError) as exc:
            self.walker.plan = None
            self.walker.state = "no route"
            self.target = None
            self.message.set(str(exc))
            self.draw_overlay()

    def walk(self):
        self._clear_manual_inputs()
        if self.walker is None:
            return
        try:
            self.walker.start()
        except PlanningError as exc:
            self.message.set(str(exc))

    def pause(self):
        self._clear_manual_inputs()
        if self.walker:
            self.walker.pause()

    def _space(self, event):
        if self._editing(event.widget):
            return
        self.pause()  # Space is a stop, never an accidental auto-resume.
        return "break"

    def apply_heading(self):
        if self.walker is None:
            return
        try:
            value = math.radians(float(self.heading_var.get()))
            if not math.isfinite(value):
                raise ValueError("Heading must be finite.")
            self.pause()
            self.walker.yaw = value
            self._send_pose(force=True)
            self.draw_overlay()
        except ValueError as exc:
            self.message.set(str(exc))

    def apply_camera(self):
        try:
            height, pitch = float(self.height_var.get()), float(self.pitch_var.get())
            if not np.isfinite([height, pitch]).all() or not 0.1 <= height <= 3 or not -80 <= pitch <= 80:
                raise ValueError("Camera height must be 0.1–3 m, pitch -80 to 80 degrees.")
            self.camera_height, self.pitch_deg = height, pitch
            self._send_pose(force=True)
        except ValueError as exc:
            self.message.set(str(exc))

    def _send_pose(self, force=False):
        if self.args.no_render or self.walker is None or not hasattr(self, "shared_pose"):
            return
        w = self.walker
        signature = tuple(round(float(v), 6) for v in (w.xy[0], w.xy[1], w.yaw, self.camera_height, self.pitch_deg))
        if not force and signature == self.last_signature:
            return
        self.last_signature = signature
        self.sequence += 1
        with self.shared_pose.get_lock():
            self.shared_pose[:] = [self.sequence, *signature, self.sim_time]

    def _show_frame(self, frame):
        self.last_frame = frame
        rgb = Image.fromarray(frame["rgb"])
        depth = frame["depth_m"]
        valid = np.isfinite(depth) & (depth > 0)
        gray = np.zeros(depth.shape, dtype=np.uint8)
        gray[valid] = np.rint(35 + 220 * (1 - np.clip(depth[valid] / self.args.depth_display_max, 0, 1))).astype(np.uint8)
        depth_img = Image.fromarray(gray)
        for label_widget, image, key in ((self.rgb_label, rgb, "rgb_photo"), (self.depth_label, depth_img, "depth_photo")):
            size = (max(180, label_widget.winfo_width() - 4), max(120, label_widget.winfo_height() - 4))
            image.thumbnail(size, Image.Resampling.BILINEAR)
            photo = ImageTk.PhotoImage(image)
            setattr(self, key, photo)
            label_widget.configure(image=photo, text="")
        p = frame["pose"]
        cp = frame["camera"]["map_pose"]
        self.render_message.set(f"RGB/depth from X={p[0]:.2f}, Y={p[1]:.2f}, yaw={math.degrees(p[2]):.1f}°, "
                                f"pitch={cp['pitch_deg']:.1f}°, height={cp['camera_height_m']:.2f} m | "
                                f"render={frame['render_seconds']:.2f} s | max request={self.args.render_hz:g} Hz. "
                                "Cyan marker is the IMAGE pose; it may lag the avatar.")

    def snapshot(self):
        if self.last_frame is None:
            self.message.set("No rendered frame yet. Map-only mode has no synthetic camera.")
            return
        self.snapshot_count += 1
        prefix = self.session / "snapshots" / f"snapshot_{self.snapshot_count:03d}"
        save_frame(self.last_frame, prefix)
        self.message.set(f"Saved RGB, raw depth, metric depth, alpha and exact camera pose: {prefix}")

    def _read_commands(self, now):
        if not self.args.command_file or now < self.next_command_check:
            return
        self.next_command_check = now + 0.5
        path = Path(self.args.command_file)
        if not path.exists():
            return
        stamp = path.stat().st_mtime_ns
        if stamp == self.last_command_stamp:
            return
        self.last_command_stamp = stamp
        try:
            command = json.loads(path.read_text())
            action = command["action"]
            if action == "goal":
                mode = command.get("mode", "object")
                if mode not in ("object", "point"):
                    raise ValueError("Command mode must be object or point.")
                self.x_var.set(str(command["x"]))
                self.y_var.set(str(command["y"]))
                self.target_mode.set(mode)
                self.plan_fields()
                if command.get("walk", False) and self.walker and self.walker.plan is not None:
                    self.walk()
            elif action == "start":
                self.heading_var.set(str(command.get("yaw_deg", 0)))
                self.set_start([command["x"], command["y"]])
            elif action == "pause":
                self.pause()
            else:
                raise ValueError("Unknown command action.")
        except (ValueError, KeyError, OSError) as exc:
            self.message.set(f"Command rejected: {exc}")

    def tick(self):
        if self.closed:
            return
        now = time.monotonic()
        dt = min(max(now - self.last_tick, 0), 0.1)
        self.last_tick = now
        self.sim_time += dt
        self._manual_update(dt)
        if self.walker:
            previous = self.walker.state
            self.walker.update(dt)
            if previous != "arrived" and self.walker.state == "arrived":
                self.message.set("Arrived and stopped. Object mode faces the target; inspect RGB to confirm actual visibility.")
            p = self.walker
            map_status, _, footprint_free = self.cell_status(p.xy)
            self.motion_text.set(f"{p.state.upper()} | X {p.xy[0]:.2f}, Y {p.xy[1]:.2f}, "
                                 f"yaw {math.degrees(p.yaw):.1f}°, pitch {self.pitch_deg:.1f}° | {map_status}")
            self._send_pose()
            if self.sim_time - self.last_log >= 0.1:
                self.csv.writerow([round(self.sim_time, 4), *p.xy.tolist(), p.yaw, p.state,
                                   self.camera_height, self.pitch_deg, int(self.collision_var.get()),
                                   int(self.strict_var.get()), map_status, int(footprint_free),
                                   p.speed, math.degrees(p.turn_rate)])
                self.last_log = self.sim_time
                self.log_file.flush()
        if self.worker:
            try:
                while True:
                    kind, text = self.statuses.get_nowait()
                    if kind == "error":
                        self.render_message.set("Renderer failed; see terminal traceback. Navigation is still usable in map-only mode.")
                    elif kind != "stopped":
                        self.render_message.set(text)
            except queue.Empty:
                pass
            frame = None
            try:
                while True:
                    frame = self.frames.get_nowait()
            except queue.Empty:
                pass
            if frame is not None:
                self._show_frame(frame)
        self._read_commands(now)
        self.draw_overlay()
        self._tick_job = self.root.after(33, self.tick)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._clear_manual_inputs()
        for job in (getattr(self, "_tick_job", None), getattr(self, "_focus_job", None)):
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except self.tk.TclError:
                    pass
        if self.worker:
            self.stop_event.set()
            self.worker.join(timeout=1.0)
            if self.worker.is_alive():
                self.worker.terminate()
                self.worker.join(timeout=1.0)
        self.log_file.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def argument_parser():
    parser = argparse.ArgumentParser(description="Standalone map navigation + persistent gsplat RGB/depth camera.")
    parser.add_argument("--map", default="outputs/ros/valmet_map.yaml", help="ROS trinary YAML map, loaded directly; ROS need not run")
    parser.add_argument("--scene", default="data/Industrial Boiler Room.ply", help="ORIGINAL Gaussian PLY, not the ROS point cloud")
    parser.add_argument("--no-render", action="store_true", help="Map/planner/UI only; no CUDA imports")
    parser.add_argument("--check-map", action="store_true", help="Print map/clearance diagnostic and exit without GUI/CUDA")
    parser.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"), help="Starting position in existing ROS-map coordinates")
    parser.add_argument("--yaw-deg", type=float, default=90.0)
    parser.add_argument("--scale", type=float, default=1.43)
    parser.add_argument("--floor-y", type=float, default=1.1)
    parser.add_argument("--camera-height", type=float, default=1.7, help="Camera height above the flat assumed floor, metres")
    parser.add_argument("--pitch-deg", type=float, default=0.0, help="Positive looks up; negative looks down")
    parser.add_argument("--radius", type=float, default=0.25, help="Circular avatar footprint radius, metres")
    parser.add_argument("--margin", type=float, default=0.03, help="ADDITIONAL clearance; input map inflation is preserved")
    parser.add_argument("--standoff", type=float, default=0.8, help="Minimum target-point to stopping-point distance")
    parser.add_argument("--max-approach", type=float, default=2.5)
    parser.add_argument("--speed", type=float, default=0.25, help="Translation speed, metres/second")
    parser.add_argument("--turn-rate", type=float, default=60.0, help="Turning speed, degrees/second")
    parser.add_argument("--look-rate", type=float, default=25.0, help="Head pitch rate while W/S or look buttons held, degrees/second")
    parser.add_argument("--noclip", action="store_true", help="Start with manual map checks OFF; inspection only")
    parser.add_argument("--manual-strict", action="store_true", help="Manual checks also use the original footprint-clearance mask")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fov", type=float, default=70.0, help="Vertical field of view, degrees")
    parser.add_argument("--render-hz", type=float, default=1.0, help="Maximum requested rendering rate, not a guaranteed FPS")
    parser.add_argument("--sh-degree", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--alpha-min", type=float, default=0.5)
    parser.add_argument("--depth-display-max", type=float, default=8.0)
    parser.add_argument("--output", default="outputs/avatar_demo")
    parser.add_argument("--record", action="store_true", help="Save every rendered RGB/depth/pose frame; consumes disk space")
    parser.add_argument("--command-file", default=None, help="Optional atomic JSON command file polled by the UI")
    parser.add_argument("--gui-test-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    numbers = [args.scale, args.floor_y, args.camera_height, args.pitch_deg, args.radius, args.margin,
               args.standoff, args.max_approach, args.speed, args.turn_rate, args.fov, args.render_hz,
               args.alpha_min, args.depth_display_max, args.yaw_deg, args.look_rate]
    if not np.isfinite(numbers).all():
        parser.error("Numeric arguments must be finite.")
    if not (args.scale > 0 and 0.1 <= args.camera_height <= 3 and abs(args.pitch_deg) <= 80
            and args.radius >= 0 and args.margin >= 0 and 0 < args.standoff <= args.max_approach
            and 0 < args.speed <= 2 and 0 < args.turn_rate <= 180 and 0 < args.look_rate <= 180 and 1 < args.fov < 179 and args.render_hz > 0
            and args.width > 0 and args.height > 0 and 0 <= args.alpha_min <= 1 and args.depth_display_max > 0):
        parser.error("Invalid dimensions, clearance, camera, motion or approach parameters.")
    floor_map = FloorMap.load(args.map)
    planner = Planner(floor_map, args.radius, args.margin)
    print(f"Map {floor_map.shape[1]} x {floor_map.shape[0]}, resolution={floor_map.resolution:g} m, origin={floor_map.origin.tolist()}")
    print(f"Known-free cells: {int((floor_map.occupancy == 0).sum())}; footprint-safe: {int(planner.walkable.sum())}; components: {planner.component_count}")
    if args.check_map:
        try:
            point = planner.set_start(args.start) if args.start is not None else planner.auto_start()
            print(f"Candidate start in map frame: {point.tolist()}; not verified against the scene")
        except PlanningError as exc:
            print(f"WARNING: {exc}")
        return
    if not args.no_render and not Path(args.scene).expanduser().is_file():
        parser.error(f"Scene not found: {args.scene}. Use --no-render for map-only testing.")
    if not args.no_render:
        args.scene = str(Path(args.scene).expanduser().resolve())
    app = AvatarApp(args, floor_map, planner)
    try:
        app.run()
    except KeyboardInterrupt:
        app.close()


if __name__ == "__main__":
    main()
