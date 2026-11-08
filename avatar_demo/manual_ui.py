"""Manual controls mixed into the existing avatar UI; renderer and source maps stay unchanged."""
from __future__ import annotations

import json
import math
import time
from collections import deque

import numpy as np

from .core import PlanningError, Walker, angle_wrap
from .manual import HeldControls, manual_step, point_allowed, tilted_pitch
from .render_worker import save_frame


class ManualControlsMixin:
    KEY_ACTIONS = {"Up": "forward", "Down": "backward", "Left": "left", "Right": "right",
                   "w": "look_up", "s": "look_down"}
    EDIT_CLASSES = {"Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox"}

    def _init_manual(self):
        tk = self.tk
        self.held = HeldControls()
        self.pending_keys = {}
        self.suppressed_keys = set()
        self.hold_buttons = {}
        self.last_free_pose = None
        self.manual_trace = deque(maxlen=2000)
        self.review_points = []
        self.bookmark_count = 0
        self.collision_var = tk.BooleanVar(value=not self.args.noclip)
        self.strict_var = tk.BooleanVar(value=self.args.manual_strict)
        self.speed_var = tk.StringVar(value=str(self.args.speed))
        self.turn_var = tk.StringVar(value=str(self.args.turn_rate))
        self.guard_text = tk.StringVar(value="")
        self.note_var = tk.StringVar(value="")
        self.review_var = tk.StringVar(value="uncertain")
        self.last_block_reason = ""
        self._focus_job = None

    def _hold_button(self, parent, text, action):
        # A mouse button press is a held input, not an auto-repeated Button command.
        button = self.tk.Button(parent, text=text, takefocus=False, padx=7, pady=3)
        source = "mouse:" + action
        self.hold_buttons[source] = button
        def press(event):
            self.root.focus_set()
            button.configure(relief="sunken")
            self._press_action(source, action)
            return "break"
        def release(event=None):
            self.held.release(source)
            button.configure(relief="raised")
            return "break"
        button.bind("<ButtonPress-1>", press)
        button.bind("<ButtonRelease-1>", release)
        button.bind("<Leave>", release)
        return button

    def _make_manual_controls(self):
        ttk = self.ttk
        row = ttk.Frame(self.root, padding=(6, 3))
        row.pack(fill="x")
        ttk.Label(row, text="Drive (hold):").pack(side="left")
        for text, action in [("Forward ↑", "forward"), ("Backward ↓", "backward"),
                             ("Turn left ←", "left"), ("Turn right →", "right")]:
            self._hold_button(row, text, action).pack(side="left", padx=2)
        ttk.Button(row, text="STOP / Esc", command=self.pause).pack(side="left", padx=(5, 12))
        ttk.Label(row, text="Head:").pack(side="left")
        self._hold_button(row, "Look up · W", "look_up").pack(side="left", padx=2)
        self._hold_button(row, "Look down · S", "look_down").pack(side="left", padx=2)
        ttk.Button(row, text="Level · H", command=lambda: self.set_pitch(0)).pack(side="left", padx=2)
        ttk.Button(row, text="Ceiling +45°", command=lambda: self.set_pitch(45)).pack(side="left", padx=2)
        ttk.Button(row, text="Floor −45°", command=lambda: self.set_pitch(-45)).pack(side="left", padx=2)

        row = ttk.Frame(self.root, padding=(6, 2))
        row.pack(fill="x")
        ttk.Checkbutton(row, text="Restrict manual movement to free cells", variable=self.collision_var,
                        command=self.guard_changed).pack(side="left")
        ttk.Checkbutton(row, text="Also require body clearance", variable=self.strict_var,
                        command=self.guard_changed).pack(side="left", padx=7)
        ttk.Button(row, text="Return to free", command=self.return_to_free).pack(side="left", padx=5)
        ttk.Label(row, text="Speed m/s").pack(side="left", padx=(10, 2))
        ttk.Entry(row, textvariable=self.speed_var, width=5).pack(side="left")
        ttk.Label(row, text="Turn °/s").pack(side="left", padx=(6, 2))
        ttk.Entry(row, textvariable=self.turn_var, width=5).pack(side="left")
        ttk.Button(row, text="Apply speeds", command=self.apply_speeds).pack(side="left", padx=5)
        self.guard_banner = self.tk.Label(self.root, textvariable=self.guard_text, anchor="w", padx=6, pady=3)
        self.guard_banner.pack(fill="x")
        self._update_guard_banner()

    def _make_review_controls(self):
        row = self.ttk.Frame(self.root, padding=(6, 3))
        row.pack(fill="x")
        self.ttk.Label(row, text="Bookmark current position:").pack(side="left")
        self.ttk.Combobox(row, textvariable=self.review_var, state="readonly", width=13,
                          values=["uncertain", "looks_free", "obstacle", "boundary"]).pack(side="left", padx=4)
        self.ttk.Label(row, text="Note").pack(side="left", padx=4)
        self.ttk.Entry(row, textvariable=self.note_var, width=45).pack(side="left", fill="x", expand=True)
        self.ttk.Button(row, text="Save bookmark + view", command=self.bookmark).pack(side="left", padx=5)

    def _bind_manual_controls(self):
        self.root.bind("<KeyPress>", self._key_press, add="+")
        self.root.bind("<KeyRelease>", self._key_release, add="+")
        self.root.bind("<FocusOut>", self._focus_changed, add="+")
        self.root.bind("<FocusIn>", self._focus_changed, add="+")
        self.root.bind("<Unmap>", self._window_unmapped, add="+")
        self.root.bind("<ButtonRelease-1>", self._release_mouse, add="+")
        # Clicking an image gives keyboard control without changing start/goal.
        self.rgb_label.bind("<Button-1>", lambda event: self.root.focus_set())
        self.depth_label.bind("<Button-1>", lambda event: self.root.focus_set())
        self._remember_free_pose()
        self.root.after_idle(self.root.focus_set)

    def _editing(self, widget=None):
        widget = widget if widget is not None else self.root.focus_get()
        return widget is not None and widget.winfo_class() in self.EDIT_CLASSES

    @staticmethod
    def _key_name(event):
        return event.keysym.lower() if len(event.keysym) == 1 else event.keysym

    def _key_press(self, event):
        if self._editing(event.widget):
            return None
        key = self._key_name(event)
        if key in self.KEY_ACTIONS:
            # X11 autorepeat can synthesize release/press pairs; cancel a deferred release.
            job = self.pending_keys.pop(key, None)
            if job is not None:
                self.root.after_cancel(job)
            if key in self.suppressed_keys:
                return "break"  # A STOP stays stopped until this key is truly released.
            self._press_action("key:" + key, self.KEY_ACTIONS[key])
            return "break"
        if key == "h":
            self.set_pitch(0)
            return "break"
        return None

    def _key_release(self, event):
        key = self._key_name(event)
        if key not in self.KEY_ACTIONS:
            return None
        job = self.pending_keys.pop(key, None)
        if job is not None:
            self.root.after_cancel(job)
        def release():
            self.pending_keys.pop(key, None)
            self.held.release("key:" + key)
            self.suppressed_keys.discard(key)
        self.pending_keys[key] = self.root.after(35, release)
        return "break" if not self._editing(event.widget) else None

    def _release_mouse(self, event=None):
        for source, button in self.hold_buttons.items():
            self.held.release(source)
            button.configure(relief="raised")

    def _clear_manual_inputs(self):
        self.suppressed_keys.update(source[4:] for source in self.held.sources if source.startswith("key:"))
        self.held.clear()
        for job in self.pending_keys.values():
            try:
                self.root.after_cancel(job)
            except self.tk.TclError:
                pass
        self.pending_keys.clear()
        self._release_mouse()
        self.last_block_reason = ""

    def _focus_changed(self, event=None):
        if self.closed:
            return
        if self._focus_job is not None:
            self.root.after_cancel(self._focus_job)
        # Defer until focus transfer completes, including transfers between child widgets.
        self._focus_job = self.root.after_idle(self._check_focus)

    def _check_focus(self):
        self._focus_job = None
        if self.closed:
            return
        focus = self.root.focus_get()
        if focus is None:
            self.pause()  # Do not continue walking when the application loses focus.
        elif self._editing(focus):
            self._clear_manual_inputs()  # Typing in coordinates must not drive the avatar.

    def _window_unmapped(self, event):
        if event.widget == self.root:
            self.pause()

    def _press_action(self, source, action):
        already = source in self.held.sources
        if action in ("forward", "backward", "left", "right"):
            if self.walker is None:
                self.message.set("Set a starting position first.")
                return
            if not already:
                # Old routes must never be resumed from a manually changed position.
                if self.walker.plan is not None or self.walker.running:
                    self.message.set("Manual control took over; previous route cleared. Plan again for automatic walking.")
                self.walker.pause()
                self.walker.plan = None
                self.walker.index = 0
                self.walker.state = "manual"
        self.held.press(source, action)

    def _manual_update(self, dt):
        forward, turn, look = self.held.axes()
        if look:
            self.pitch_deg = tilted_pitch(self.pitch_deg, look, dt, self.args.look_rate)
            self.pitch_var.set(f"{self.pitch_deg:.1f}")
        if self.walker is None:
            return
        w = self.walker
        if forward or turn:
            result = manual_step(w.xy, w.yaw, forward, turn, dt, w.speed, w.turn_rate,
                                 self.map, self.planner.walkable,
                                 collision=self.collision_var.get(), strict=self.strict_var.get())
            w.xy, w.yaw = result.xy, result.yaw
            w.running = False
            w.state = "manual blocked" if result.blocked else "manual"
            if result.blocked and result.reason != self.last_block_reason:
                self.message.set("Manual translation stopped: " + result.reason +
                                 ". Head tilt and turning remain available; no map cells are changed.")
            self.last_block_reason = result.reason if result.blocked else ""
        elif w.state in ("manual", "manual blocked"):
            w.state = "manual idle"
            self.last_block_reason = ""
        self._remember_free_pose()
        if w.state.startswith("manual"):
            guard = bool(self.collision_var.get())
            strict = bool(self.strict_var.get())
            if (not self.manual_trace or np.linalg.norm(w.xy - self.manual_trace[-1][0]) >= .06
                    or (guard, strict) != self.manual_trace[-1][1:]):
                self.manual_trace.append((w.xy.copy(), guard, strict))
        if (forward or turn) and not self._editing():
            self.heading_var.set(f"{math.degrees(w.yaw):.1f}")

    def _remember_free_pose(self):
        if self.walker is not None and point_allowed(self.map, self.planner.walkable, self.walker.xy,
                                                     self.strict_var.get()):
            self.last_free_pose = (self.walker.xy.copy(), self.walker.yaw)

    def _update_guard_banner(self):
        if self.collision_var.get():
            if self.strict_var.get():
                text = "MANUAL MAP CHECKS ON: existing footprint-free area only (green). Not a 3-D/head-clearance guarantee."
            else:
                text = "MANUAL MAP CHECKS ON: input free cells allowed (green + amber); occupied/unknown blocked. POINT-CENTRE CHECK ONLY."
            self.guard_banner.configure(bg="#dcebdc", fg="#17391f")
        else:
            text = "FREE-ROAM INSPECTION — MAP CHECKS OFF: may cross equipment, walls, unknown and outside the map. NOT A VALID WALKABLE ROUTE."
            self.guard_banner.configure(bg="#a32f2f", fg="white")
        self.guard_text.set(text)

    def guard_changed(self):
        self.pause()
        self._update_guard_banner()
        if self.walker and self.collision_var.get():
            if not point_allowed(self.map, self.planner.walkable, self.walker.xy, self.strict_var.get()):
                self.message.set("Map checks enabled while outside the allowed area. Translation is blocked; use Return to free or turn checks OFF to move back.")
            else:
                self._remember_free_pose()
                self.message.set("Manual map checks enabled. Automatic A* still uses the original strict footprint map.")
        elif self.walker:
            self.message.set("Free-roam is for visual inspection only. It never edits the source map or proves traversability.")
        self.draw_overlay()

    def return_to_free(self):
        self.pause()
        try:
            if (self.last_free_pose is not None and
                    point_allowed(self.map, self.planner.walkable, self.last_free_pose[0], self.strict_var.get())):
                xy, yaw = self.last_free_pose
            else:
                xy, yaw = self.planner.auto_start(), math.radians(self.args.yaw_deg)
            self.walker = Walker(xy, yaw, self.args.speed, math.radians(self.args.turn_rate))
            self.target = None
            self._remember_free_pose()
            self._send_pose(force=True)
            self.message.set("Returned to a map-allowed position by explicit TELEPORT/reset, not a navigated path. Guard toggle is unchanged.")
            self.draw_overlay()
        except PlanningError as exc:
            self.message.set(str(exc))

    def apply_speeds(self):
        try:
            speed, turn = float(self.speed_var.get()), float(self.turn_var.get())
            if not np.isfinite([speed, turn]).all() or not (0 < speed <= 2 and 0 < turn <= 180):
                raise ValueError("Speed must be >0 and <=2 m/s; turn rate >0 and <=180 deg/s.")
            self.pause()
            self.args.speed, self.args.turn_rate = speed, turn
            if self.walker:
                self.walker.speed, self.walker.turn_rate = speed, math.radians(turn)
            self.message.set(f"Speeds applied: {speed:g} m/s, {turn:g} deg/s (manual and automatic).")
        except ValueError as exc:
            self.message.set(str(exc))

    def set_pitch(self, value):
        self.pitch_deg = float(np.clip(value, -80, 80))
        self.pitch_var.set(f"{self.pitch_deg:.1f}")
        # Presets do not stop a planned walk or change camera position.
        for source, action in list(self.held.sources.items()):
            if action in ("look_up", "look_down"):
                self.held.release(source)
        self._send_pose(force=True)

    def cell_status(self, xy):
        cell = self.map.world_to_cell(xy)
        if not self.map.contains(cell):
            return "outside_map", None, False
        value = int(self.map.occupancy[cell])
        status = {-1: "unknown", 0: "candidate_free", 100: "occupied"}.get(value, "other")
        return status, value, bool(self.planner.walkable[cell])

    def _draw_manual_overlay(self):
        trace = list(self.manual_trace)
        # Draw a few polylines rather than thousands of separate Tk items.
        run, previous_color = [], None
        def flush():
            if len(run) >= 4:
                self.canvas.create_line(*run, fill=previous_color, width=2,
                                        dash=(2, 3), tags="overlay")
        for a, b in zip(trace[:-1], trace[1:]):
            if np.linalg.norm(b[0] - a[0]) > .5:  # Never draw a reset as travel.
                flush()
                run, previous_color = [], None
                continue
            color = "#58b6aa" if a[1] and b[1] else "#e47d9a"
            if color != previous_color:
                flush()
                run = list(self.world_to_pixel(a[0]))
                previous_color = color
            run.extend(self.world_to_pixel(b[0]))
        flush()
        for number, xy, kind in self.review_points:
            u, v = self.world_to_pixel(xy)
            self.canvas.create_rectangle(u-3, v-3, u+3, v+3, outline="#f7e69c", tags="overlay")
            self.canvas.create_text(u+5, v-7, text=str(number), fill="#f7e69c", anchor="w", tags="overlay")

    def bookmark(self):
        if self.walker is None:
            self.message.set("No current avatar position to bookmark.")
            return
        self.pause()
        self.bookmark_count += 1
        folder = self.session / "bookmarks" / f"bookmark_{self.bookmark_count:03d}"
        folder.mkdir(parents=True, exist_ok=False)
        w = self.walker
        status, value, strict_free = self.cell_status(w.xy)
        cell = self.map.world_to_cell(w.xy)
        item = {
            "id": self.bookmark_count, "wall_time": time.time(), "simulation_time_s": self.sim_time,
            "assessment": self.review_var.get(), "note": self.note_var.get(),
            "map_xy_m": w.xy.tolist(), "map_cell_row_col": list(cell),
            "yaw_rad": w.yaw, "yaw_deg": math.degrees(w.yaw),
            "camera_height_m": self.camera_height, "camera_pitch_deg": self.pitch_deg,
            "scene_camera_xyz": [w.xy[0]/self.args.scale,
                                 self.args.floor_y-self.camera_height/self.args.scale, w.xy[1]/self.args.scale],
            "scene_scale_assumed": self.args.scale, "floor_y_scene": self.args.floor_y,
            "manual_collision_check": bool(self.collision_var.get()),
            "manual_body_clearance": bool(self.strict_var.get()),
            "map_status": status, "input_occupancy": value, "footprint_candidate_free": strict_free,
            "scope": "User observation at CURRENT avatar location, not an automatically verified free cell or a clicked image object.",
            "map_edited": False,
        }
        if self.last_frame is not None:
            save_frame(self.last_frame, folder / "view")
            p = self.last_frame["pose"]
            camera = self.last_frame["camera"]
            item["displayed_frame"] = {
                "relative_prefix": str((folder / "view").relative_to(self.session)),
                "camera": camera,
                "position_difference_m": float(np.linalg.norm(w.xy - p[:2])),
                "yaw_difference_deg": math.degrees(angle_wrap(w.yaw - p[2])),
                "pitch_difference_deg": self.pitch_deg - camera["map_pose"]["pitch_deg"],
                "height_difference_m": self.camera_height - camera["map_pose"]["camera_height_m"],
                "note": "May lag the current avatar. Exact frame pose is in view_camera.json; do not assume they coincide.",
            }
        else:
            item["displayed_frame"] = None
        (folder / "bookmark.json").write_text(json.dumps(item, indent=2))
        with (self.session / "review_points.jsonl").open("a") as stream:
            stream.write(json.dumps(item) + "\n")
        self.review_points.append((self.bookmark_count, w.xy.copy(), self.review_var.get()))
        self.message.set(f"Bookmark {self.bookmark_count} saved at CURRENT position ({w.xy[0]:.3f}, {w.xy[1]:.3f}). "
                         "Image may be from an earlier pose; both poses saved. Source map unchanged.")
        self.draw_overlay()
