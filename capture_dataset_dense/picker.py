"""Tk station/corridor selector; no GPU initialization or navigation-map edits."""
from __future__ import annotations
import copy
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from PIL import ImageDraw, ImageTk

from .clearance import clearance_layers, pin_clearance, required_clearance

from .planning import (HEADER, FloorMap, choose_stations, choose_pilots, create_dense_plan,
                       preview, pixel_to_world, world_to_pixel, save_dense_plan, validate_segment)


class DensePicker:
    def __init__(self, root, args, cfg, selection):
        self.root, self.args, self.cfg = root, args, cfg
        self.fmap = FloorMap.load(args.map)
        self.clearance_layers = clearance_layers(self.fmap)
        self.audit = {}
        self.selection = copy.deepcopy(selection)
        self.draft, self.draft_allow = [], False
        self.stations, self.saved = [], False
        self.allow_unknown = tk.BooleanVar(value=False)
        self.mode = tk.StringVar(value='pin')
        self.status = tk.StringVar()
        self.hover_text = tk.StringVar(value='Coordinates use the existing ROS map frame, with provisional scale.')
        root.title('Valmet | Dense camera coverage + capture-only corridor')
        ttk.Label(root, text='Pin mode: click individual cameras. Corridor mode: click a centerline with bends, then Finish corridor.').pack(anchor='w',padx=8,pady=4)
        bar=ttk.Frame(root);bar.pack(fill='x',padx=8)
        ttk.Radiobutton(bar,text='Pin cameras',variable=self.mode,value='pin').pack(side='left')
        ttk.Radiobutton(bar,text='Draw corridor',variable=self.mode,value='corridor').pack(side='left')
        ttk.Checkbutton(bar,text='Allow GRAY for capture ONLY',variable=self.allow_unknown).pack(side='left',padx=10)
        ttk.Button(bar,text='Finish corridor',command=self.finish_corridor).pack(side='left',padx=3)
        ttk.Button(bar,text='Undo corridor point/line',command=self.undo_corridor).pack(side='left',padx=3)
        bar2=ttk.Frame(root);bar2.pack(fill='x',padx=8,pady=3)
        ttk.Label(bar2,text='Right-click: undo draft or remove pin. Shift-click: exclude disk.').pack(side='left')
        ttk.Button(bar2,text='Undo exclusion',command=self.undo_exclusion).pack(side='left',padx=4)
        ttk.Button(bar2,text='Recompute',command=self.refresh).pack(side='left',padx=4)
        ttk.Button(bar2,text='SAVE PLAN + CLOSE',command=self.save).pack(side='left',padx=6)
        self.canvas=tk.Canvas(root,highlightthickness=0,background='#20252e')
        self.canvas.pack(padx=8,pady=4)
        ttk.Label(root,textvariable=self.hover_text).pack(fill='x',padx=8)
        ttk.Label(root,textvariable=self.status,wraplength=min(1150,root.winfo_screenwidth()-60)).pack(fill='x',padx=8,pady=5)
        self.canvas.bind('<Button-1>',self.click)
        self.canvas.bind('<Button-3>',self.remove)
        self.canvas.bind('<Motion>',self.hover)
        root.protocol('WM_DELETE_WINDOW',self.close)
        self.refresh()

    def locate(self,event):
        if event.y < HEADER or not 0 <= event.x < self.size[0] or event.y >= self.size[1]+HEADER:
            return None
        xy=pixel_to_world(self.fmap,[event.x,event.y-HEADER],self.size)
        cell=self.fmap.world_to_cell(xy)
        return (xy,cell) if self.fmap.contains(cell) else None

    def click(self,event):
        loc=self.locate(event)
        if loc is None:return
        xy,cell=loc
        if event.state & 0x0001:
            self.selection['excluded_capture_disks'].append({'x':float(xy[0]),'y':float(xy[1]),'radius':.4})
            self.refresh();return
        state=int(self.fmap.occupancy[cell])
        if state==100:
            self.status.set('Occupied: not accepted. Capture tool never reclassifies these cells. Choose a different centerline.');return
        if state==-1 and not self.allow_unknown.get():
            self.status.set('Enable "Allow GRAY for capture ONLY" before selecting this gray location.');return
        xy=self.fmap.cell_to_world(cell)
        if self.mode.get()=='corridor':
            if self.draft and np.linalg.norm(xy-np.asarray(self.draft[-1]))<1e-8:return
            allow=self.draft_allow or self.allow_unknown.get()
            if self.draft:
                try:validate_segment(self.fmap,self.draft[-1],xy,allow)
                except ValueError as exc:self.status.set(str(exc));return
            self.draft.append([float(xy[0]),float(xy[1])]);self.draft_allow=allow
            self.refresh()
            self.status.set(f'{len(self.draft)} corridor vertices. Continue along the passage; press Finish corridor to allocate dense samples.')
            return
        # Explicit pins cannot bypass the buffer. Corridors are filtered per sample.
        _, check = pin_clearance(self.fmap, {'x':xy[0], 'y':xy[1], 'allow_unknown':state==-1},
                                 self.clearance_layers, required_clearance(self.cfg))
        if not check['accepted']:
            self.status.set(f"Too close to wall/occupied boundary: conservative clearance {check['wall_clearance_lower_bound_m']:.2f} m "
                            f"< required {check['required_clearance_m']:.2f} m. Pin farther inward.")
            return
        pins=self.selection['pins']
        if any(self.fmap.world_to_cell([p['x'],p['y']])==cell for p in pins):return
        pins.append({'x':float(xy[0]),'y':float(xy[1]),'allow_unknown':state==-1,
                     'note':'User-selected stationary capture, not a verified navigation location.'})
        self.refresh()

    def finish_corridor(self):
        if len(self.draft)<2:
            self.status.set('Draw at least two corridor vertices before Finish corridor.');return
        item={'points':copy.deepcopy(self.draft),'allow_unknown':self.draft_allow,
              'note':'Explicit user-defined capture centerline, not a walked or validated route.'}
        proposed=copy.deepcopy(self.selection);proposed['corridors'].append(item)
        try:choose_stations(self.fmap,proposed,self.cfg)
        except ValueError as exc:self.status.set(str(exc));return
        self.selection=proposed;self.draft=[];self.draft_allow=False
        self.refresh()

    def undo_corridor(self):
        if self.draft:
            self.draft.pop()
            if not self.draft:self.draft_allow=False
        elif self.selection['corridors']:
            self.selection['corridors'].pop()
        self.refresh()

    def remove(self,event):
        if self.draft:self.undo_corridor();return
        loc=self.locate(event)
        pins=self.selection['pins']
        if loc is None or not pins:return
        xy,_=loc
        dd=np.array([np.linalg.norm(xy-[p['x'],p['y']]) for p in pins])
        if dd.min()<=.65:pins.pop(int(dd.argmin()));self.refresh()

    def undo_exclusion(self):
        if self.selection['excluded_capture_disks']:
            self.selection['excluded_capture_disks'].pop();self.refresh()

    def hover(self,event):
        loc=self.locate(event)
        if loc is None:return
        xy,cell=loc
        state={0:'known-free',-1:'UNKNOWN',100:'occupied'}[int(self.fmap.occupancy[cell])]
        near=''
        if self.stations:
            d=np.array([np.linalg.norm(xy-[p['x'],p['y']]) for p in self.stations]);s=self.stations[int(d.argmin())]
            if d.min()<.3:near=f" | nearest {s['station_id']} ({s['selection']})"
        self.hover_text.set(f'X={xy[0]:.3f}, Y={xy[1]:.3f} provisional m | {state} | wall buffer >= {self.clearance_layers["wall_m"][cell]:.2f} m | required {required_clearance(self.cfg):.2f} m{near}')

    def refresh(self):
        error=None
        self.audit = {}
        try:self.stations=choose_stations(self.fmap,self.selection,self.cfg,diagnostics=self.audit)
        except ValueError as exc:error=str(exc);self.stations=[]
        image=preview(self.fmap,self.stations,self.selection,
                      choose_pilots(self.stations,self.cfg['pilot_positions']),
                      max_size=(min(1180,self.root.winfo_screenwidth()-70),
                                max(180,self.root.winfo_screenheight()-330)),
                      cfg=self.cfg, diagnostics=self.audit)
        self.size=(image.width,image.height-HEADER)
        if self.draft:
            d=ImageDraw.Draw(image);pts=[]
            for xy in self.draft:
                x,y=world_to_pixel(self.fmap,xy,self.size);pts.append((x,y+HEADER))
            if len(pts)>1:d.line(pts,fill='#ffffff',width=3)
            for x,y in pts:d.ellipse((x-4,y-4,x+4,y+4),fill='#ffffff',outline='#08afa7')
        self.photo=ImageTk.PhotoImage(image)
        self.canvas.configure(width=image.width,height=image.height)
        self.canvas.delete('all');self.canvas.create_image(0,0,image=self.photo,anchor='nw')
        n=len(self.stations);nc=sum(bool(s.get('corridor_id')) for s in self.stations)
        nu=sum(s['unknown_capture_only'] for s in self.stations)
        self.status.set(error or f"{n}/{self.cfg['requested_positions']} stations | {nc} on corridor lines | {nu} gray capture-only | "
                        f"{self.cfg['requested_frames']} total views, heights {self.cfg['height_range_m'][0]:.1f}-{self.cfg['height_range_m'][1]:.1f} m. "
                        f"Buffer {required_clearance(self.cfg):.2f} m; {self.audit.get('rejected_count',0)} selected samples skipped. "
                        'No 3-D clearance check. Original map remains unchanged.')

    def save(self):
        if self.draft:
            self.status.set('Finish the draft corridor or undo its vertices before saving.');return
        try:
            self.stations=choose_stations(self.fmap,self.selection,self.cfg)
            plan=create_dense_plan(self.args.map,self.args.scene,self.stations,self.selection,self.cfg)
            save_dense_plan(self.args.out,plan,self.selection)
            self.saved=True
            from .__main__ import print_summary
            print_summary(plan);self.root.destroy()
        except Exception as exc:messagebox.showerror('Cannot save dense capture plan',str(exc))

    def close(self):
        if messagebox.askyesno('Discard unsaved edits?','Close without saving this new capture plan?'):
            self.root.destroy()


def run_picker(args,cfg,selection):
    root=tk.Tk();DensePicker(root,args,cfg,selection);root.mainloop()
