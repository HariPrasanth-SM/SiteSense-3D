"""Small Tk region/priority editor for camera spawning only."""
from __future__ import annotations
import copy
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import ImageDraw, ImageTk

from .core import generate, preview, save_results


class Editor:
    def __init__(self, root, fmap, region, args):
        self.root, self.fmap, self.region, self.args = root, fmap, copy.deepcopy(region), args
        self.records, self.report, self.layers = [], None, None
        self.draft, self.saved = [], False
        self.mode = tk.StringVar(value='outline' if not region['polygon'] else 'corridor')
        self.status = tk.StringVar(value='Draw a room outline around the main room INCLUDING the gray passage; then Finish shape.')
        self.hover = tk.StringVar(value='Units: provisional map metres. Dark cells always blocked; gray is allowed inside your outline.')
        root.title('Valmet | Camera spawns only | soft clearance, narrow passages retained')
        ttk.Label(root, text='1. Outline room to exclude exterior gray noise.  2. Mark a priority corridor (optional).  '
                            '3. Generate.  4. Save.', wraplength=1050).pack(anchor='w', padx=8, pady=6)
        bar = ttk.Frame(root); bar.pack(fill='x', padx=8)
        for name, key in [('Room outline', 'outline'), ('Priority corridor', 'corridor'), ('Pin camera','pin')]:
            ttk.Radiobutton(bar, text=name, value=key, variable=self.mode, command=self.change_mode).pack(side='left', padx=4)
        ttk.Button(bar, text='Finish shape', command=self.finish).pack(side='left', padx=4)
        ttk.Button(bar, text='Undo', command=self.undo).pack(side='left', padx=4)
        ttk.Button(bar, text='Generate points', command=self.generate).pack(side='left', padx=4)
        ttk.Button(bar, text='SAVE + CLOSE', command=self.save).pack(side='left', padx=4)
        ttk.Label(root, text=f'{args.positions} stations | preferred clearance {args.preferred_clearance:.2f} m (NOT mandatory) | '
                            f'max local shift {args.max_shift:.2f} m | right-click undo | Shift-click exclude 0.30 m disk').pack(anchor='w', padx=8, pady=4)
        self.canvas=tk.Canvas(root, highlightthickness=0, background='#20252e')
        self.canvas.pack(padx=8, pady=4)
        ttk.Label(root, textvariable=self.hover, wraplength=1100).pack(anchor='w', padx=8)
        ttk.Label(root, textvariable=self.status, wraplength=1100).pack(anchor='w', padx=8, pady=6)
        self.canvas.bind('<Button-1>', self.click)
        self.canvas.bind('<Button-3>', lambda e:self.undo())
        self.canvas.bind('<Motion>', self.motion)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.redraw()
        if self.region['polygon']:
            self.generate()

    def change_mode(self):
        if self.draft:
            self.draft=[]
            self.status.set('Unfinished draft cleared. Committed room/corridors retained.')
            self.redraw()

    def xy_to_pixel(self, xy):
        c,r=self.fmap.world_to_grid(xy)
        return float(c/self.fmap.shape[1]*self.size[0]), float((self.fmap.shape[0]-r)/self.fmap.shape[0]*self.size[1]+74)

    def event_xy(self, e):
        if not 0 <= e.x < self.size[0] or not 74 <= e.y < self.size[1]+74:
            return None
        col=e.x/self.size[0]*self.fmap.shape[1]
        row=(1-(e.y-74)/self.size[1])*self.fmap.shape[0]
        # Half-open upper edge.
        row=min(row, np.nextafter(float(self.fmap.shape[0]), 0.))
        cell=(int(row), int(col))
        if not self.fmap.contains(cell): return None
        return self.fmap.cells_to_world(cell).tolist(),cell

    def invalidate(self):
        self.records, self.report, self.layers = [], None, None

    def click(self,e):
        loc=self.event_xy(e)
        if loc is None:return
        xy,cell=loc
        if e.state & 0x0001:
            self.region['excluded_capture_disks'].append({'x':xy[0],'y':xy[1],'radius':.30})
            self.invalidate(); self.status.set('Capture-only exclusion added. Regenerate before saving.');self.redraw();return
        mode=self.mode.get()
        if mode!='outline' and self.fmap.grid[cell]==100:
            self.status.set('Dark/occupied: cannot pin or place a corridor vertex here. Outline vertices may be on walls.');return
        if mode=='pin':
            if not any(self.fmap.world_to_cell([p['x'],p['y']])==cell for p in self.region['pins']):
                self.region['pins'].append({'x':xy[0],'y':xy[1],'allow_unknown':bool(self.fmap.grid[cell]==-1)})
            self.invalidate();self.status.set('Camera seed added; it may shift locally if space is available. Generate to preview.')
        else:
            if not self.draft or xy!=self.draft[-1]:self.draft.append(xy)
            self.status.set(f'{len(self.draft)} vertices for {mode}. Finish shape when complete.')
        self.redraw()

    def finish(self):
        mode=self.mode.get()
        minimum=3 if mode=='outline' else 2
        if mode=='pin' or len(self.draft)<minimum:
            self.status.set(f'Need at least {minimum} vertices for a shape.');return
        if mode=='outline':
            self.region['polygon']=copy.deepcopy(self.draft)
            self.status.set('Outline stored. Green AND gray inside are capture candidates. Generate, or add a priority corridor.')
        else:
            self.region['corridors'].append({'points':copy.deepcopy(self.draft),'allow_unknown':True,
                                             'note':'Priority camera coverage, not a route.'})
            self.status.set('Priority corridor stored. Individual dark/outside samples are skipped; this is NOT a walked line.')
        self.draft=[];self.invalidate();self.redraw()

    def undo(self):
        if self.draft:self.draft.pop()
        elif self.mode.get()=='corridor' and self.region['corridors']:self.region['corridors'].pop();self.invalidate()
        elif self.mode.get()=='pin' and self.region['pins']:self.region['pins'].pop();self.invalidate()
        elif self.region['excluded_capture_disks']:self.region['excluded_capture_disks'].pop();self.invalidate()
        elif self.mode.get()=='outline':self.region['polygon']=[];self.invalidate()
        self.redraw()

    def generate(self):
        if self.draft:
            self.status.set('Finish the draft shape or undo it before generating.');return False
        try:
            a=self.args
            self.records,self.report,self.layers=generate(self.fmap,self.region,count=a.positions,
                preferred=a.preferred_clearance,max_shift=a.max_shift,corridor_spacing=a.corridor_spacing,
                gray_fraction=a.gray_fraction,seed=a.seed,scale=a.scale,floor_y=a.floor_y,camera_height=a.camera_height)
            r=self.report
            self.status.set(f"{r['actual_positions']}/{r['requested_positions']} points | {r['gray_camera_positions']} in gray | "
                            f"{r['corridor_positions']} priority corridor | {r['nudged_positions']} moved locally | "
                            f"{r['below_preferred_clearance']} BELOW preferred clearance, KEPT. Review then SAVE.")
            self.redraw();return True
        except Exception as exc:
            self.status.set(str(exc));return False

    def motion(self,e):
        loc=self.event_xy(e)
        if loc is None:return
        xy,cell=loc
        state={100:'OCCUPIED',0:'green/free',-1:'GRAY/unknown'}[int(self.fmap.grid[cell])]
        extra=''
        if self.layers is not None:
            extra=f" | clearance >= {self.layers['clearance_m'][cell]:.3f} m | allowed={bool(self.layers['allowed'][cell])}"
        self.hover.set(f'X={xy[0]:.3f}, Y={xy[1]:.3f} | {state}{extra}')

    def redraw(self):
        im=preview(self.fmap,self.region,self.records,self.layers,
                   max_size=(min(1150,self.root.winfo_screenwidth()-70),max(200,self.root.winfo_screenheight()-280)))
        self.size=(im.width,im.height-74)
        if self.draft:
            d=ImageDraw.Draw(im); pts=[self.xy_to_pixel(p) for p in self.draft]
            if len(pts)>1:d.line(pts,fill='#ffffff',width=3)
            for x,y in pts:d.ellipse((x-4,y-4,x+4,y+4),fill='#11c9bd',outline='white')
        self.photo=ImageTk.PhotoImage(im)
        self.canvas.configure(width=im.width,height=im.height)
        self.canvas.delete('all');self.canvas.create_image(0,0,image=self.photo,anchor='nw')

    def save(self):
        if not self.generate():return
        try:
            save_results(self.args.out,self.fmap,self.region,self.records,self.report,self.layers)
            print_summary(self.report,self.args.out)
            self.saved=True;self.root.destroy()
        except Exception as exc:
            messagebox.showerror('Could not save',str(exc))

    def close(self):
        if messagebox.askyesno('Close without saving?','Unsaved point positions will be discarded. Source map stays unchanged.'):
            self.root.destroy()


def print_summary(r,out):
    print(f"Saved {r['actual_positions']}/{r['requested_positions']} camera positions to {out}")
    print(f"Gray: {r['gray_camera_positions']}; corridor: {r['corridor_positions']}; locally shifted: {r['nudged_positions']}")
    print(f"Below preferred clearance (retained, not dropped): {r['below_preferred_clearance']}")
    print('Virtual-camera selection only. Inspect camera_positions.png. No render or navigation map edits.')
