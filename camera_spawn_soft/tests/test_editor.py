import argparse
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from camera_spawn_soft.core import generate, empty_region
from test_core import write_map, region_all


@unittest.skipUnless(os.environ.get('DISPLAY'), 'Tk needs a display; use xvfb-run for CI')
class EditorTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        g=np.full((50,65),100,np.int8);g[5:45,5:60]=0
        g[34:42,9:55]=100;g[38,9:55]=-1
        self.m=write_map(self.p,g)
        self.args=argparse.Namespace(positions=100,preferred_clearance=.18,max_shift=.30,
            corridor_spacing=.35,gray_fraction=.2,seed=42,scale=1.43,floor_y=1.1,
            camera_height=1.7,out=str(self.p/'out'))
        self.root=tk.Tk()
    def tearDown(self):
        try:self.root.destroy()
        except Exception:pass
        self.tmp.cleanup()
    def make_editor(self,region):
        from camera_spawn_soft.editor import Editor
        e=Editor(self.root,self.m,region,self.args);self.root.update();return e
    def test_region_loaded_and_saves(self):
        e=self.make_editor(region_all(self.m))
        self.assertEqual(len(e.records),100)
        e.save()
        self.assertTrue(e.saved)
        self.assertTrue((self.p/'out/camera_positions.png').exists())
    def test_blank_requires_outline(self):
        e=self.make_editor(empty_region())
        self.assertFalse(e.generate())
        self.assertIn('outline',e.status.get())
    def test_finish_outline(self):
        e=self.make_editor(empty_region())
        e.draft=region_all(self.m)['polygon'];e.finish()
        self.assertTrue(e.generate());self.assertEqual(len(e.records),100)
    def test_pin_gray_serialization(self):
        from types import SimpleNamespace
        e=self.make_editor(region_all(self.m));e.mode.set('pin')
        xy=self.m.cells_to_world([38,20]);x,y=e.xy_to_pixel(xy)
        e.click(SimpleNamespace(x=x,y=y,state=0))
        self.assertTrue(e.generate())
        e.save();self.assertTrue(e.saved)
    def test_corridor_priority(self):
        e=self.make_editor(region_all(self.m));e.mode.set('corridor')
        e.draft=self.m.cells_to_world(np.array([[38,10],[38,54]])).tolist();e.finish()
        self.assertTrue(e.generate());self.assertGreater(e.report['corridor_positions'],10)


if __name__=='__main__':unittest.main()
