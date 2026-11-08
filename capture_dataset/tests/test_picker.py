import os
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from types import SimpleNamespace

from capture_dataset.picker import Picker
from capture_dataset.common import load_plan, world_to_pixel
from test_capture import create_map, make_args, config_from_args

@unittest.skipUnless(os.environ.get('DISPLAY'), 'requires X display or Xvfb')
class PickerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.path=Path(self.tmp.name)
        self.fmap=create_map(self.path)
        self.root=tk.Tk()
        self.args=make_args(map=str(self.path/'map.yaml'),scene=str(self.path/'dummy.ply'),out=str(self.path/'data'))
        self.ui=Picker(self.root,self.args,config_from_args(self.args)); self.root.update()
    def tearDown(self):
        try: self.root.destroy()
        except tk.TclError: pass
        self.tmp.cleanup()
    def event(self,cell,state=0):
        xy=self.fmap.cell_to_world(cell)
        x,y=world_to_pixel(self.fmap,xy,self.ui.size)
        return SimpleNamespace(x=x,y=y+62,state=state)
    def test_gray_requires_checkbox(self):
        self.ui.click(self.event((30,10)))
        self.assertEqual(len(self.ui.pins),0)
        self.ui.allow_unknown.set(True); self.ui.click(self.event((30,10)))
        self.assertEqual(len(self.ui.pins),1)
        self.assertTrue(self.ui.pins[0]['allow_unknown'])
    def test_free_pin_and_remove(self):
        e=self.event((5,5)); self.ui.click(e)
        self.assertEqual(len(self.ui.pins),1)
        self.ui.remove(e); self.assertEqual(len(self.ui.pins),0)
    def test_occupied_rejected(self):
        self.ui.allow_unknown.set(True); self.ui.click(self.event((0,0)))
        self.assertEqual(len(self.ui.pins),0)
    def test_exclusion_undo(self):
        self.ui.click(self.event((5,5),1)); self.assertEqual(len(self.ui.excluded),1)
        self.ui.undo_exclusion(); self.assertEqual(len(self.ui.excluded),0)
    def test_save_and_close(self):
        self.ui.allow_unknown.set(True); self.ui.click(self.event((30,10)))
        self.ui.save()
        self.assertTrue(self.ui.saved)
        p=load_plan(self.path/'data')
        self.assertEqual(len(p['frames']),480)
        self.assertTrue(p['stations'][0]['unknown_capture_only'])
    def test_window_fits(self):
        self.assertLessEqual(self.root.winfo_height(),self.root.winfo_screenheight())
        self.assertLessEqual(self.root.winfo_width(),self.root.winfo_screenwidth())

if __name__=='__main__': unittest.main()
