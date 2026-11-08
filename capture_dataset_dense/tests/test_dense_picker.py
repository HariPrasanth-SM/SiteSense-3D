import os
from pathlib import Path
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest

from capture_dataset.common import load_plan, world_to_pixel
from capture_dataset_dense.planning import HEADER
from capture_dataset_dense.picker import DensePicker
from test_dense import fixture,args_for,config_from_args


@unittest.skipUnless(os.environ.get('DISPLAY'),'requires X display or Xvfb')
class DensePickerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)
        self.fmap=fixture(self.path);self.root=tk.Tk()
        self.args=args_for(map=str(self.path/'map.yaml'),scene=str(self.path/'fake.ply'),out=str(self.path/'dataset'))
        self.cfg=config_from_args(self.args)
        self.ui=DensePicker(self.root,self.args,self.cfg,dict(pins=[],corridors=[],excluded_capture_disks=[]))
        self.root.update()
    def tearDown(self):
        try:self.root.destroy()
        except tk.TclError:pass
        self.tmp.cleanup()
    def event(self,cell,state=0):
        xy=self.fmap.cell_to_world(cell);x,y=world_to_pixel(self.fmap,xy,self.ui.size)
        return SimpleNamespace(x=x,y=y+HEADER,state=state)
    def test_starts_200_stations(self):
        self.assertEqual(len(self.ui.stations),200)
    def test_gray_pin_needs_opt_in(self):
        self.ui.click(self.event((53,10)));self.assertEqual(len(self.ui.selection['pins']),0)
        self.ui.allow_unknown.set(True);self.ui.click(self.event((53,10)))
        self.assertEqual(len(self.ui.selection['pins']),1)
    def test_draw_corridor_and_reserve_stations(self):
        self.ui.mode.set('corridor');self.ui.allow_unknown.set(True)
        for cell in [(53,6),(53,38),(53,73)]:self.ui.click(self.event(cell))
        self.assertEqual(len(self.ui.draft),3)
        self.ui.finish_corridor()
        self.assertEqual(len(self.ui.selection['corridors']),1)
        self.assertGreater(sum(bool(s['corridor_id']) for s in self.ui.stations),35)
        self.assertEqual(len(self.ui.stations),200)
    def test_reject_occupied_crossing(self):
        self.ui.mode.set('corridor');self.ui.allow_unknown.set(True)
        self.ui.click(self.event((30,15)));self.ui.click(self.event((30,45)))
        self.assertEqual(len(self.ui.draft),1)
        self.assertIn('OCCUPIED',self.ui.status.get())
    def test_finish_one_vertex_refused(self):
        self.ui.mode.set('corridor');self.ui.click(self.event((5,5)));self.ui.finish_corridor()
        self.assertEqual(len(self.ui.selection['corridors']),0)
    def test_undo_draft_and_completed_corridor(self):
        self.ui.mode.set('corridor');self.ui.allow_unknown.set(True)
        self.ui.click(self.event((53,6)));self.ui.click(self.event((53,38)))
        self.ui.undo_corridor();self.assertEqual(len(self.ui.draft),1)
        self.ui.click(self.event((53,38)));self.ui.finish_corridor()
        self.ui.undo_corridor();self.assertEqual(len(self.ui.selection['corridors']),0)
    def test_no_save_unfinished_draft(self):
        self.ui.mode.set('corridor');self.ui.click(self.event((5,5)));self.ui.save()
        self.assertFalse(self.ui.saved);self.assertFalse((self.path/'dataset'/'plan.json').exists())
    def test_save_2000_and_new_selection(self):
        self.ui.mode.set('corridor');self.ui.allow_unknown.set(True)
        self.ui.click(self.event((53,6)));self.ui.click(self.event((53,73)));self.ui.finish_corridor()
        self.ui.save();self.assertTrue(self.ui.saved)
        p=load_plan(self.path/'dataset');self.assertEqual(len(p['frames']),2000)
        self.assertEqual(len(p['corridors']),1)
    def test_window_fits(self):
        self.assertLessEqual(self.root.winfo_width(),self.root.winfo_screenwidth())
        self.assertLessEqual(self.root.winfo_height(),self.root.winfo_screenheight())
    def test_exclusion_and_undo(self):
        self.ui.click(self.event((10,10),1))
        self.assertEqual(len(self.ui.selection['excluded_capture_disks']),1)
        self.ui.undo_exclusion();self.assertEqual(len(self.ui.selection['excluded_capture_disks']),0)


if __name__=='__main__':unittest.main()
