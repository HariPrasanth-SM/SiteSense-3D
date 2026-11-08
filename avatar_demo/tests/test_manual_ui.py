"""Optional CPU GUI tests: xvfb-run -a python -m unittest discover -s avatar_demo/tests -v"""
import json
import math
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from avatar_demo.app import AvatarApp, argument_parser
from avatar_demo.core import FloorMap, Planner


@unittest.skipUnless(os.environ.get('DISPLAY'), 'Tk UI tests need a display; use xvfb-run -a')
class ManualUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.args=argument_parser().parse_args(['--no-render','--output', self.temp.name])
        grid=np.zeros((60,80),np.int8)
        grid[[0,-1],:]=100;grid[:,[0,-1]]=100;grid[:,40]=100
        self.map=FloorMap(grid,.1,np.zeros(3))
        self.planner=Planner(self.map)
        self.app=AvatarApp(self.args,self.map,self.planner)
        self.app.root.focus_force()
        self.pump(.06)

    def tearDown(self):
        self.app.close()
        self.temp.cleanup()

    def pump(self,duration=.05):
        until=time.monotonic()+duration
        while time.monotonic()<until:
            self.app.root.update()
            time.sleep(.003)

    def event(self,key,widget=None):
        return SimpleNamespace(keysym=key,widget=widget or self.app.root)

    def test_keyboard_hold_and_release(self):
        w=self.app.walker;w.yaw=0;start=w.xy.copy()
        self.app._key_press(self.event('Up'))
        self.app._manual_update(.1)
        self.assertGreater(w.xy[0],start[0])
        self.app._key_release(self.event('Up'))
        self.pump(.07)
        self.assertEqual(self.app.held.axes(),(0,0,0))
        x=w.xy.copy();self.app._manual_update(.1)
        np.testing.assert_array_equal(w.xy,x)

    def test_key_autorepeat_pair_does_not_clear_hold(self):
        self.app._key_press(self.event('Up'))
        self.app._key_release(self.event('Up'))
        self.app._key_press(self.event('Up'))
        self.pump(.06)
        self.assertEqual(self.app.held.axes(),(1,0,0))

    def test_text_entry_does_not_drive(self):
        entry=self.app.ttk.Entry(self.app.root)
        entry.pack();entry.focus_force();self.pump()
        self.app._key_press(self.event('w',entry))
        self.app._key_press(self.event('Up',entry))
        self.assertEqual(self.app.held.axes(),(0,0,0))
        entry.destroy()

    def test_focus_into_entry_clears_held_keys(self):
        self.app._key_press(self.event('Up'))
        entry=self.app.ttk.Entry(self.app.root)
        entry.pack();entry.focus_force();self.pump()
        self.assertEqual(self.app.held.axes(),(0,0,0))
        entry.destroy()

    def test_focus_loss_pauses_motion(self):
        self.app._press_action('test','forward')
        original=self.app.root.focus_get
        self.app.root.focus_get=lambda:None
        self.app._check_focus()
        self.app.root.focus_get=original
        self.assertEqual(self.app.held.axes(),(0,0,0))
        self.assertFalse(self.app.walker.running)

    def test_manual_takeover_clears_plan(self):
        start=self.app.walker.xy.copy();target=start+[.5,.5]
        p=self.planner.plan(start,target,mode='point')
        self.app.walker.assign(p);self.app.walker.start()
        self.app._press_action('test','left')
        self.assertIsNone(self.app.walker.plan)
        self.assertFalse(self.app.walker.running)
        self.app.walk()
        self.assertFalse(self.app.walker.running)

    def test_pitch_does_not_cancel_autowalk(self):
        start=self.app.walker.xy.copy();p=self.planner.plan(start,start+[.5,.5],mode='point')
        self.app.walker.assign(p);self.app.walker.start()
        self.app._press_action('test','look_up');self.app._manual_update(.1)
        self.assertIs(self.app.walker.plan,p)
        self.assertTrue(self.app.walker.running)
        self.assertEqual(self.app.pitch_deg,2.5)

    def test_mouse_press_and_release(self):
        b=self.app.hold_buttons['mouse:forward']
        b.event_generate('<ButtonPress-1>',x=3,y=3)
        self.pump(.03)
        self.assertEqual(self.app.held.axes(),(1,0,0))
        b.event_generate('<ButtonRelease-1>',x=3,y=3)
        self.pump(.03)
        self.assertEqual(self.app.held.axes(),(0,0,0))

    def test_guard_toggle_is_manual_only_and_never_edits_map(self):
        before=self.map.occupancy.copy();walkable=self.planner.walkable.copy()
        self.app.collision_var.set(False);self.app.guard_changed()
        self.assertIn('CHECKS OFF',self.app.guard_text.get())
        self.app.set_start([4.05,2.55])
        self.app.collision_var.set(True);self.app.guard_changed()
        self.app.walker.yaw=0
        old=self.app.walker.xy.copy()
        self.app._press_action('test','forward');self.app._manual_update(.1)
        np.testing.assert_array_equal(self.app.walker.xy,old)
        self.app.return_to_free()
        self.assertEqual(self.map.occupancy[self.map.world_to_cell(self.app.walker.xy)],0)
        np.testing.assert_array_equal(before,self.map.occupancy)
        np.testing.assert_array_equal(walkable,self.planner.walkable)

    def test_bookmark_without_image_and_speeds(self):
        self.app.review_var.set('boundary');self.app.note_var.set('test boundary')
        self.app.bookmark()
        file=self.app.session/'review_points.jsonl'
        data=json.loads(file.read_text())
        self.assertEqual(data['assessment'],'boundary')
        self.assertIsNone(data['displayed_frame'])
        self.assertFalse(data['map_edited'])
        self.app.speed_var.set('.18');self.app.turn_var.set('30');self.app.apply_speeds()
        self.assertEqual(self.app.walker.speed,.18)
        self.assertAlmostEqual(self.app.walker.turn_rate,math.radians(30))

    def test_bookmark_preserves_frame_pose_not_current_pose(self):
        from avatar_demo.core import camera_matrices
        w=self.app.walker
        frame_xy=w.xy+[.3,.2]
        K,c2w,w2c=camera_matrices(*frame_xy,w.yaw,1.7)
        self.app.last_frame={
            'rgb':np.zeros((6,8,3),np.uint8),'depth_scene':np.ones((6,8),np.float32),
            'depth_m':np.full((6,8),1.43,np.float32),'alpha':np.ones((6,8),np.float32),
            'pose':[float(frame_xy[0]),float(frame_xy[1]),w.yaw],
            'camera':{'K':K.tolist(),'camera_to_world':c2w.tolist(),'world_to_camera':w2c.tolist(),
                      'map_pose':{'x':float(frame_xy[0]),'y':float(frame_xy[1]),'yaw_rad':w.yaw,
                                  'pitch_deg':0.,'camera_height_m':1.7}}}
        self.app.bookmark()
        data=json.loads((self.app.session/'review_points.jsonl').read_text())
        self.assertAlmostEqual(data['displayed_frame']['position_difference_m'],np.hypot(.3,.2))
        self.assertTrue((self.app.session/'bookmarks/bookmark_001/view_camera.json').exists())

    def test_stop_survives_keyboard_repeat_until_release(self):
        self.app._key_press(self.event('Up'))
        self.app.pause()
        self.app._key_release(self.event('Up'))
        self.app._key_press(self.event('Up'))  # Synthetic X11 repeat pair.
        self.pump(.06)
        self.assertEqual(self.app.held.axes(),(0,0,0))
        self.app._key_release(self.event('Up'))
        self.pump(.06)
        self.app._key_press(self.event('Up'))
        self.assertEqual(self.app.held.axes(),(1,0,0))

    def test_presets_and_stop(self):
        self.app._press_action('test','look_up')
        self.app.set_pitch(-45)
        self.assertEqual(self.app.pitch_deg,-45)
        self.assertEqual(self.app.held.axes(),(0,0,0))
        self.app._press_action('test2','forward')
        self.app._space(self.event('space'))
        self.assertEqual(self.app.held.axes(),(0,0,0))

if __name__=='__main__':
    unittest.main()
