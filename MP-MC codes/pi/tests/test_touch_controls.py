"""Isolated touch traces, storage, async dispatch and authenticated HTTP tests."""
import asyncio
import importlib.util
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

PI_DIR = Path(__file__).resolve().parents[1] / "adam"
sys.path.insert(0, str(PI_DIR))
import touch_controls as touch


def mapping(sensor="touch1", **events):
    return {sensor: {key: {"action": value, "value": None} for key, value in events.items()}}


class ClassifierTests(unittest.TestCase):
    def build(self, assignments=None, protected=lambda sensor: False):
        self.events, self.legacy = [], []
        self.assignments = assignments or {}
        self.classifier = touch.TouchClassifier(lambda: self.assignments,
            lambda sensor, event, binding: self.events.append((sensor, event, binding)),
            self.legacy.append, protected)
        self.classifier.feed([0, 0, 0, 0], 0)
        return self.classifier

    def test_single_tap_keeps_fixed_reaction_even_on_a_configured_pad(self):
        c=self.build(mapping(hold="volume_up"))
        c.feed([1,0,0,0], .1)
        c.firmware_gesture(1,.11)
        c.feed([0,0,0,0], .2)
        c.tick(1)
        self.assertEqual(self.events, [])
        self.assertEqual(self.legacy, [1])

    def test_touch3_double_waits_for_a_possible_third_tap(self):
        c=self.build(mapping("touch3", double="volume_down", triple="volume_up"))
        for state,when in [(1,.1),(0,.2),(1,.3),(0,.4)]:
            c.feed([0,0,state,0],when)
        c.tick(.69)
        self.assertEqual(self.events, [])
        c.tick(.71)
        self.assertEqual([e[1] for e in self.events], ["double"])
        self.assertEqual(self.legacy, [])

    def test_touch3_triple_never_also_dispatches_double_or_single(self):
        c=self.build(mapping("touch3", double="volume_down", triple="volume_up"))
        for state,when in [(1,.1),(0,.2),(1,.3),(0,.4),(1,.5),(0,.6)]:
            c.feed([0,0,state,0],when)
            if state:c.firmware_gesture(3,when+.01)
        c.tick(2)
        self.assertEqual([e[1] for e in self.events], ["triple"])
        self.assertEqual(self.legacy, [])

    def test_two_separate_touch3_taps_keep_two_fixed_reactions(self):
        c=self.build(mapping("touch3", double="volume_up", hold="none"))
        for state,when in [(1,.1),(0,.2),(1,.7),(0,.8)]:
            c.feed([0,0,state,0],when)
        c.tick(1.2)
        self.assertEqual(self.events, [])
        self.assertEqual(self.legacy, [3,3])

    def test_long_press_fires_once_on_every_pad_with_no_trailing_tap(self):
        for index,sensor in enumerate(touch.SENSORS):
            c=self.build(mapping(sensor, hold="volume_mute"))
            states=[0]*4;states[index]=1
            c.feed(states,.1)
            for when in (.4,.76,1.4,2.4):
                c.tick(when)
                c.firmware_gesture({0:1,1:1,2:3,3:2}[index],when)
            c.feed([0]*4,2.5);c.tick(3)
            self.assertEqual([(e[0],e[1]) for e in self.events], [(sensor,"hold")])
            self.assertEqual(self.legacy, [])

    def test_second_press_held_produces_hold_only(self):
        c=self.build(mapping("touch3",double="volume_down",hold="volume_mute"))
        c.feed([0,0,1,0],.1);c.feed([0,0,0,0],.2)
        c.feed([0,0,1,0],.3);c.tick(1)
        c.feed([0,0,0,0],1.2);c.tick(2)
        self.assertEqual([e[1] for e in self.events],["hold"])
        self.assertEqual(self.legacy,[])

    def test_unmapped_stop_preserves_legacy_and_deduplicates_repeats(self):
        c=self.build();c.feed([0,0,1,0],.1)
        for when in (.22,.4,.6,.8):c.firmware_gesture(3,when)
        c.feed([0,0,0,0],1);c.feed([0,0,1,0],1.2);c.firmware_gesture(3,1.35)
        self.assertEqual(self.legacy,[3,3]);self.assertEqual(self.events,[])

    def test_petting_chord_keeps_reaction_and_cancels_custom_holds(self):
        c=self.build({**mapping("touch3",hold="volume_up"),**mapping("touch4",hold="volume_down")})
        c.feed([0,0,1,1],.1);c.firmware_gesture(2,.11);c.firmware_gesture(2,.4)
        c.tick(1);c.feed([0]*4,1.2);c.tick(2)
        self.assertEqual(self.legacy,[2]);self.assertEqual(self.events,[])

    def test_protected_state_stays_latched_until_finger_releases(self):
        protected={"value":True}
        c=self.build(mapping("touch3",hold="volume_mute"),lambda sensor:protected["value"])
        c.feed([0,0,1,0],.1);c.firmware_gesture(3,.2)
        protected["value"]=False;c.tick(1);c.firmware_gesture(3,1)
        c.feed([0]*4,1.2);c.tick(2)
        self.assertEqual(self.legacy,[3]);self.assertEqual(self.events,[])

    def test_pad_held_at_start_does_not_execute(self):
        events=[]
        c=touch.TouchClassifier(lambda:mapping(hold="volume_mute"),lambda *e:events.append(e),lambda code:None)
        c.feed([1,0,0,0],0);c.tick(2);c.feed([0]*4,3);c.tick(4)
        self.assertEqual(events,[])

    def test_fixed_taps_and_other_pad_multitaps_are_rejected(self):
        for bad in (mapping(tap="none"),mapping(double="volume_up"),mapping("touch4",triple="volume_down"),
                    mapping(hold="write_clipboard"),mapping(hold="dispatch_coding_task"),{"top":{}}):
            with self.assertRaises(ValueError):touch.validate_assignments(bad)

    def test_legacy_in_memory_tap_mapping_can_never_execute(self):
        c=self.build(mapping(tap="volume_up"))
        c.feed([1,0,0,0],.1);c.feed([0]*4,.2);c.tick(1)
        self.assertEqual(self.events,[]);self.assertEqual(self.legacy,[1])


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = touch.AssignmentStore(Path(self.directory.name) / "touch.json")
        self.link = types.SimpleNamespace(connected=True, touch_q=queue.Queue(), gesture_q=queue.Queue())
        self.events = []
        self.calls = []
        async def broadcast(payload):
            self.events.append(dict(payload))
        def dispatch(action, value):
            self.calls.append((action, value))
            return {"status": "ok"}
        self.runtime = touch.TouchRuntime(self.link, self.store, dispatch, broadcast)

    async def asyncTearDown(self):
        await self.runtime.stop()
        self.directory.cleanup()

    async def test_persisted_ack_survives_restart_and_failed_save_keeps_old_state(self):
        expected = touch.validate_assignments(mapping(hold="volume_up"))
        self.assertEqual(await self.store.save(expected), expected)
        replacement = touch.AssignmentStore(self.store.path)
        await replacement.load()
        self.assertEqual(replacement.snapshot(), expected)
        before = self.store.path.read_bytes()
        with patch.object(touch.os, "replace", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                await self.store.save(mapping(hold="volume_down"))
        self.assertEqual(self.store.snapshot(), expected)
        self.assertEqual(self.store.path.read_bytes(), before)

    async def test_dispatch_is_off_event_loop_and_one_press_executes_once(self):
        gate = threading.Event()
        entered = threading.Event()
        def slow(action, value):
            entered.set()
            gate.wait(1)
            self.calls.append((action, value))
            return {"status": "ok"}
        self.runtime.dispatch = slow
        await self.store.save(mapping(hold="volume_up"))
        self.runtime.start()
        self.runtime._enqueue("touch1", "hold", {"action": "volume_up", "value": None})
        for _ in range(30):
            if entered.is_set(): break
            await asyncio.sleep(.01)
        self.assertTrue(entered.is_set())
        # The event loop remains usable while the laptop request is blocked.
        self.link.touch_q.put([0, 0, 0, 0])
        await asyncio.sleep(.05)
        self.assertTrue(self.runtime.classifier.pads["touch1"].initialized)
        gate.set()
        await asyncio.wait_for(self.runtime.actions.join(), 1)
        self.assertEqual(self.calls, [("volume_up", None)])
        self.assertEqual(self.events[0]["handled_by"], "pi")
        self.assertEqual(self.events[0]["status"], "ok")
        self.assertNotIn("value", self.events[0])

    async def test_stale_queued_action_is_discarded_when_assignments_change(self):
        await self.store.save(mapping(hold="volume_up"))
        self.runtime._enqueue("touch1", "hold", {"action": "volume_up", "value": None})
        await self.store.save(mapping(hold="volume_down"))
        self.runtime.start()
        await asyncio.wait_for(self.runtime.actions.join(), 1)
        self.assertEqual(self.calls, [])

    async def test_http_write_requires_sync_token_and_echoes_exact_persisted_assignments(self):
        config = types.ModuleType("config")
        config.SYNC_HOST, config.SYNC_PORT, config.SYNC_TOKEN = "127.0.0.1", 0, "isolated-test-key"
        memories = types.ModuleType("memory_store")
        memories.memory, memories.conv_log = {}, []
        scheduler = types.ModuleType("scheduler")
        scheduler.snapshot = lambda: {"schedules": [], "todos": []}
        spec = importlib.util.spec_from_file_location("isolated_sync_api", PI_DIR / "sync_api.py")
        api = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"config": config, "memory_store": memories, "scheduler": scheduler}):
            spec.loader.exec_module(api)
        with patch.object(touch, "_runtime", self.runtime):
            server = await asyncio.start_server(api._handle, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            async def call(path, body=None, token=""):
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                encoded = json.dumps(body).encode() if body is not None else b""
                request = (f"{'POST' if body is not None else 'GET'} {path} HTTP/1.1\r\nHost: localhost\r\n"
                           f"Content-Length: {len(encoded)}\r\nX-ADAM-Token: {token}\r\n\r\n").encode() + encoded
                writer.write(request)
                await writer.drain()
                reply = await asyncio.wait_for(reader.read(), 2)
                writer.close()
                await writer.wait_closed()
                header, payload = reply.split(b"\r\n\r\n", 1)
                return int(header.split()[1]), json.loads(payload)
            try:
                code, ping = await call("/api/ping")
                self.assertEqual(code, 200)
                self.assertTrue(ping["capabilities"]["touch_assignments"])
                expected = touch.validate_assignments(mapping(hold="volume_up"))
                code, _ = await call("/api/touch/assignments", {"assignments": expected})
                self.assertEqual(code, 403)
                self.assertEqual(self.store.snapshot(), {})
                code, reply = await call("/api/touch/assignments", {"assignments": expected}, "isolated-test-key")
                self.assertEqual(code, 200)
                self.assertEqual(reply["assignments"], expected)
                code, saved = await call("/api/touch/assignments")
                self.assertEqual(saved["assignments"], expected)
            finally:
                server.close()
                await server.wait_closed()


if __name__ == "__main__":
    unittest.main()
