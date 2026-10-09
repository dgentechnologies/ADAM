"""Run the real voice handler in isolation from camera/Gemini/audio imports."""
import ast
import asyncio
from pathlib import Path
import sys
import threading
import unittest

PI_DIR = Path(__file__).resolve().parents[1] / "adam"
sys.path.insert(0, str(PI_DIR))
import laptop_actions


class VoiceDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_and_control_run_off_audio_loop_and_preserve_zero(self):
        source = PI_DIR / "tool_handler.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                        and node.name == "_handle_laptop_control")
        audio_thread = threading.get_ident()
        threads = []
        commands = []

        def manifest():
            threads.append(threading.get_ident())
            return laptop_actions.fallback_manifest()

        def control(action, value):
            threads.append(threading.get_ident())
            commands.append((action, value))
            return {"status": "ok", "brightness": 0}

        namespace = {"asyncio": asyncio, "laptop_actions": laptop_actions,
                     "get_laptop_actions": manifest, "laptop_control_sync": control}
        # Compile the production function unchanged; heavy module-level
        # imports are unrelated to this dispatch contract.
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        result = await namespace["_handle_laptop_control"]({"action": "brightness_set", "value": "0%"})
        self.assertEqual(result, {"status": "ok", "brightness": 0})
        self.assertEqual(commands, [("brightness_set", 0)])
        self.assertEqual(len(threads), 2)
        self.assertTrue(all(thread != audio_thread for thread in threads))
