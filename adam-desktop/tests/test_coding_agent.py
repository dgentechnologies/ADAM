"""CLI integration contract tests; no paid CLI or model request is executed."""

import io
import json
import logging
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
with patch.dict(sys.modules, {"config": SimpleNamespace(logger=logging.getLogger("adam-coding-fixture"))}):
    from coding_agent import CodingAgentManager, CodingTask, TaskState, MAX_TASKS


class Input:
    def __init__(self):
        self.value = ""
        self.closed = False

    def write(self, value):
        self.value += value

    def flush(self):
        pass

    def close(self):
        self.closed = True


class Process:
    def __init__(self, events=(), code=0):
        self.stdin = Input()
        self.stdout = io.StringIO("\n".join(json.dumps(event) if isinstance(event, dict) else event for event in events) + "\n")
        self.code = code
        self.returncode = None
        self.pid = 12345

    def wait(self, timeout=None):
        self.returncode = self.code
        return self.code

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = self.code = 1

    kill = terminate


class Group:
    def __init__(self, process):
        self.process = process
        self.stopped = False
        self.closed = False

    def stop(self):
        self.stopped = True
        self.process.terminate()

    def close(self):
        self.closed = True


class CodingAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cwd = self.temp.name
        self.groups = []
        self.calls = []
        self.process = Process([{"type": "turn.completed"}])
        def popen(command, **kwargs):
            self.calls.append((command, kwargs))
            return self.process
        def group(process):
            created = Group(process)
            self.groups.append(created)
            return created
        self.manager = CodingAgentManager(popen_factory=popen,
            finder=lambda tool: str(Path(self.cwd) / (tool + ".exe")), group_factory=group)

    def wait_task(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = self.manager.get_status()["active_task"]
            if status and status["end_time"]:
                return status
            time.sleep(0.01)
        self.fail("Worker did not finish")

    def test_missing_workspace_tool_and_simulation_are_rejected(self):
        self.assertEqual(self.manager.dispatch("Do work", "codex")["status"], "error")
        self.assertEqual(self.manager.dispatch("Do work", "codex", ".")["status"], "error")
        self.assertEqual(self.manager.dispatch("Do work", "unknown", self.cwd)["status"], "error")
        self.assertEqual(self.manager.dispatch("Do work", "codex", self.cwd, simulate=True)["status"], "error")
        self.manager._finder = lambda _: None
        result = self.manager.dispatch("Do work", "codex", self.cwd)
        self.assertEqual(result["status"], "error")
        self.assertIn("Install Codex", result["reason"])
        self.assertEqual(len(self.manager.tasks), 0)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.manager.get_status()["tools"]["codex"]["available"])

    def test_codex_real_json_contract_stdin_and_safe_flags(self):
        self.process = Process([
            {"type": "item.started", "item": {"type": "command_execution", "command": "private command"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Changed the selected file."}},
            {"type": "turn.completed"}])
        prompt = '--help & echo secret | unsafe " quote'
        with patch("coding_agent.logger") as logger:
            self.manager.dispatch(prompt, "codex", self.cwd)
            task = self.wait_task()
        self.assertEqual(task["state"], TaskState.DONE)
        self.assertEqual(task["result"], "Changed the selected file.")
        command, kwargs = self.calls[0]
        self.assertNotIn(prompt, command)
        self.assertEqual(command[-1], "-")
        self.assertIn("--json", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "workspace-write")
        self.assertEqual(command[command.index("--ask-for-approval") + 1], "on-request")
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", command)
        self.assertNotIn("--full-auto", command)
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["cwd"], str(Path(self.cwd).resolve()))
        self.assertEqual(self.process.stdin.value, prompt + "\n")
        self.assertNotIn("secret", str(logger.mock_calls))
        self.assertNotIn("selected file", str(logger.mock_calls))
        self.assertNotIn("private command", json.dumps(task["output_tail"]))

    def test_claude_result_permission_denial_is_not_success(self):
        self.process = Process([
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "I need access."}]}},
            {"type": "result", "is_error": False, "result": "Cannot edit yet.", "permission_denials": [{"tool_name": "Edit"}]}])
        self.manager.dispatch("Fix project", "claude", self.cwd)
        task = self.wait_task()
        self.assertEqual(task["state"], TaskState.ERROR)
        self.assertIn("needs CLI permission", task["last_message"])
        command = self.calls[0][0]
        self.assertIn("--print", command)
        self.assertIn("--verbose", command)
        self.assertIn("stream-json", command)
        self.assertEqual(command[command.index("--permission-mode") + 1], "default")
        self.assertEqual(self.manager.send_input(task["id"], "yes")["status"], "error")

    def test_protocol_failure_and_empty_success_exit_are_errors(self):
        self.process = Process([{"type": "turn.failed", "error": {"message": "Authentication required"}}])
        self.manager.dispatch("Read project", "codex", self.cwd)
        self.assertEqual(self.wait_task()["state"], TaskState.ERROR)
        self.process = Process(["Usage help only"])
        self.manager.dispatch("Read project", "codex", self.cwd)
        self.assertIn("without completing", self.wait_task()["last_message"])

    def test_cancellation_waits_for_owned_process_and_stays_cancelled(self):
        stopped = threading.Event()
        entered = threading.Event()
        class BlockingOutput:
            def readline(self, _limit):
                entered.set()
                stopped.wait(3)
                return ""
            def close(self):
                pass
        self.process.stdout = BlockingOutput()
        original_terminate = self.process.terminate
        def terminate():
            original_terminate()
            stopped.set()
        self.process.terminate = terminate
        self.manager.dispatch("Long work", "codex", self.cwd)
        self.assertTrue(entered.wait(2))
        self.assertEqual(self.manager.dispatch("Other work", "codex", self.cwd)["status"], "error")
        self.assertEqual(self.manager.cancel()["status"], "ok")
        task = self.wait_task()
        self.assertTrue(self.groups[0].stopped)
        self.assertTrue(task["cancelled"])
        self.assertEqual(task["state"], TaskState.ERROR)
        self.assertEqual(task["last_message"], "Cancelled by user")

    def test_history_and_output_are_bounded(self):
        for number in range(MAX_TASKS + 5):
            task = CodingTask(str(number), "Task", "codex", self.cwd)
            task.state = TaskState.DONE
            task.end_time = time.time()
            self.manager.tasks[task.id] = task
        self.manager.dispatch("New task", "codex", self.cwd)
        self.wait_task()
        self.assertLessEqual(len(self.manager.tasks), MAX_TASKS)
        self.assertLessEqual(len(self.manager.get_history(1000)), MAX_TASKS)
        task = self.manager.tasks[self.manager.active_task_id]
        for _ in range(150):
            self.manager._record(task, "x" * 5000)
        self.assertEqual(len(task.output_log), 100)
        self.assertEqual(len(task.to_dict()["output_tail"]), 12)
        self.assertLessEqual(max(map(len, task.output_log)), 2000)

    def test_spawn_failure_is_safe_and_does_not_reveal_prompt(self):
        self.manager._popen = Mock(side_effect=OSError("secret prompt would otherwise be leaked"))
        with patch("coding_agent.logger") as logger:
            self.manager.dispatch("secret prompt", "codex", self.cwd)
            task = self.wait_task()
        self.assertEqual(task["state"], TaskState.ERROR)
        self.assertNotIn("secret", task["last_message"])
        self.assertNotIn("secret", str(logger.mock_calls))

    def test_npm_wrapper_requires_known_entrypoint_and_never_batch_shell(self):
        entry = Path(self.cwd) / "node_modules/@openai/codex/bin/codex.js"
        entry.parent.mkdir(parents=True)
        entry.write_text("// Test fixture only")
        self.manager._finder = lambda name: str(Path(self.cwd) / ("codex.cmd" if name == "codex" else "node.exe"))
        with patch("coding_agent.sys.platform", "win32"):
            launcher, reason = self.manager._launcher("codex")
        self.assertEqual(reason, "")
        self.assertEqual(launcher, [str(Path(self.cwd) / "node.exe"), str(entry)])
        self.assertFalse(any(item.endswith(".cmd") for item in launcher))


if __name__ == "__main__":
    unittest.main()
