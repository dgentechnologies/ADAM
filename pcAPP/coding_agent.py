"""Run installed coding CLIs in an explicitly selected workspace.

Prompts travel over stdin, process output stays in bounded in-memory history,
and a missing tool never becomes a simulated success. Each launched process is
placed in an owned process group so Cancel cannot target unrelated tools.
"""
from __future__ import annotations

from collections import deque
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

from config import logger

MAX_TASKS = 30
MAX_OUTPUT_LINE = 128 * 1024
MAX_PROMPT = 12000
MAX_RUN_SECONDS = 3600
TOOLS = {"claude": "Claude Code", "codex": "Codex CLI"}


class TaskState:
    IDLE = "idle"
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"  # Retained for callers; print-mode runs cannot prompt interactively.
    DONE = "done"
    ERROR = "error"


class CodingTask:
    def __init__(self, task_id: str, prompt: str, tool: str = "claude", cwd: Optional[str] = None):
        self.id = task_id
        self.prompt = prompt
        self.tool = tool
        self.cwd = cwd
        self.state = TaskState.RUNNING
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.duration = 0.0
        self.exit_code: Optional[int] = None
        self.output_log = deque(maxlen=100)
        self.last_message = "Starting installed coding tool…"
        self.process: Optional[subprocess.Popen] = None
        self.input_needed_prompt = ""
        self.result = ""
        self.cancelled = False
        self._cancel = threading.Event()
        self._group = None
        self._protocol_error = ""
        self._completed_event = False
        self._stop_reason = ""

    def to_dict(self) -> Dict[str, Any]:
        duration = self.duration if self.end_time else round(time.time() - self.start_time, 1)
        return {"id": self.id, "prompt": self.prompt, "tool": self.tool, "cwd": self.cwd,
                "state": self.state, "start_time": self.start_time, "end_time": self.end_time,
                "duration": duration, "exit_code": self.exit_code, "last_message": self.last_message,
                "input_needed_prompt": self.input_needed_prompt, "output_tail": list(self.output_log)[-12:],
                "result": self.result, "cancelled": self.cancelled, "interactive": False}


class _BasicLimitInfo(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount",
        "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimitInfo(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimitInfo), ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class _OwnedProcessGroup:
    """Windows Job Object / POSIX process group containing only this launch."""
    def __init__(self, process):
        self.process = process
        self._lock = threading.Lock()
        self._handle = None
        self._kernel = None
        if sys.platform == "win32":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            kernel.CreateJobObjectW.restype = wintypes.HANDLE
            kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.CreateJobObjectW(None, None)
            limits = _ExtendedLimitInfo()
            limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not handle:
                raise OSError("A protected coding process group could not be created.")
            if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                kernel.CloseHandle(handle)
                raise OSError("The coding process group could not be configured.")
            if not kernel.AssignProcessToJobObject(handle, wintypes.HANDLE(int(process._handle))):
                kernel.CloseHandle(handle)
                raise OSError("The coding tool could not be attached to its process group.")
            self._kernel, self._handle = kernel, handle

    def stop(self):
        with self._lock:
            if self._handle:
                if not self._kernel.TerminateJobObject(self._handle, 1):
                    raise OSError("The coding process group could not be stopped.")
            elif sys.platform != "win32" and self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def close(self):
        with self._lock:
            if self._handle:
                self._kernel.CloseHandle(self._handle)
                self._handle = None


class CodingAgentManager:
    """Manage noninteractive tasks without bypassing the installed CLI's approvals."""
    def __init__(self, on_state_change: Optional[Callable[[Dict[str, Any]], None]] = None,
                 *, popen_factory=None, finder=None, group_factory=None):
        self.tasks: Dict[str, CodingTask] = {}
        self.active_task_id: Optional[str] = None
        self.on_state_change = on_state_change
        self._lock = threading.RLock()
        self._popen = popen_factory or subprocess.Popen
        self._finder = finder or shutil.which
        self._group_factory = group_factory or _OwnedProcessGroup

    def _notify(self, task: CodingTask):
        if self.on_state_change:
            try:
                with self._lock:
                    snapshot = task.to_dict()
                self.on_state_change(snapshot)
            except Exception:
                logger.warning("Coding task status callback failed")

    def _launcher(self, tool: str) -> tuple[list[str] | None, str]:
        path = self._finder(tool)
        if not path:
            return None, f"Install {TOOLS[tool]} and sign in to it before starting a task."
        executable = Path(path)
        if sys.platform == "win32" and executable.suffix.lower() in {".cmd", ".bat", ".ps1"}:
            # Invoke known npm entry points through Node without a batch shell.
            package = "@openai/codex/bin/codex.js" if tool == "codex" else "@anthropic-ai/claude-code/cli.js"
            entry = executable.parent / "node_modules" / package
            node = self._finder("node")
            if not node or not entry.is_file():
                return None, f"Install the native {TOOLS[tool]} executable or its official npm package with Node.js."
            return [node, str(entry)], ""
        return [str(executable)], ""

    def _tools(self) -> dict:
        result = {}
        for name, label in TOOLS.items():
            command, reason = self._launcher(name)
            result[name] = {"name": label, "available": command is not None, "reason": reason,
                            "requires_cli_login": True, "interactive": False}
        return result

    def dispatch(self, prompt: str, tool: str = "claude", cwd: Optional[str] = None,
                 simulate: bool = False) -> Dict[str, Any]:
        if simulate:
            return {"status": "error", "reason": "Choose an installed coding tool and a workspace to run a real task."}
        if not isinstance(tool, str) or tool not in TOOLS:
            return {"status": "error", "reason": "Choose Claude Code or Codex CLI."}
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT or "\x00" in prompt:
            return {"status": "error", "reason": f"Enter a task between 1 and {MAX_PROMPT} characters."}
        if not isinstance(cwd, str) or not cwd.strip():
            return {"status": "error", "reason": "Select an existing project folder before starting a task."}
        try:
            workspace = Path(cwd).expanduser()
            if not workspace.is_absolute():
                raise ValueError()
            workspace = workspace.resolve(strict=True)
            if not workspace.is_dir():
                raise ValueError()
        except (ValueError, OSError):
            return {"status": "error", "reason": "The selected project folder does not exist or cannot be opened."}
        command, reason = self._launcher(tool)
        if command is None:
            return {"status": "error", "reason": reason, "tools": self._tools()}
        with self._lock:
            active = self.tasks.get(self.active_task_id)
            if active and active.end_time is None:
                return {"status": "error", "reason": "A coding task is already running.", "active_task": active.to_dict()}
            while len(self.tasks) >= MAX_TASKS:
                self.tasks.pop(next(iter(self.tasks)))
            task = CodingTask("task_" + uuid.uuid4().hex[:16], prompt.strip(), tool, str(workspace))
            self.tasks[task.id] = task
            self.active_task_id = task.id
        logger.info("Starting coding task %s with %s", task.id, tool)
        self._notify(task)
        threading.Thread(target=self._run_process, args=(task, command), name="ADAM coding task", daemon=True).start()
        with self._lock:
            return {"status": "dispatched", "task": task.to_dict()}

    def _record(self, task: CodingTask, text: str, *, result=False):
        if not isinstance(text, str) or not text.strip():
            return
        # Only bounded in-memory UI output, never the persistent logger.
        text = text.strip()
        with self._lock:
            task.output_log.append(text[:2000])
            task.last_message = text[:180]
            if result:
                task.result = text[:12000]
        self._notify(task)

    def _parse_line(self, task: CodingTask, line: str):
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            self._record(task, line)
            return
        if not isinstance(event, dict):
            return
        event_type = event.get("type")
        if task.tool == "codex":
            if event_type in {"item.started", "item.updated", "item.completed"}:
                item = event.get("item", {})
                if not isinstance(item, dict):
                    return
                kind = item.get("type")
                if kind == "agent_message" and event_type == "item.completed":
                    self._record(task, item.get("text", ""), result=True)
                elif kind == "command_execution" and event_type == "item.started":
                    self._record(task, "Running a workspace command…")
                elif kind == "file_change" and event_type == "item.completed":
                    self._record(task, "Workspace file changes recorded.")
            elif event_type in {"turn.failed", "error"}:
                error = event.get("error", {})
                message = error.get("message", "") if isinstance(error, dict) else ""
                message = message or event.get("message") or "The coding tool could not finish the task."
                task._protocol_error = str(message)[:1000]
                self._record(task, task._protocol_error)
            elif event_type == "turn.completed":
                task._protocol_error = ""
                task._completed_event = True
        else:
            if event_type == "assistant":
                message = event.get("message", {})
                blocks = message.get("content", []) if isinstance(message, dict) else []
                if not isinstance(blocks, list):
                    return
                for block in blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        self._record(task, block.get("text", ""), result=True)
                    elif isinstance(block, dict) and block.get("type") == "tool_use":
                        self._record(task, "Using a workspace tool…")
            elif event_type == "result":
                task._completed_event = True
                if event.get("permission_denials"):
                    task._protocol_error = "This task needs CLI permission. Open Claude Code in the selected folder to review access, then start a new task."
                elif event.get("is_error"):
                    task._protocol_error = "Claude Code could not complete this task. Check its sign-in and the task output."
                self._record(task, event.get("result", ""), result=True)

    def _finish(self, task: CodingTask, message: str, *, error=False):
        with self._lock:
            task.state = TaskState.ERROR if error else TaskState.DONE
            task.last_message = message
            task.end_time = time.time()
            task.duration = round(task.end_time - task.start_time, 1)
        self._notify(task)

    def _run_process(self, task: CodingTask, launcher):
        command = list(launcher)
        if task.tool == "codex":
            command += ["--ask-for-approval", "on-request", "exec", "--json", "--sandbox",
                        "workspace-write", "--ephemeral", "--skip-git-repo-check", "-"]
        else:
            command += ["--print", "--input-format", "text", "--output-format", "stream-json", "--verbose",
                        "--permission-mode", "default", "--no-session-persistence"]
        watchdog = None
        try:
            if task._cancel.is_set():
                self._finish(task, task._stop_reason or "Cancelled by user", error=True)
                return
            kwargs = {"cwd": task.cwd, "stdin": subprocess.PIPE, "stdout": subprocess.PIPE,
                      "stderr": subprocess.STDOUT, "text": True, "encoding": "utf-8",
                      "errors": "replace", "bufsize": 1, "shell": False}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            process = self._popen(command, **kwargs)
            with self._lock:
                task.process = process
            try:
                group = self._group_factory(process)
            except Exception:
                process.terminate()
                process.wait(timeout=5)
                raise OSError("The coding tool could not be isolated for safe cancellation.")
            with self._lock:
                task._group = group
            if task._cancel.is_set():
                group.stop()
            else:
                # Drain stdout while feeding stdin: either pipe can fill during
                # CLI startup. The job watchdog also bounds a blocked writer.
                def feed_prompt():
                    try:
                        if not task._cancel.is_set():
                            process.stdin.write(task.prompt + "\n")
                            process.stdin.flush()
                    except (OSError, ValueError):
                        pass  # Exit status / completion events report the failure.
                    finally:
                        try:
                            process.stdin.close()
                        except (OSError, ValueError):
                            pass
                threading.Thread(target=feed_prompt, name="ADAM coding input", daemon=True).start()
            watchdog = threading.Timer(MAX_RUN_SECONDS, lambda: self._request_stop(task, "Task exceeded the one-hour limit."))
            watchdog.daemon = True
            watchdog.start()
            while True:
                line = process.stdout.readline(MAX_OUTPUT_LINE + 1)
                if not line:
                    break
                if len(line) > MAX_OUTPUT_LINE:
                    while line and not line.endswith("\n"):
                        line = process.stdout.readline(MAX_OUTPUT_LINE + 1)
                    self._record(task, "A very large tool output was omitted from the activity view.")
                    continue
                if not task._cancel.is_set() and line.strip():
                    self._parse_line(task, line.strip())
            exit_code = process.wait()
            with self._lock:
                task.exit_code = exit_code
            if task._cancel.is_set():
                self._finish(task, task._stop_reason or "Cancelled by user", error=True)
            elif task._protocol_error:
                self._finish(task, task._protocol_error, error=True)
            elif exit_code != 0:
                self._finish(task, f"{TOOLS[task.tool]} exited with code {exit_code}. Check its sign-in and the task output.", error=True)
            elif not task._completed_event:
                self._finish(task, "The tool exited without completing a task. Check its version, sign-in and output.", error=True)
            else:
                self._finish(task, "Task completed. Review the result and workspace changes.")
        except Exception:
            logger.warning("Coding task %s could not run", task.id)
            self._finish(task, task._stop_reason if task._cancel.is_set() else
                         "The coding tool could not start or its connection closed. Check the installation and sign-in.", error=True)
        finally:
            if watchdog:
                watchdog.cancel()
            if task.process and task.process.poll() is None:
                try:
                    if task._group:
                        task._group.stop()
                    else:
                        task.process.kill()
                    task.process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    logger.warning("Coding task %s cleanup is still pending", task.id)
            if task._group:
                task._group.close()
            if task.process:
                for stream in (task.process.stdin, task.process.stdout):
                    if stream:
                        try:
                            stream.close()
                        except OSError:
                            pass

    def send_input(self, task_id: str, user_input: str) -> Dict[str, Any]:
        with self._lock:
            if task_id not in self.tasks:
                return {"status": "error", "reason": "Task not found."}
        return {"status": "error", "reason": "These tasks run without an interactive terminal. Review CLI permissions in your project terminal, then start a new task."}

    def _request_stop(self, task: CodingTask, reason: str) -> bool:
        with self._lock:
            if task.end_time is not None:
                return False
            task._stop_reason = reason
            task.cancelled = reason == "Cancelled by user"
            task._cancel.set()
            task.last_message = "Stopping the coding task…"
            group = task._group
        try:
            if group:
                group.stop()
        except OSError:
            with self._lock:
                task.last_message = "The coding task could not be stopped. Try Cancel again."
            self._notify(task)
            return False
        self._notify(task)
        return True

    def cancel(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            task = self.tasks.get(task_id or self.active_task_id)
            if not task:
                return {"status": "error", "reason": "No active task to cancel."}
        if not self._request_stop(task, "Cancelled by user"):
            return {"status": "error", "reason": "The task has finished or could not be stopped.", "task": task.to_dict()}
        return {"status": "ok", "task": task.to_dict()}

    def get_status(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            task = self.tasks.get(task_id or self.active_task_id)
            return {"active_task": task.to_dict() if task else None, "state": task.state if task else TaskState.IDLE,
                    "history_count": len(self.tasks), "tools": self._tools(), "requires_workspace": True,
                    "interactive": False}

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            limit = max(0, min(int(limit), MAX_TASKS))
        except (ValueError, TypeError):
            limit = 20
        with self._lock:
            all_tasks = sorted(self.tasks.values(), key=lambda task: task.start_time, reverse=True)
            return [task.to_dict() for task in all_tasks[:limit]]
