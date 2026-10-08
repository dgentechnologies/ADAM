"""All Windows hardware access stays on one COM-owning worker thread."""
import queue
import threading
import time
import gc


class WindowsHardwareWorker(threading.Thread):
    def __init__(self):
        # comtypes initializes the thread that imports it first and registers
        # process-exit cleanup. Import on the creating (application) thread;
        # the worker owns its separate, explicitly balanced apartment below.
        import comtypes
        # The brightness package creates a module-level COM locator on first
        # import. Keep that process-lifetime object on the application thread.
        import screen_brightness_control
        super().__init__(name="ADAM hardware", daemon=True)
        self.cmd_queue = queue.Queue(maxsize=32)
        self.stopping = threading.Event()
        self._audio = None
        self._state = {"volume": None, "brightness": None, "volume_error": "Checking audio…",
                       "brightness_error": "Checking display…"}
        self._lock = threading.Lock()
        self.start()

    def _endpoint(self):
        if self._audio is None:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            from comtypes import CLSCTX_ALL
            device = AudioUtilities.GetSpeakers()
            if device is None:
                raise RuntimeError("No active Windows audio output")
            self._audio = getattr(device, "EndpointVolume", None)
            if self._audio is None:
                interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self._audio = interface.QueryInterface(IAudioEndpointVolume)
        return self._audio

    def _execute(self, task, args):
        if task == "get_vol":
            return round(self._endpoint().GetMasterVolumeLevelScalar() * 100)
        if task == "set_vol":
            self._endpoint().SetMasterVolumeLevelScalar(args[0] / 100, None)
            return self._execute("get_vol", ())
        if task == "set_mute":
            self._endpoint().SetMute(bool(args[0]), None)
            return bool(self._endpoint().GetMute())
        import screen_brightness_control as sbc
        if task == "set_bri":
            sbc.set_brightness(args[0])
        values = sbc.get_brightness()
        if not values:
            raise RuntimeError("This display does not expose brightness control")
        return int(values[0] if isinstance(values, list) else values)

    def _refresh(self):
        for field, task in (("volume", "get_vol"), ("brightness", "get_bri")):
            try:
                value = self._execute(task, ())
                error = ""
            except Exception:
                value = None
                error = "No supported audio output" if field == "volume" else "Brightness control is not supported by this display"
                if field == "volume":
                    self._audio = None
            with self._lock:
                self._state[field] = value
                self._state[field + "_error"] = error

    def run(self):
        initialized = False
        try:
            import comtypes
            comtypes.CoInitialize()
            initialized = True
            next_refresh = 0
            while not self.stopping.is_set():
                if time.monotonic() >= next_refresh:
                    self._refresh()
                    next_refresh = time.monotonic() + 4
                try:
                    task, args, response, deadline = self.cmd_queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                try:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Hardware request expired before execution")
                    value = self._execute(task, args)
                    response.put((True, value))
                    next_refresh = 0
                except Exception as error:
                    self._audio = None
                    response.put((False, str(error)))
                finally:
                    self.cmd_queue.task_done()
        finally:
            # Release COM interfaces before releasing their owning apartment.
            self._audio = None
            gc.collect()
            if initialized:
                comtypes.CoUninitialize()

    def call_sync(self, task, args=(), timeout=4):
        if not self.is_alive() or self.stopping.is_set():
            raise RuntimeError("Windows hardware service is not running")
        response = queue.Queue(maxsize=1)
        try:
            self.cmd_queue.put_nowait((task, args, response, time.monotonic() + timeout))
        except queue.Full:
            raise RuntimeError("Hardware controls are busy; try again") from None
        try:
            ok, value = response.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("Windows did not acknowledge the hardware request") from None
        if not ok:
            raise RuntimeError(value)
        return value

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def stop(self):
        self.stopping.set()
