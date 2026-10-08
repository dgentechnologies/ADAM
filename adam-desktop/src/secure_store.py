"""Small atomic secret store protected by the current Windows user's DPAPI key.

No plaintext fallback is provided. Tests may inject a cipher explicitly; the app
always uses DPAPI. Tokens never belong in settings.json or browser storage.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import re
import tempfile
import threading


class SecretStoreError(RuntimeError):
    pass


def data_directory() -> Path:
    override = os.environ.get("ADAM_DATA_DIR")
    return Path(override) if override else Path(os.environ.get("APPDATA", str(Path.home()))) / "ADAM"


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


class _DpapiCipher:
    def _transform(self, value: bytes, encrypt: bool) -> bytes:
        if os.name != "nt":
            raise SecretStoreError("Secure credentials require Windows Data Protection.")
        source_buffer = ctypes.create_string_buffer(value)
        source = _Blob(len(value), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)))
        entropy_buffer = ctypes.create_string_buffer(b"ADAM.Companion.credentials.v1")
        entropy = _Blob(len(entropy_buffer.raw), ctypes.cast(entropy_buffer, ctypes.POINTER(ctypes.c_ubyte)))
        result = _Blob()
        crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        if encrypt:
            call = crypt32.CryptProtectData
            call.argtypes = [ctypes.POINTER(_Blob), wintypes.LPCWSTR, ctypes.POINTER(_Blob),
                             ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
            success = call(ctypes.byref(source), "ADAM credential", ctypes.byref(entropy),
                           None, None, 0x1, ctypes.byref(result))
        else:
            call = crypt32.CryptUnprotectData
            call.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.POINTER(_Blob),
                             ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
            success = call(ctypes.byref(source), None, ctypes.byref(entropy),
                           None, None, 0x1, ctypes.byref(result))
        if not success:
            raise SecretStoreError("Windows could not access the protected credential. Sign in again.")
        try:
            return ctypes.string_at(result.pbData, result.cbData)
        finally:
            kernel32.LocalFree(result.pbData)

    def encrypt(self, value: bytes) -> bytes:
        return self._transform(value, True)

    def decrypt(self, value: bytes) -> bytes:
        return self._transform(value, False)


class SecretStore:
    def __init__(self, directory: Path | None = None, cipher=None):
        self.directory = Path(directory) if directory is not None else data_directory() / "credentials"
        self.cipher = cipher if cipher is not None else _DpapiCipher()
        self._lock = threading.RLock()

    def _path(self, name: str) -> Path:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", name):
            raise ValueError("Invalid credential name.")
        return self.directory / (name + ".dpapi")

    def load_secret(self, name: str) -> str | None:
        with self._lock:
            path = self._path(name)
            if not path.exists():
                return None
            try:
                if path.stat().st_size > 128 * 1024:
                    raise SecretStoreError("The protected credential is damaged. Sign in again.")
                return self.cipher.decrypt(path.read_bytes()).decode("utf-8")
            except (OSError, UnicodeError) as exc:
                raise SecretStoreError("The protected credential could not be read. Sign in again.") from exc

    def save_secret(self, name: str, value: str) -> None:
        if not isinstance(value, str) or len(value.encode("utf-8")) > 64 * 1024:
            raise ValueError("Invalid credential value.")
        with self._lock:
            path = self._path(name)
            encrypted = self.cipher.encrypt(value.encode("utf-8"))
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".credential-", delete=False) as file:
                    temporary = Path(file.name)
                    file.write(encrypted)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, path)
            except OSError as exc:
                raise SecretStoreError("The protected credential could not be saved.") from exc
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def delete_secret(self, name: str) -> None:
        with self._lock:
            try:
                self._path(name).unlink(missing_ok=True)
            except OSError as exc:
                raise SecretStoreError("The protected credential could not be removed.") from exc


_stores: dict[str, SecretStore] = {}
_stores_lock = threading.Lock()


def _store() -> SecretStore:
    path = data_directory() / "credentials"
    with _stores_lock:
        return _stores.setdefault(str(path.resolve()), SecretStore(path))


def load_secret(name: str) -> str | None:
    return _store().load_secret(name)


def save_secret(name: str, value: str) -> None:
    _store().save_secret(name, value)


def delete_secret(name: str) -> None:
    _store().delete_secret(name)
