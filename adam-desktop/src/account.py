"""Firebase identity for the Windows companion, with system-browser Google OAuth."""

from __future__ import annotations

import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit
import webbrowser

import requests

import secure_store


FIREBASE_PROJECT_ID = "adam-ai1"
FIREBASE_API_KEY = "AIzaSyDM5YdYPWo_oAs4GoUWKQgj0QQoHh2GkEI"
IDENTITY_URL = "https://identitytoolkit.googleapis.com/v1/accounts:"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


class AccountError(RuntimeError):
    def __init__(self, message: str, code: str = "account_error"):
        super().__init__(message)
        self.code = code


_ERRORS = {
    "EMAIL_EXISTS": "An account already exists for this email. Sign in instead.",
    "EMAIL_NOT_FOUND": "The email or password is incorrect.",
    "INVALID_PASSWORD": "The email or password is incorrect.",
    "INVALID_LOGIN_CREDENTIALS": "The email or password is incorrect.",
    "INVALID_EMAIL": "Enter a valid email address.",
    "WEAK_PASSWORD": "Use a password with at least 6 characters.",
    "USER_DISABLED": "This account has been disabled. Contact support.",
    "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Please try again later.",
    "OPERATION_NOT_ALLOWED": "This sign-in method needs to be enabled in Firebase.",
    "API_KEY_INVALID": "Account services need to be configured. Contact support.",
    "INVALID_ID_TOKEN": "Your session expired. Sign in again.",
    "TOKEN_EXPIRED": "Your session expired. Sign in again.",
    "INVALID_REFRESH_TOKEN": "Your session expired. Sign in again.",
    "USER_NOT_FOUND": "Your session expired. Sign in again.",
    "INVALID_IDP_RESPONSE": "Google sign-in could not be verified. Please try again.",
    "FEDERATED_USER_ID_ALREADY_LINKED": "This Google account is already linked to another account.",
    "CREDENTIAL_TOO_OLD_LOGIN_AGAIN": "Please sign in again to continue.",
}


def _client_path() -> Path:
    override = os.environ.get("ADAM_GOOGLE_CLIENT_FILE")
    if override:
        return Path(override)
    base = (Path(sys.executable).parent if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parents[1] / "config" / "local")
    sidecar = base / "firebase-desktop-client.json"
    if sidecar.exists() or not getattr(sys, "frozen", False):
        return sidecar
    # PyInstaller one-file releases unpack bundled public native OAuth config
    # under _MEIPASS. An environment override or executable sidecar wins.
    return Path(getattr(sys, "_MEIPASS", base)) / "firebase-desktop-client.json"


def _google_client() -> dict:
    try:
        path = _client_path()
        if path.stat().st_size > 32 * 1024:
            raise ValueError()
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        client = raw.get("installed", {})
        if (not isinstance(client, dict) or not isinstance(client.get("client_id"), str)
                or not client["client_id"].endswith(".apps.googleusercontent.com")
                or not client["client_id"].startswith("759320300226-")
                or not isinstance(client.get("client_secret"), str)
                or not client["client_secret"]
                or client.get("project_id", FIREBASE_PROJECT_ID) != FIREBASE_PROJECT_ID):
            raise ValueError()
        return client
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise AccountError("Google sign-in needs the Desktop OAuth client for the ADAM Firebase project.",
                           "google_not_configured") from exc


class AccountService:
    def __init__(self, store=None, session=None, browser_open=None, clock=None):
        self._store = store if store is not None else secure_store
        self._http = session if session is not None else requests.Session()
        self._browser_open = browser_open if browser_open is not None else webbrowser.open
        self._clock = clock if clock is not None else time.time
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._epoch = 0
        self._session: dict | None = None
        self._error: str | None = None
        self._google_cancel: threading.Event | None = None
        self._google_pending = False
        try:
            raw = self._store.load_secret("firebase-session")
            if raw:
                record = json.loads(raw)
                if (not isinstance(record, dict) or not isinstance(record.get("user"), dict)
                        or not record["user"].get("uid") or not isinstance(record.get("refreshToken"), str)
                        or not isinstance(record.get("idToken"), str)
                        or not isinstance(record.get("expiresAt"), (float, int))):
                    raise ValueError()
                self._session = record
        except (ValueError, TypeError, secure_store.SecretStoreError):
            self._error = "Saved sign-in could not be restored. Please sign in again."

    def status(self) -> dict:
        try:
            _google_client()
            configured = True
        except AccountError:
            configured = False
        with self._lock:
            return {"authenticated": self._session is not None,
                    "user": dict(self._session["user"]) if self._session else None,
                    "google": {"configured": configured, "pending": self._google_pending},
                    "error": self._error}

    @property
    def user(self) -> dict | None:
        with self._lock:
            return dict(self._session["user"]) if self._session else None

    def _post(self, url: str, *, body: dict, form: bool = False) -> dict:
        try:
            response = self._http.post(url, **({"data": body} if form else {"json": body}),
                                       timeout=(5, 20), allow_redirects=False)
            result = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise AccountError("Could not reach account services. Check your connection and try again.", "network") from exc
        if not isinstance(result, dict):
            raise AccountError("Account services returned an unexpected response. Please try again.")
        if not response.ok or "error" in result:
            error = result.get("error", {})
            code = error.get("message", "") if isinstance(error, dict) else str(error)
            code = code.split(" : ")[0]
            raise AccountError(_ERRORS.get(code, "Sign-in could not be completed. Please try again."),
                               code.lower() if code in _ERRORS else "account_error")
        return result

    def _firebase(self, action: str, body: dict) -> dict:
        return self._post(IDENTITY_URL + action + "?key=" + FIREBASE_API_KEY, body=body)

    def _begin(self) -> int:
        with self._lock:
            self._epoch += 1
            if self._google_cancel:
                self._google_cancel.set()
            self._google_pending = False
            self._error = None
            return self._epoch

    def _commit(self, response: dict, epoch: int, fallback: dict | None = None) -> None:
        fallback = fallback or {}
        uid = response.get("localId") or response.get("user_id") or fallback.get("uid")
        id_token = response.get("idToken") or response.get("id_token")
        refresh = response.get("refreshToken") or response.get("refresh_token")
        try:
            duration = int(response.get("expiresIn", response.get("expires_in", 3600)))
            if not uid or not isinstance(uid, str) or not id_token or not refresh or not 1 <= duration <= 86400:
                raise ValueError()
        except (ValueError, TypeError) as exc:
            raise AccountError("Account services returned an incomplete session. Please sign in again.") from exc
        user = {"uid": uid, "email": response.get("email", fallback.get("email", "")),
                "displayName": response.get("displayName", fallback.get("displayName", "")),
                "photoURL": response.get("photoUrl", fallback.get("photoURL", ""))}
        record = {"user": user, "idToken": id_token, "refreshToken": refresh,
                  "expiresAt": self._clock() + duration}
        with self._lock:
            if epoch != self._epoch:
                raise AccountError("Sign-in was cancelled.", "cancelled")
            try:
                self._store.save_secret("firebase-session", json.dumps(record))
            except secure_store.SecretStoreError as exc:
                raise AccountError("Windows could not save your sign-in securely. Please try again.", "secure_storage") from exc
            self._session = record
            self._error = None

    def email_login(self, email: str, password: str, create: bool = False, name: str = "") -> dict:
        email = email.strip() if isinstance(email, str) else ""
        if not email or "@" not in email or len(email) > 254:
            raise AccountError("Enter a valid email address.", "invalid_email")
        if not isinstance(password, str) or not password or len(password) > 4096:
            raise AccountError("Enter your password.", "invalid_password")
        if create and len(password) < 6:
            raise AccountError(_ERRORS["WEAK_PASSWORD"], "weak_password")
        name = name.strip()[:80] if isinstance(name, str) else ""
        epoch = self._begin()
        try:
            result = self._firebase("signUp" if create else "signInWithPassword",
                                    {"email": email, "password": password, "returnSecureToken": True})
            profile_warning = None
            if create and name:
                with self._lock:
                    if epoch != self._epoch:
                        raise AccountError("Sign-in was cancelled.", "cancelled")
                try:
                    updated = self._firebase("update", {"idToken": result["idToken"], "displayName": name,
                                                         "returnSecureToken": True})
                    result.update(updated)
                except AccountError:
                    # Account creation already succeeded. Preserve the usable
                    # session if the optional profile update loses connectivity.
                    profile_warning = "Your account is ready. Your profile name could not be saved online yet."
            self._commit(result, epoch, {"email": email, "displayName": name})
            if profile_warning:
                with self._lock:
                    if epoch == self._epoch:
                        self._error = profile_warning
            return self.status()
        except AccountError as exc:
            with self._lock:
                if epoch == self._epoch:
                    self._error = str(exc)
            raise

    def reset_password(self, email: str) -> dict:
        if not isinstance(email, str) or not email.strip() or "@" not in email or len(email) > 254:
            raise AccountError("Enter a valid email address.", "invalid_email")
        try:
            self._firebase("sendOobCode", {"requestType": "PASSWORD_RESET", "email": email.strip()})
        except AccountError as exc:
            if exc.code != "email_not_found":
                raise
        return {"message": "If an account uses this email, a password reset link will arrive shortly."}

    def signout(self) -> dict:
        with self._lock:
            self._begin()
            self._session = None
            try:
                self._store.delete_secret("firebase-session")
            except secure_store.SecretStoreError as exc:
                self._error = "Sign-in was cleared from this session, but Windows could not remove the saved credential."
                raise AccountError(self._error, "secure_storage") from exc
            return self.status()

    def id_token(self) -> str:
        with self._refresh_lock:
            with self._lock:
                if self._session is None:
                    raise AccountError("Sign in to sync your ADAM account.", "signed_out")
                session = dict(self._session)
                epoch = self._epoch
                if session["expiresAt"] > self._clock() + 60:
                    return session["idToken"]
            try:
                refreshed = self._post("https://securetoken.googleapis.com/v1/token?key=" + FIREBASE_API_KEY,
                                       body={"grant_type": "refresh_token", "refresh_token": session["refreshToken"]}, form=True)
                self._commit(refreshed, epoch, session["user"])
            except AccountError as exc:
                with self._lock:
                    if epoch == self._epoch:
                        self._error = str(exc)
                        if exc.code in {"invalid_refresh_token", "token_expired", "user_disabled", "user_not_found"}:
                            self._session = None
                            self._epoch += 1
                            self._store.delete_secret("firebase-session")
                raise
            with self._lock:
                if epoch != self._epoch or not self._session:
                    raise AccountError("Sign-in was cancelled.", "cancelled")
                return self._session["idToken"]

    def cancel_google(self) -> dict:
        self._begin()
        return self.status()

    def start_google(self) -> dict:
        client = _google_client()
        epoch = self._begin()
        cancel = threading.Event()
        with self._lock:
            self._google_cancel = cancel
            self._google_pending = True
        threading.Thread(target=self._google_flow, args=(client, epoch, cancel),
                         name="ADAM Google sign-in", daemon=True).start()
        return self.status()

    def _google_flow(self, client: dict, epoch: int, cancel: threading.Event) -> None:
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).decode("ascii").rstrip("=")
        state = secrets.token_urlsafe(32)
        callback: dict = {}

        class LoopbackServer(HTTPServer):
            def get_request(self):
                connection, address = super().get_request()
                connection.settimeout(2)
                return connection, address

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass  # The callback URL contains a one-time authorization code.

            def do_GET(self):
                parsed = urlsplit(self.path)
                query = parse_qs(parsed.query)
                valid = (parsed.path == "/callback" and self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"
                         and secrets.compare_digest(query.get("state", [""])[0], state))
                if not valid or cancel.is_set():
                    self.send_error(400, "Invalid or expired sign-in request")
                    return
                callback.update({"code": query.get("code", [""])[0], "error": query.get("error", [""])[0]})
                content = ("<!doctype html><html><meta name='viewport' content='width=device-width,initial-scale=1'>"
                           "<title>ADAM sign-in</title><body style='background:#101010;color:#eee;font:18px system-ui;"
                           "padding:12vh 8vw'><h1>Return to ADAM</h1><p>You can close this tab and continue in the app.</p></body></html>").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(content)

        try:
            with LoopbackServer(("127.0.0.1", 0), Handler) as server:
                server.timeout = 0.3
                redirect = f"http://127.0.0.1:{server.server_port}/callback"
                url = GOOGLE_AUTH_URL + "?" + urlencode({"client_id": client["client_id"], "redirect_uri": redirect,
                      "response_type": "code", "scope": "openid email profile", "state": state,
                      "code_challenge": challenge, "code_challenge_method": "S256", "prompt": "select_account"})
                if not self._browser_open(url):
                    raise AccountError("Your browser could not be opened. Set a default browser and try again.", "browser")
                deadline = time.monotonic() + 180
                while not callback and not cancel.is_set() and time.monotonic() < deadline:
                    server.handle_request()
                if cancel.is_set():
                    return
                if not callback:
                    raise AccountError("Google sign-in timed out. Please try again.", "timeout")
                if callback.get("error") or not callback.get("code"):
                    raise AccountError("Google sign-in was cancelled.", "cancelled")
                tokens = self._post(GOOGLE_TOKEN_URL, form=True, body={"client_id": client["client_id"],
                    "client_secret": client["client_secret"], "code": callback["code"], "code_verifier": verifier,
                    "grant_type": "authorization_code", "redirect_uri": redirect})
                if cancel.is_set():
                    return
                access_token = tokens.get("access_token")
                if not isinstance(access_token, str) or not access_token:
                    raise AccountError("Google could not verify this sign-in. Please try again.")
                # Firebase supports Google's access-token credential directly.
                # Do not ask it to accept an ID token whose audience is this
                # Desktop OAuth client rather than its configured web client.
                result = self._firebase("signInWithIdp", {"postBody": urlencode({"access_token": access_token,
                    "providerId": "google.com"}), "requestUri": redirect, "returnSecureToken": True,
                    "returnIdpCredential": False})
                self._commit(result, epoch)
        except (AccountError, OSError) as exc:
            with self._lock:
                if epoch == self._epoch:
                    self._error = str(exc) if isinstance(exc, AccountError) else "The local sign-in callback could not be opened. Please try again."
        finally:
            with self._lock:
                if epoch == self._epoch:
                    self._google_pending = False
                    self._google_cancel = None
