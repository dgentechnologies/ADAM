"""Real loopback sockets: occupied ports and exact desktop ownership."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest

from desktop_runtime import (activate_instance, read_instance, remove_instance,
                             reserve_listener, verify_instance, write_instance)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "desktop-instance.json"
        self.identity = "a" * 48
        self.payload = {"app": "ADAM Companion", "instance_id": self.identity, "pid": os.getpid()}
        self.redirect = False
        self.activations = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.send_response(302 if fixture.redirect else 200)
                self.send_header("Location", "http://127.0.0.1:1/wrong-app")
                self.end_headers()
                self.wfile.write(json.dumps(fixture.payload).encode())

            def do_POST(self):
                fixture.activations.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def test_occupied_port_reserves_a_different_socket(self):
        listener = reserve_listener(self.port)
        self.addCleanup(listener.close)
        actual = listener.getsockname()[1]
        self.assertNotEqual(actual, self.port)
        contender = socket.socket()
        self.addCleanup(contender.close)
        with self.assertRaises(OSError):
            contender.bind(("127.0.0.1", actual))

    def test_owned_service_is_verified(self):
        self.assertTrue(verify_instance(self.port, self.identity, os.getpid(), "ADAM Companion"))

    def test_same_product_name_from_another_instance_is_rejected(self):
        self.payload["instance_id"] = "other" * 12
        self.assertFalse(verify_instance(self.port, self.identity, os.getpid(), "ADAM Companion"))

    def test_wrong_process_is_rejected(self):
        self.payload["pid"] += 1
        self.assertFalse(verify_instance(self.port, self.identity, os.getpid(), "ADAM Companion"))

    def test_redirect_is_not_followed(self):
        self.redirect = True
        self.assertFalse(verify_instance(self.port, self.identity, os.getpid(), "ADAM Companion"))

    def test_second_launch_uses_runtime_port_and_identity(self):
        write_instance(self.path, self.port, self.identity)
        self.assertTrue(activate_instance(self.path, "ADAM Companion", seconds=1))
        self.assertEqual(self.activations, [{"instance_id": self.identity}])

    def test_stale_record_never_activates_another_app(self):
        write_instance(self.path, self.port, "b" * 48)
        self.assertFalse(activate_instance(self.path, "ADAM Companion", seconds=.1))
        self.assertEqual(self.activations, [])

    def test_cleanup_removes_only_owned_record_and_rejects_bad_data(self):
        write_instance(self.path, self.port, self.identity)
        remove_instance(self.path, "another")
        self.assertIsNotNone(read_instance(self.path))
        remove_instance(self.path, self.identity)
        self.assertFalse(self.path.exists())
        self.path.write_text('{"port": "bad"}', encoding="utf-8")
        self.assertIsNone(read_instance(self.path))
