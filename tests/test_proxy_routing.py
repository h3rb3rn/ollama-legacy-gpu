"""Integration test: ollama-proxy.py routing over two fake Ollama backends (no GPU, no network)."""
import http.client
import http.server
import importlib.util
import json
import socket
import threading
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


proxy = load('ollama-proxy')
prefix_router = load('prefix_router')


def fake_backend(name):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            self._send({"models": [{"name": "qwen3.6:35b"}]})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self._send({"backend": name})

        def _send(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def dead_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


class ProxyRoutingTest(unittest.TestCase):
    def setUp(self):
        self.a_server, self.a_url = fake_backend("a")
        self.b_server, self.b_url = fake_backend("b")
        self._saved = (proxy.BACKEND_URL, proxy.ROUTER, proxy.PREFIX_KEYS, proxy.maybe_optimize_model)
        # Auto-optimize would start a real benchmark run; the routing test does not need it.
        proxy.maybe_optimize_model = lambda model: ""
        self.httpd = proxy.ThreadedHTTPServer(("127.0.0.1", 0), proxy.ProxyHandler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        for server in (self.a_server, self.b_server):
            server.shutdown()
            server.server_close()
        proxy.BACKEND_URL, proxy.ROUTER, proxy.PREFIX_KEYS, proxy.maybe_optimize_model = self._saved

    def _route_over(self, urls):
        proxy.BACKEND_URL = urls[0]
        proxy.ROUTER = prefix_router.PrefixRouter(urls)
        proxy.PREFIX_KEYS = prefix_router.prefix_keys

    def _chat(self, messages):
        body = json.dumps({"model": "qwen3.6:35b", "messages": messages}).encode()
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("POST", "/api/chat", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        data = json.loads(resp.read())
        conn.close()
        return data["backend"]

    def test_routing_off_sends_everything_to_the_primary(self):
        proxy.BACKEND_URL = self.b_url
        proxy.ROUTER = None
        self.assertEqual(self._chat([{"role": "user", "content": "hi"}]), "b")

    def test_next_turn_of_a_conversation_returns_to_the_same_backend(self):
        self._route_over([self.a_url, self.b_url])
        turn1 = [{"role": "user", "content": "x" * 3000}]
        first = self._chat(turn1)
        turn2 = turn1 + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
        self.assertEqual(self._chat(turn2), first)

    def test_dead_backend_is_retried_on_the_other_one(self):
        dead = dead_url()
        self._route_over([dead, self.b_url])
        self.assertEqual(self._chat([{"role": "user", "content": "hi"}]), "b")
        # the dead backend is now marked down, so the next request does not try it first
        self.assertEqual(proxy.ROUTER.choose("qwen3.6:35b", [], 0), self.b_url)

    def test_backend_without_the_model_is_not_used(self):
        self._route_over([self.a_url, self.b_url])
        proxy.ROUTER.set_models(self.a_url, ["other:latest"])
        proxy.ROUTER.set_models(self.b_url, ["qwen3.6:35b"])
        self.assertEqual(self._chat([{"role": "user", "content": "hi"}]), "b")


if __name__ == '__main__':
    unittest.main()
