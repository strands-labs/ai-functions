"""Serve an inspection archive locally; no model or benchmark dependencies required."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .capture import Recorder, replay


def make_server(recorder: Recorder, port: int = 8768) -> ThreadingHTTPServer:
    """Bind a read-only viewer to loopback. Only fixed routes are served."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlsplit(self.path)
            query = parse_qs(route.query)
            content_type = "application/json; charset=utf-8"
            try:
                if route.path == "/":
                    body = Path(__file__).with_name("index.html").read_bytes()
                    content_type = "text/html; charset=utf-8"
                elif route.path == "/api/state":
                    seq = int(query["seq"][0]) if "seq" in query else None
                    state = recorder.state(seq)
                    journal = recorder.journal()
                    state["last_seq"] = journal[-1]["seq"] if journal else 0
                    body = json.dumps(state).encode()
                elif route.path == "/api/replay":
                    body = json.dumps(replay(recorder.journal())).encode()
                elif route.path == "/api/export":
                    body = json.dumps({"format": "skillify-observation-v1", "events": recorder.journal()}).encode()
                else:
                    self.send_error(404)
                    return
            except (ValueError, KeyError) as exc:
                self.send_error(400, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'",
            )
            if route.path == "/api/export":
                self.send_header("Content-Disposition", 'attachment; filename="skills-recording.json"')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="capture directory containing skills.sqlite3")
    parser.add_argument("--port", type=int, default=8768, help="loopback port; 0 chooses a free port")
    imports = parser.add_mutually_exclusive_group()
    imports.add_argument("--import-skill", type=Path, help="Markdown file or directory containing SKILL.md")
    imports.add_argument("--import-library", type=Path)
    imports.add_argument("--import-json", type=Path)
    imports.add_argument("--import-legacy", type=Path, help="earlier ScienceWorld inspection.sqlite3")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    recorder = Recorder(args.data)
    if args.import_skill:
        recorder.import_skill(args.import_skill)
    elif args.import_library:
        recorder.import_library(args.import_library)
    elif args.import_json:
        recorder.import_json(args.import_json)
    elif args.import_legacy:
        recorder.import_legacy(args.import_legacy)
    server = make_server(recorder, args.port)
    print(f"Skill inspection: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
