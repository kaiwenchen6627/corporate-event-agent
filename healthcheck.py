#!/usr/bin/env python3
"""Container healthcheck: succeed when the HTTP server answers at all.

A 401 (returned when AGENT_AUTH is set and no credentials are supplied) still
proves the process is alive and serving, so only connection-level failures are
treated as unhealthy. Uses only the standard library so it works in the slim
runtime image without extra packages.
"""
import os
import sys
import urllib.error
import urllib.request

port = os.environ.get("PORT", "8080")
url = f"http://127.0.0.1:{port}/health"

try:
    urllib.request.urlopen(url, timeout=3)
except urllib.error.HTTPError:
    pass  # 401 / 403 — server is up, we simply did not send credentials
except Exception:
    sys.exit(1)
