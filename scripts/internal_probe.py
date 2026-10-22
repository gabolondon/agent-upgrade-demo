"""Run inside web-gateway: GET the edge over the internal network, retrying while it is 502."""

import time
import urllib.error
import urllib.request

code = 0
for _ in range(60):
    try:
        code = urllib.request.urlopen("http://edge:80/health").status
    except urllib.error.HTTPError as e:
        code = e.code
    if code != 502:
        break
    time.sleep(1)
print("GET /health -> HTTP", code)
