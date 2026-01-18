#!/usr/bin/env python3
import os, sys, time, json, hmac, hashlib, mimetypes
import requests

SECRET = os.environ.get("RENDEREXPO_HMAC_SECRET", "")
if not SECRET:
    print("ERROR: RENDEREXPO_HMAC_SECRET is not set")
    sys.exit(1)

def _to_bytes(body):
    if body is None:
        return b""
    if isinstance(body, bytes):
        return body
    return str(body).encode("utf-8")

def sign(body_bytes: bytes):
    ts = str(int(time.time()))
    nonce = os.urandom(16).hex()

    # Server expects EXACT: f"{ts}\n{nonce}\n{body}"
    msg = (ts + "\n" + nonce + "\n").encode("utf-8") + body_bytes
    sig = hmac.new(SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return ts, nonce, sig

def send_prepared(prepped, timeout=600):
    body_bytes = _to_bytes(prepped.body)
    ts, nonce, sig = sign(body_bytes)

    prepped.headers["X-RENDEREXPO-TIMESTAMP"] = ts
    prepped.headers["X-RENDEREXPO-NONCE"] = nonce
    prepped.headers["X-RENDEREXPO-SIGNATURE"] = sig

    s = requests.Session()
    resp = s.send(prepped, timeout=timeout)

    print("STATUS:", resp.status_code)
    ctype = resp.headers.get("content-type", "")
    if "application/json" in ctype:
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text[:4000])
    else:
        # For images/binary responses, don't spam output
        if "image/" in ctype or "application/octet-stream" in ctype:
            print(f"(binary response) content-type={ctype}, bytes={len(resp.content)}")
        else:
            print(resp.text[:4000])
    return resp

def request_json(method, url, json_body=None):
    req = requests.Request(method=method, url=url, json=json_body)
    prepped = req.prepare()
    return send_prepared(prepped)

def guess_mime(path):
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"

def request_multipart(method, url, fields: dict, file_fields: dict):
    """
    fields: {"prompt": "text", "mode": "insert_only", ...}
    file_fields: {"image": "/path/to/file.png", "images[]": "/path/a.jpg", ...}

    Supports repeated keys by allowing comma-separated list in file path:
      images=/a.png,/b.png,/c.png
    """
    data = {}
    for k, v in (fields or {}).items():
        if v is None:
            continue
        data[k] = str(v)

    files = []
    for k, v in (file_fields or {}).items():
        if v is None:
            continue
        # allow multiple files: key=/a.png,/b.png
        paths = [p.strip() for p in str(v).split(",") if p.strip()]
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"File not found: {p}")
            fn = os.path.basename(p)
            mime = guess_mime(p)
            f = open(p, "rb")
            files.append((k, (fn, f, mime)))

    req = requests.Request(method=method, url=url, data=data, files=files)
    prepped = req.prepare()
    try:
        return send_prepared(prepped)
    finally:
        # close opened file handles
        for item in files:
            try:
                item[1][1].close()
            except Exception:
                pass

def parse_kv_list(kvs):
    """
    Parse ["a=1","b=hello"] into dict {"a":"1","b":"hello"}.
    """
    out = {}
    for kv in kvs:
        if "=" not in kv:
            raise ValueError(f"Expected key=value, got: {kv}")
        k, v = kv.split("=", 1)
        out[k] = v
    return out

def usage():
    print(
"""Usage:

JSON requests:
  python rx_call.py GET  http://0.0.0.0:8002/openapi.json
  python rx_call.py POST http://0.0.0.0:8002/api/sd35/render '{"prompt":"x","width":512,"height":512}'

Multipart (file upload) requests:
  python rx_call.py MULTIPART POST  <url>  --field key=value --file key=/path/img.png

Examples:
  python rx_call.py MULTIPART POST http://0.0.0.0:8002/api/sd35/render-from-image \\
      --field prompt="modern interior, neutral palette" \\
      --field strength=0.7 \\
      --file image=/workspace-data/RENDEREXPO-AI-Studio-Backend/sd35_renderexpo_pro_v2_latest.png

Multiple files (repeat key, or comma-separated):
  python rx_call.py MULTIPART POST http://0.0.0.0:8002/api/vr/reconstruct/plan \\
      --file images=/path/a.png,/path/b.png,/path/c.png \\
      --field prompt="modern living room"

Saves binary response:
  python rx_call.py GETBIN <url> <output_file>
"""
    )
    sys.exit(1)

def getbin(url, out_path):
    req = requests.Request(method="GET", url=url)
    prepped = req.prepare()
    resp = send_prepared(prepped)
    if resp.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(resp.content)
        print(f"Saved -> {out_path}")
    else:
        print("GETBIN failed")
    return resp

if __name__ == "__main__":
    if len(sys.argv) < 3:
        usage()

    # Special mode: GETBIN <url> <output_file>
    if sys.argv[1].upper() == "GETBIN":
        if len(sys.argv) < 4:
            usage()
        getbin(sys.argv[2], sys.argv[3])
        sys.exit(0)

    # Special mode: MULTIPART POST|PUT|PATCH <url> [--field a=b] [--file k=/path]
    if sys.argv[1].upper() == "MULTIPART":
        if len(sys.argv) < 4:
            usage()
        method = sys.argv[2].upper()
        url = sys.argv[3]

        fields_kv = []
        files_kv = []

        i = 4
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == "--field":
                i += 1
                fields_kv.append(sys.argv[i])
            elif arg == "--file":
                i += 1
                files_kv.append(sys.argv[i])
            else:
                raise ValueError(f"Unknown arg: {arg}")
            i += 1

        fields = parse_kv_list(fields_kv)
        file_fields = parse_kv_list(files_kv)
        request_multipart(method, url, fields, file_fields)
        sys.exit(0)

    # Default JSON mode
    method = sys.argv[1].upper()
    url = sys.argv[2]

    if method in ("GET", "DELETE"):
        request_json(method, url)
        sys.exit(0)

    if len(sys.argv) >= 4:
        payload = json.loads(sys.argv[3])
        request_json(method, url, json_body=payload)
    else:
        request_json(method, url)
