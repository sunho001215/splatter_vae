"""HTTP(S) downloads with byte-range resume, streaming verification against official checksums and a shared rate limiter.

One connection per file (no internal parallelism). Redirects are followed by hand so that an ``Authorization`` header
is only ever sent to the host it was configured for (never to a CDN a redirect points at). A partial temp file is
re-hashed and continued with ``Range: bytes=<n>-``; a server that ignores the range restarts the file from zero.

Checksum kinds (all published by the sources themselves):

- ``md5``: GCS object md5 (JSON API ``md5Hash``, hex here);
- ``sha256``: Hugging Face LFS object id;
- ``git_sha1``: git blob id of a non-LFS file (Hugging Face tree ``oid``, GitHub contents ``sha``);
- ``lfs_pointer_sha1``: git blob id of the LFS *pointer* of a file whose sha256 a gated repository hides from anonymous
  listings; the pointer is rebuilt from the content's sha256 and size.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import ssl
import threading
import time
import urllib.parse

USER_AGENT = "s4d-datasets/1"
CHUNK = 1 << 20
CONNECT_TIMEOUT = 30.0
READ_TIMEOUT = 60.0  # a stalled socket raises after this many seconds without data
MAX_REDIRECTS = 8
SECURE_SCHEMES = ("https",)  # every request and redirect; tests add "http" for their local server
LFS_POINTER = "version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n"


class TransferError(RuntimeError):
    """A failed attempt (network, HTTP status, size or checksum); the caller retries with backoff."""


class Interrupted(RuntimeError):
    """The owner asked this transfer to stop; the partial file is kept for resumption."""


class AuthRequired(TransferError):
    """HTTP 401/403 from a gated source: needs a token (never retried blindly)."""


def gcs_md5_hex(value: str) -> str:
    return base64.b64decode(value).hex()


class Verifier:
    """Streaming digest for one checksum kind; ``expected`` is the published value."""

    def __init__(self, checksum: dict, size: int):
        if len(checksum) != 1:
            raise ValueError(f"exactly one checksum kind expected, got {checksum}")
        (self.kind, self.expected), = checksum.items()
        self.size = int(size)
        if self.kind == "md5":
            self._h = hashlib.md5(usedforsecurity=False)
        elif self.kind in ("sha256", "lfs_pointer_sha1"):
            self._h = hashlib.sha256()
        elif self.kind == "git_sha1":
            self._h = hashlib.sha1(f"blob {self.size}\0".encode(), usedforsecurity=False)
        else:
            raise ValueError(f"unknown checksum kind {self.kind!r}")

    def update(self, data) -> None:
        self._h.update(data)

    def value(self) -> str:
        digest = self._h.hexdigest()
        if self.kind != "lfs_pointer_sha1":
            return digest
        pointer = LFS_POINTER.format(sha256=digest, size=self.size).encode()
        return hashlib.sha1(b"blob %d\0" % len(pointer) + pointer, usedforsecurity=False).hexdigest()

    def matches(self) -> bool:
        return self.value() == self.expected


def hash_file(path, checksum: dict, size: int) -> bool:
    verifier = Verifier(checksum, size)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            verifier.update(block)
    return verifier.matches()


class RateLimiter:
    """Token bucket shared by all transfers of the process; ``rate`` None means unlimited."""

    def __init__(self, rate: float | None = None, clock=time.monotonic, sleep=time.sleep):
        self._lock = threading.Lock()
        self._clock, self._sleep = clock, sleep
        self.rate = None
        self._tokens = 0.0
        self._stamp = clock()
        self.set_rate(rate)

    def set_rate(self, rate: float | None) -> None:
        with self._lock:
            self.rate = None if rate is None else max(1.0, float(rate))
            self._tokens = 0.0
            self._stamp = self._clock()

    def acquire(self, amount: int) -> None:
        while True:
            with self._lock:
                if self.rate is None:
                    return
                now = self._clock()
                self._tokens = min(self.rate * 0.5, self._tokens + (now - self._stamp) * self.rate)
                self._stamp = now
                if self._tokens >= 0:
                    self._tokens -= amount  # may go negative: the debt is paid by the next callers' wait
                    return
                wait = -self._tokens / self.rate
            self._sleep(min(wait, 1.0))


def _connection(url: urllib.parse.SplitResult, timeout: float):
    port = url.port
    if url.scheme == "https":
        return http.client.HTTPSConnection(url.hostname, port, timeout=timeout, context=ssl.create_default_context())
    if url.scheme == "http":
        return http.client.HTTPConnection(url.hostname, port, timeout=timeout)
    raise TransferError(f"unsupported URL scheme {url.scheme!r}")


def _check_scheme(url: urllib.parse.SplitResult) -> None:
    if url.scheme not in SECURE_SCHEMES:
        raise TransferError(f"refused non-https URL scheme {url.scheme!r}")


def open_url(url: str, *, headers: dict | None = None, auth: tuple[str, str] | None = None,
             timeout: float = READ_TIMEOUT):
    """GET with manual redirects (https only); ``auth`` = (host, header value) is sent only to that exact host.
    Returns (connection, response); the caller closes the connection. Header/URL validation errors are reported
    without their content (they may contain the token)."""
    headers = dict(headers or {})
    for _ in range(MAX_REDIRECTS + 1):
        parts = urllib.parse.urlsplit(url)
        _check_scheme(parts)
        request_headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity", **headers}
        if auth and parts.hostname == auth[0]:
            request_headers["Authorization"] = auth[1]
        connection = _connection(parts, CONNECT_TIMEOUT)
        try:
            path = urllib.parse.urlunsplit(("", "", parts.path or "/", parts.query, ""))
            connection.request("GET", path, headers=request_headers)
            if connection.sock is not None:
                connection.sock.settimeout(timeout)
            response = connection.getresponse()
        except (OSError, http.client.HTTPException) as error:
            connection.close()
            raise TransferError(f"{type(error).__name__}: {error}") from error
        except (ValueError, TypeError):
            connection.close()
            raise TransferError("request rejected: invalid header or URL (details withheld)") from None
        if response.status in (301, 302, 303, 307, 308):
            location = response.getheader("Location")
            response.read()
            connection.close()
            if not location:
                raise TransferError(f"redirect {response.status} without Location")
            url = urllib.parse.urljoin(url, location)
            continue
        return connection, response
    raise TransferError("too many redirects")


def fetch_json(url: str, *, auth=None, attempts: int = 4) -> tuple[object, dict]:
    """Small JSON GET (listings) with bounded retries; returns (payload, lower-cased headers)."""
    last = None
    for attempt in range(attempts):
        connection = None
        try:
            connection, response = open_url(url, headers={"Accept": "application/json"}, auth=auth)
            body = response.read()
            if response.status != 200:
                raise TransferError(f"HTTP {response.status} for {url}: {body[:200]!r}")
            return json.loads(body), {k.lower(): v for k, v in response.getheaders()}
        except (TransferError, OSError, http.client.HTTPException, ValueError) as error:
            last = error
            time.sleep(min(30.0, 2.0 * 2**attempt))
        finally:
            if connection is not None:
                connection.close()
    raise TransferError(f"listing failed after {attempts} attempts: {last}")


def fetch_bytes(url: str, checksum: dict, size: int, *, auth=None, attempts: int = 4) -> bytes:
    """Small verified file held in memory (the flow shard manifest the plan needs)."""
    last = None
    for attempt in range(attempts):
        connection = None
        try:
            connection, response = open_url(url, auth=auth)
            body = response.read()
            if response.status != 200:
                raise TransferError(f"HTTP {response.status} for {url}")
            verifier = Verifier(checksum, size)
            verifier.update(body)
            if len(body) != size or not verifier.matches():
                raise TransferError(f"{url}: size or checksum mismatch")
            return body
        except (TransferError, OSError, http.client.HTTPException) as error:
            last = error
            time.sleep(min(30.0, 2.0 * 2**attempt))
        finally:
            if connection is not None:
                connection.close()
    raise TransferError(f"fetch failed after {attempts} attempts: {last}")


def _content_range_start(response) -> tuple[int, int] | None:
    value = response.getheader("Content-Range") or ""
    if not value.startswith("bytes ") or "/" not in value:
        return None
    span, total = value[6:].split("/", 1)
    start = span.split("-", 1)[0]
    try:
        return int(start), int(total)
    except ValueError:
        return None


def download(item: dict, temp_path, *, limiter: RateLimiter, on_progress, should_stop, auth=None) -> int:
    """Download (or continue) ``item`` into ``temp_path`` and verify size and checksum. Returns the number of bytes
    fetched by this call. ``on_progress(fetched_delta, bytes_on_disk)`` reports absolute temp-file progress (reset to
    0 when a server ignores the range). Raises Interrupted (partial kept), AuthRequired or TransferError (a corrupt
    file is removed so the next attempt starts clean). The temp file is fsynced before returning."""
    size = int(item["size"])
    verifier = Verifier(item["checksum"], size)
    have = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
    if have > size:
        os.unlink(temp_path)
        have = 0
    if have:
        with open(temp_path, "rb") as handle:
            for block in iter(lambda: handle.read(CHUNK), b""):
                verifier.update(block)
    fetched = 0
    if have < size:
        headers = {"Range": f"bytes={have}-"} if have else {}
        connection, response = open_url(item["url"], headers=headers, auth=auth)
        try:
            if response.status in (401, 403):
                raise AuthRequired(f"HTTP {response.status}: gated source needs a token")
            if have and response.status == 200:  # the server ignored the range: start over
                verifier = Verifier(item["checksum"], size)
                have = 0
                on_progress(0, 0)
            elif have and response.status == 206:
                span = _content_range_start(response)
                if span != (have, size):
                    raise TransferError(f"unexpected Content-Range {response.getheader('Content-Range')!r}")
            elif response.status != 200:
                raise TransferError(f"HTTP {response.status}")
            generation = item.get("generation")
            if generation and response.getheader("x-goog-generation") not in (None, str(generation)):
                raise TransferError("object generation changed at the source")
            expected = size - have
            length = response.getheader("Content-Length")
            if length is not None and int(length) != expected:
                raise TransferError(f"Content-Length {length} != {expected} remaining bytes")
            with open(temp_path, "r+b" if have else "wb") as out:
                out.seek(have)
                out.truncate()
                while have < size:
                    if should_stop():
                        raise Interrupted("stop requested")
                    try:
                        block = response.read(min(CHUNK, size - have))
                    except (OSError, http.client.HTTPException) as error:
                        raise TransferError(f"read failed: {type(error).__name__}: {error}") from error
                    if not block:
                        raise TransferError(f"connection closed at {have} of {size} bytes")
                    limiter.acquire(len(block))
                    out.write(block)
                    verifier.update(block)
                    have += len(block)
                    fetched += len(block)
                    on_progress(len(block), have)
                out.flush()
                os.fsync(out.fileno())
        finally:
            connection.close()
    else:  # complete from an earlier run: make sure its bytes are durable before the caller publishes it
        fd = os.open(temp_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    if not verifier.matches():
        os.unlink(temp_path)
        raise TransferError(f"checksum mismatch ({verifier.kind}): got {verifier.value()}, expected {verifier.expected}")
    return fetched
