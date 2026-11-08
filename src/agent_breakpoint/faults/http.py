"""HTTP request/response values passed to model and dependency fault hooks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "content-length", "content-encoding", "host",
}


def clean_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


@dataclass
class HttpRequest:
    method: str
    path: str  # includes query string
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    def json(self) -> Any:
        try:
            return json.loads(self.body) if self.body else None
        except ValueError:
            return None

    def with_json(self, obj: Any) -> HttpRequest:
        return replace(self, body=json.dumps(obj).encode())

    @property
    def wants_stream(self) -> bool:
        data = self.json()
        if isinstance(data, dict) and data.get("stream") is True:
            return True
        return "stream" in self.path.lower()  # bedrock converse-stream / invoke-with-response-stream


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    @property
    def content_type(self) -> str:
        for k, v in self.headers.items():
            if k.lower() == "content-type":
                return v
        return ""

    @property
    def is_sse(self) -> bool:
        return "text/event-stream" in self.content_type

    @property
    def is_ndjson(self) -> bool:
        ct = self.content_type
        return "ndjson" in ct or "x-ndjson" in ct

    @property
    def is_eventstream(self) -> bool:
        return "vnd.amazon.eventstream" in self.content_type

    def json(self) -> Any:
        if self.is_sse or self.is_ndjson or self.is_eventstream:
            return None
        try:
            return json.loads(self.body) if self.body else None
        except ValueError:
            return None

    def with_json(self, obj: Any) -> HttpResponse:
        return replace(self, body=json.dumps(obj).encode())

    def with_body(self, body: bytes) -> HttpResponse:
        return replace(self, body=body)

    def to_record(self) -> dict[str, Any]:
        return {"status": self.status, "headers": clean_headers(self.headers),
                "body": self.body.decode("utf-8", "surrogateescape")}

    @classmethod
    def from_record(cls, data: dict[str, Any]) -> HttpResponse:
        return cls(status=data["status"], headers=dict(data.get("headers", {})),
                   body=data["body"].encode("utf-8", "surrogateescape"))
