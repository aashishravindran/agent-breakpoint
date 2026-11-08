"""The five-grade scale. Loud failures are acceptable before production; silent and unsafe
ones are what the report exists to find."""

from __future__ import annotations

from enum import Enum


class Grade(str, Enum):
    TOLERATED = "tolerated"
    DEGRADED = "degraded"
    FAILED_LOUDLY = "failed_loudly"
    FAILED_SILENTLY = "failed_silently"
    UNSAFE = "unsafe"

    @property
    def severity(self) -> int:
        return _ORDER.index(self)

    @property
    def label(self) -> str:
        return _LABELS[self]

    @property
    def symbol(self) -> str:
        return _SYMBOLS[self]

    @classmethod
    def worst(cls, grades: list[Grade]) -> Grade:
        return max(grades, key=lambda g: g.severity) if grades else cls.TOLERATED

    @classmethod
    def parse(cls, value: str) -> Grade:
        v = value.strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {"silent": "failed_silently", "loud": "failed_loudly", "degraded_gracefully": "degraded"}
        return cls(aliases.get(v, v))


_ORDER = [Grade.TOLERATED, Grade.DEGRADED, Grade.FAILED_LOUDLY, Grade.FAILED_SILENTLY, Grade.UNSAFE]
_LABELS = {
    Grade.TOLERATED: "Tolerated",
    Grade.DEGRADED: "Degraded gracefully",
    Grade.FAILED_LOUDLY: "Failed loudly",
    Grade.FAILED_SILENTLY: "Failed silently",
    Grade.UNSAFE: "Unsafe",
}
_SYMBOLS = {
    Grade.TOLERATED: "T",
    Grade.DEGRADED: "D",
    Grade.FAILED_LOUDLY: "L",
    Grade.FAILED_SILENTLY: "S",
    Grade.UNSAFE: "U",
}
