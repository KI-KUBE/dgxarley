"""Minimal community.general.dict_kv so the netpol tasks render under the fake collection."""

from typing import Any


def dict_kv(value: Any, key: str) -> dict[str, Any]:
    return {key: value}


class FilterModule:
    def filters(self) -> dict[str, Any]:
        return {"dict_kv": dict_kv}
