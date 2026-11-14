"""PDCA: deterministic profile-driven, lossless JSON transform.

This is a JSON-value codec: object ordering and original whitespace are not preserved.
No AI-produced Python or dynamic code is ever executed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
from typing import Any

VERSION = 1
MARK = "§"
CODES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
RESERVED = {"default", "missing", "extra", "literal"}
SCHEMA_MAX = 128
ATOM_MAX = len(CODES)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def profile_hash(profile: dict) -> str:
    return hashlib.sha256(canonical(profile).encode("utf-8")).hexdigest()


def token_count(text: str, encoding: str = "o200k_base") -> int:
    """Fixed OpenAI reference encoding; not an exact GPT-6 token claim."""
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("Install 'pdca[tokenizer]' to benchmark tokens") from exc
    return len(tiktoken.get_encoding(encoding).encode(text))


def walk(value: Any, path: str = "$"):
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk(item, path + "." + key)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item, path + "[]")


def _build_schemas(data: Any) -> dict:
    objects: dict[str, list[dict]] = defaultdict(list)
    for path, val in walk(data):
        if isinstance(val, dict):
            objects[path].append(val)
    schemas = {}
    for path, entries in sorted(objects.items()):
        if len(entries) < 3:
            continue
        counts = Counter(k for obj in entries for k in obj)
        keys = sorted((k for k, count in counts.items() if count / len(entries) >= 0.8))
        if 1 < len(keys) <= SCHEMA_MAX:
            schemas[path] = keys
    return schemas


def _build_defaults(data: Any) -> dict:
    values: dict[str, list[Any]] = defaultdict(list)
    for path, val in walk(data):
        if path.startswith("$[]") and isinstance(val, (dict, str, int, float, bool, type(None))):
            values[path].append(val)
    defaults = {}
    for path, items in sorted(values.items()):
        if len(items) < 3:
            continue
        counts = Counter(canonical(v) for v in items)
        candidate, n = counts.most_common(1)[0]
        if n / len(items) >= 0.8 and len(candidate) >= 7:
            defaults[path] = json.loads(candidate)
    return defaults


def _candidate_atoms(data: Any) -> list[str]:
    strings = [v for _, v in walk(data) if isinstance(v, str)]
    counter = Counter()
    for s in strings:
        if len(s) >= 18:
            counter[s] += 1
        for sentence in re.findall(r"[^。\n]{18,150}。", s):
            counter[sentence] += 1
    options = []
    for text, c in counter.items():
        if c >= 2 and len(text) <= 500 and MARK not in text:
            occurrences = sum(s.count(text) for s in strings)
            if occurrences >= 2:
                options.append((occurrences * (len(text) - 3) - len(text) - 8, text))
    return [s for score, s in sorted(options, key=lambda v: (-v[0], v[1])) if score > 0][:ATOM_MAX]


def train(data: Any, *, atoms: bool = True) -> dict:
    """Train an explicit reusable profile from representative JSON values."""
    # No code or selectors are generated; a profile is declarative data only.
    profile = {"version": VERSION, "schemas": _build_schemas(data), "defaults": _build_defaults(data), "atoms": []}
    if atoms:
        profile["atoms"] = _candidate_atoms(data)
    validate_profile(profile)
    return profile


def validate_profile(profile: dict) -> None:
    if not isinstance(profile, dict) or set(profile) != {"version", "schemas", "defaults", "atoms"}:
        raise ValueError("Invalid PDCA profile keys")
    if profile["version"] != VERSION:
        raise ValueError("Unsupported PDCA profile version")
    if not isinstance(profile["schemas"], dict) or not isinstance(profile["defaults"], dict):
        raise ValueError("Invalid schemas/defaults")
    for path, keys in profile["schemas"].items():
        if not isinstance(path, str) or not path.startswith("$") or not isinstance(keys, list):
            raise ValueError("Invalid schema")
        if len(keys) > SCHEMA_MAX or len(set(keys)) != len(keys) or not all(isinstance(k, str) for k in keys):
            raise ValueError("Invalid schema fields")
    if not isinstance(profile["atoms"], list) or len(profile["atoms"]) > ATOM_MAX:
        raise ValueError("Invalid atoms")
    if len(set(profile["atoms"])) != len(profile["atoms"]) or not all(isinstance(s, str) and len(s) >= 1 and len(s) <= 500 and MARK not in s for s in profile["atoms"]):
        raise ValueError("Invalid atom values")
    if len(canonical(profile)) > 1_000_000:
        raise ValueError("Profile too large")


def _pack_string(s: str, atoms: list[str]) -> str:
    s = s.replace(MARK, MARK + MARK)
    for i, atom in enumerate(atoms):
        s = s.replace(atom, MARK + CODES[i])
    return s


def _unpack_string(s: str, atoms: list[str]) -> str:
    out = []
    i = 0
    while i < len(s):
        if s[i] != MARK:
            out.append(s[i]); i += 1
            continue
        if i + 1 >= len(s):
            raise ValueError("Truncated atom code")
        code = s[i + 1]
        if code == MARK:
            out.append(MARK)
        elif code in CODES[:len(atoms)]:
            out.append(atoms[CODES.index(code)])
        else:
            raise ValueError("Unknown atom code")
        i += 2
    return "".join(out)


def _marker(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("$pdca"), str) and value.get("$pdca") in RESERVED


def _encode(value: Any, path: str, p: dict) -> Any:
    if path in p["defaults"] and canonical(value) == canonical(p["defaults"][path]):
        return {"$pdca": "default"}
    if path in p["schemas"] and not isinstance(value, dict):
        return {"$pdca": "literal", "value": value}
    if _marker(value):
        # Escape marker-shaped user objects without recursively interpreting them.
        return {"$pdca": "literal", "value": value}
    if isinstance(value, str):
        return _pack_string(value, p["atoms"])
    if isinstance(value, list):
        return [_encode(v, path + "[]", p) for v in value]
    if isinstance(value, dict):
        schema = p["schemas"].get(path)
        if schema:
            row = [_encode(value[k], path + "." + k, p) if k in value else {"$pdca": "missing"} for k in schema]
            extras = {k: _encode(v, path + "." + k, p) for k, v in value.items() if k not in schema}
            if extras:
                row.append({"$pdca": "extra", "value": extras})
            return row
        return {k: _encode(v, path + "." + k, p) for k, v in value.items()}
    return value


def _decode(value: Any, path: str, p: dict) -> Any:
    if isinstance(value, dict) and value.get("$pdca") == "default":
        if path not in p["defaults"]:
            raise ValueError("Unexpected default marker")
        return p["defaults"][path]
    if isinstance(value, dict) and value.get("$pdca") == "literal":
        return value["value"]
    if isinstance(value, dict) and value.get("$pdca") == "missing":
        raise ValueError("Unexpected missing marker")
    if isinstance(value, dict) and value.get("$pdca") == "extra":
        raise ValueError("Unexpected extra marker")
    if isinstance(value, str):
        return _unpack_string(value, p["atoms"])
    if isinstance(value, list):
        keys = p["schemas"].get(path)
        if keys:
            if len(value) not in (len(keys), len(keys) + 1):
                raise ValueError("Invalid row length")
            result = {}
            for key, item in zip(keys, value):
                if item == {"$pdca": "missing"}:
                    continue
                result[key] = _decode(item, path + "." + key, p)
            if len(value) == len(keys) + 1:
                extra = value[-1]
                if not isinstance(extra, dict) or extra.get("$pdca") != "extra" or not isinstance(extra.get("value"), dict):
                    raise ValueError("Invalid extra fields")
                for key, item in extra["value"].items():
                    if key in result or key in keys:
                        raise ValueError("Conflicting extra field")
                    result[key] = _decode(item, path + "." + key, p)
            return result
        return [_decode(v, path + "[]", p) for v in value]
    if isinstance(value, dict):
        if path in p["schemas"]:
            raise ValueError("Expected encoded row at " + path)
        return {k: _decode(v, path + "." + k, p) for k, v in value.items()}
    return value


def encode(data: Any, profile: dict, *, embed_profile: bool = True) -> dict:
    validate_profile(profile)
    packet = {"pdca": VERSION, "profile_sha256": profile_hash(profile), "data_sha256": hashlib.sha256(canonical(data).encode("utf-8")).hexdigest(), "payload": _encode(data, "$", profile)}
    if embed_profile:
        packet["profile"] = profile
    return packet


def decode(packet: dict, profile: dict | None = None) -> Any:
    if not isinstance(packet, dict) or packet.get("pdca") != VERSION:
        raise ValueError("Invalid PDCA packet")
    chosen = profile if profile is not None else packet.get("profile")
    if chosen is None:
        raise ValueError("External profile is required")
    validate_profile(chosen)
    if profile_hash(chosen) != packet.get("profile_sha256"):
        raise ValueError("Profile hash mismatch")
    if "payload" not in packet:
        raise ValueError("Missing payload")
    decoded = _decode(packet["payload"], "$", chosen)
    if hashlib.sha256(canonical(decoded).encode("utf-8")).hexdigest() != packet.get("data_sha256"):
        raise ValueError("Decoded data checksum mismatch")
    return decoded


def assert_roundtrip(data: Any, profile: dict) -> None:
    if canonical(decode(encode(data, profile))) != canonical(data):
        raise ValueError("Lossless round-trip check failed")


def optimize(data: Any, *, candidate_atoms: list[str] | None = None, encoding: str = "o200k_base") -> tuple[dict, dict]:
    """Measure entire embedded packet; accept only a lossless, smaller candidate."""
    base = train(data, atoms=False)
    try:
        token_count("", encoding)
        score = lambda obj: token_count(canonical(obj), encoding)
        metric = "tokens"
    except RuntimeError:
        score = lambda obj: len(canonical(obj).encode("utf-8"))
        metric = "bytes"
    best = base
    best_size = score(encode(data, base))
    auto = _candidate_atoms(data)
    candidates = auto + (candidate_atoms or [])
    for atom in candidates:
        if atom in best["atoms"] or len(best["atoms"]) >= ATOM_MAX or not isinstance(atom, str) or not (4 <= len(atom) <= 500) or MARK in atom:
            continue
        proposal = {**best, "atoms": [*best["atoms"], atom]}
        if sum(s.count(atom) for _, s in walk(data) if isinstance(s, str)) < 2:
            continue
        try:
            assert_roundtrip(data, proposal)
            result = score(encode(data, proposal))
        except (ValueError, TypeError):
            continue
        if result < best_size:
            best, best_size = proposal, result
    assert_roundtrip(data, best)
    original_size = score(data)
    return best, {"metric": metric, "baseline": original_size, "encoded": best_size, "saved": original_size - best_size, "rate": 1 - best_size / original_size if original_size else 0, "encoding": encoding if metric == "tokens" else None}