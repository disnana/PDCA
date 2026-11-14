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
DEFAULT_CODE = MARK + "="
MISSING_CODE = MARK + "?"
REF_CODE = MARK + "!"
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


def _build_constants(data: Any, schemas: dict) -> dict:
    objects: dict[str, list[dict]] = defaultdict(list)
    for path, val in walk(data):
        if path in schemas and isinstance(val, dict):
            objects[path].append(val)
    result = {}
    for path, entries in sorted(objects.items()):
        if len(entries) < 3:
            continue
        constants = {}
        for field in schemas[path]:
            if all(field in record for record in entries):
                base = canonical(entries[0][field])
                if all(canonical(record[field]) == base for record in entries[1:]):
                    constants[field] = entries[0][field]
        if constants:
            result[path] = constants
    return result


def _build_defaults(data: Any) -> dict:
    values: dict[str, list[Any]] = defaultdict(list)
    for path, val in walk(data):
        if path.startswith("$[]") and path != "$[]" and isinstance(val, (list, dict, str, int, float, bool, type(None))):
            values[path].append(val)
    defaults = {}
    for path, items in sorted(values.items()):
        if len(items) < 3:
            continue
        counts = Counter(canonical(v) for v in items)
        candidate, n = counts.most_common(1)[0]
        if n / len(items) >= 0.8 and (len(candidate) >= 7 or candidate in ("[]", "{}", "null")):
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
    # Common URL prefixes and repeated column values are often much more
    # efficient than full-string dictionary entries.
    prefixes = Counter()
    for string in strings:
        if string.startswith("http"):
            m = re.match(r"https?://[^/]+/[^/]+/", string)
            if m:
                prefixes[m.group()] += 1
    for value, n in prefixes.items():
        if n >= 2 and MARK not in value:
            options.append((n * (len(value) - 3) - len(value) - 8, value))
    return [s for score, s in sorted(options, key=lambda v: (-v[0], v[1])) if score > 0][:ATOM_MAX]


def train(data: Any, *, atoms: bool = True) -> dict:
    """Train an explicit reusable profile from representative JSON values."""
    # No code or selectors are generated; a profile is declarative data only.
    schemas = _build_schemas(data)
    constants = _build_constants(data, schemas)
    defaults = _build_defaults(data)
    for parent, fields in constants.items():
        for field in fields:
            defaults.pop(parent + "." + field, None)
            for key in list(defaults):
                if key.startswith(parent + "." + field + "."):
                    del defaults[key]
    profile = {"version": VERSION, "schemas": schemas, "defaults": defaults, "atoms": [], "field_refs": {"$[].specifications[].value": "model_number"} if "$[]" in schemas and "model_number" in schemas["$[]"] and "$[].specifications[]" in schemas else {}, "constants": constants}
    if atoms:
        profile["atoms"] = _candidate_atoms(data)
    validate_profile(profile)
    return profile


def validate_profile(profile: dict) -> None:
    if not isinstance(profile, dict) or not {"version", "schemas", "defaults", "atoms"}.issubset(profile) or set(profile) - {"version", "schemas", "defaults", "atoms", "field_refs", "constants"}:
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
    if not isinstance(profile.get("field_refs", {}), dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not k.startswith("$[]")
        or "." in v for k, v in profile.get("field_refs", {}).items()
    ):
        raise ValueError("Invalid field references")
    if not isinstance(profile.get("constants", {}), dict):
        raise ValueError("Invalid constants")
    for path, values in profile.get("constants", {}).items():
        if path not in profile["schemas"] or not isinstance(values, dict) or not set(values).issubset(profile["schemas"][path]):
            raise ValueError("Invalid constant schema")
    if len(canonical(profile)) > 1_000_000:
        raise ValueError("Profile too large")


def _pack_string(s: str, atoms: list[str]) -> str:
    """Encode original characters once; never reprocess emitted atom codes.

    At each position, the first matching atom in profile order wins.  This
    keeps replacement deterministic even when atoms overlap.  A literal
    marker is escaped as two markers, also without being scanned again.
    """
    out = []
    offset = 0
    while offset < len(s):
        if s.startswith(MARK, offset):
            out.append(MARK + MARK)
            offset += len(MARK)
            continue
        for index, atom in enumerate(atoms):
            if s.startswith(atom, offset):
                out.append(MARK + CODES[index])
                offset += len(atom)
                break
        else:
            out.append(s[offset])
            offset += 1
    return "".join(out)


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


def _encode(value: Any, path: str, p: dict, record: dict | None = None) -> Any:
    if path in p["defaults"] and canonical(value) == canonical(p["defaults"][path]):
        return DEFAULT_CODE
    if path in p["schemas"] and not isinstance(value, dict):
        return {"$pdca": "literal", "value": value}
    source_key = p.get("field_refs", {}).get(path)
    if source_key and isinstance(value, str) and isinstance(record, dict) and isinstance(record.get(source_key), str):
        source = record[source_key]
        if source and value.startswith(source):
            return REF_CODE + _pack_string(value[len(source):], p["atoms"])
    if _marker(value):
        # Escape marker-shaped user objects without recursively interpreting them.
        return {"$pdca": "literal", "value": value}
    if isinstance(value, str):
        return _pack_string(value, p["atoms"])
    if isinstance(value, list):
        return [_encode(v, path + "[]", p, v if path == "$" and isinstance(v, dict) else record) for v in value]
    if isinstance(value, dict):
        schema = p["schemas"].get(path)
        if schema:
            const = p.get("constants", {}).get(path, {})
            active = [k for k in schema if k not in const]
            row = [_encode(value[k], path + "." + k, p, record) if k in value else MISSING_CODE for k in active]
            extras = {k: _encode(v, path + "." + k, p, record) for k, v in value.items() if k not in schema or (k in const and canonical(v) != canonical(const[k]))}
            for k in const:
                if k not in value:
                    extras[k] = MISSING_CODE
            if extras:
                row.append({"$pdca": "extra", "value": extras})
            return row
        return {k: _encode(v, path + "." + k, p, record) for k, v in value.items()}
    return value


def _decode(value: Any, path: str, p: dict, record: dict | None = None) -> Any:
    if value == DEFAULT_CODE or (isinstance(value, dict) and value.get("$pdca") == "default"):
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
        if value.startswith(REF_CODE) and path in p.get("field_refs", {}):
            key = p["field_refs"][path]
            if not isinstance(record, dict) or not isinstance(record.get(key), str):
                raise ValueError("Invalid reference to " + key)
            return record[key] + _unpack_string(value[len(REF_CODE):], p["atoms"])
        return _unpack_string(value, p["atoms"])
    if isinstance(value, list):
        keys = p["schemas"].get(path)
        if keys:
            const = p.get("constants", {}).get(path, {})
            active = [k for k in keys if k not in const]
            if len(value) not in (len(active), len(active) + 1):
                raise ValueError("Invalid row length")
            result = dict(const)
            for key, item in zip(active, value):
                if item == MISSING_CODE or item == {"$pdca": "missing"}:
                    continue
                result[key] = _decode(item, path + "." + key, p, result if path == "$[]" else record)
            if len(value) == len(active) + 1:
                extra = value[-1]
                if not isinstance(extra, dict) or extra.get("$pdca") != "extra" or not isinstance(extra.get("value"), dict):
                    raise ValueError("Invalid extra fields")
                for key, item in extra["value"].items():
                    if key in active or (key in result and key not in const):
                        raise ValueError("Conflicting extra field")
                    if key in const and (item == MISSING_CODE or item == {"$pdca": "missing"}):
                        result.pop(key, None)
                    else:
                        result[key] = _decode(item, path + "." + key, p, result if path == "$[]" else record)
            return result
        return [_decode(v, path + "[]", p, record) for v in value]
    if isinstance(value, dict):
        if path in p["schemas"]:
            raise ValueError("Expected encoded row at " + path)
        return {k: _decode(v, path + "." + k, p, record) for k, v in value.items()}
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