"""V2 regression checks for schema constants, reference escaping, and old packets."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pdca.core import canonical, decode, encode, optimize, train
from pdca.providers import build_prompt, parse_atoms


def sample():
    return [
        {
            "model_number": "ABCDEFGHIJK" * 6 + str(i),
            "product_name": "Phone device",
            "stock_status": "low_stock",
            "input": {"url": "https://example.com/catalog"},
            "sale_price": {"value": 990 + i, "currency": "JPY", "symbol": "¥"},
            "specifications": [
                {"name": "製品型番", "value": "ABCDEFGHIJK" * 6 + str(i) + " ※取扱説明書"},
                {"name": "ご注意", "value": "これは充電器です。"},
            ],
        }
        for i in range(10)
    ]


class V2Tests(unittest.TestCase):
    def test_constants_and_reference(self):
        dataset = sample()
        profile = train(dataset)
        self.assertEqual(profile["constants"]["$[]"]["input"]["url"], "https://example.com/catalog")
        self.assertEqual(profile["constants"]["$[].sale_price"]["currency"], "JPY")
        packet = encode(dataset, profile)
        self.assertIn("§!", canonical(packet))
        self.assertEqual(decode(packet), dataset)

    def test_constant_changes_and_missing(self):
        dataset = sample()
        p = train(dataset)
        future = copy.deepcopy(dataset)
        future[0]["sale_price"]["currency"] = "USD"
        future[1]["sale_price"].pop("symbol")
        future[2]["input"] = None
        future[3].pop("input")
        future[4]["specifications"][0]["value"] = "§!literal"
        future[5]["model_number"] = None
        future[6]["specifications"] = []
        future[7]["specifications"][0]["value"] = "random unrelated product"
        self.assertEqual(decode(encode(future, p)), future)

    def test_legacy_profile_and_packet(self):
        legacy = train(sample())
        legacy.pop("constants")
        legacy.pop("field_refs")
        encoded = encode(sample(), legacy)
        self.assertEqual(decode(encoded), sample())

    def test_profile_invariant_enforcement(self):
        p = train(sample())
        malformed = copy.deepcopy(p)
        malformed["constants"]["$[]"]["bogus"] = "oops"
        with self.assertRaises(ValueError):
            encode(sample(), malformed)
        packet = encode(sample(), p)
        packet["payload"][0][0] = "corrupted!"
        with self.assertRaises(ValueError):
            decode(packet)

    def test_cli_merge_profiles_offline(self):
        data = sample()
        p = train(data)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text(canonical(data), encoding="utf-8")
            (root / "first.json").write_text(canonical(encode(data, p)), encoding="utf-8")
            (root / "second.json").write_text(canonical(p), encoding="utf-8")
            result = subprocess.run([
                sys.executable, "-m", "pdca", "optimize", str(root / "data.json"),
                "-o", str(root / "out.json"),
                "--candidate-profiles", str(root / "first.json"), str(root / "second.json"),
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(decode(json.loads((root / "out.json").read_text())), data)


if __name__ == "__main__":
    unittest.main()