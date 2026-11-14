import json
import os
import random
import tempfile
import unittest
from pathlib import Path

from pdca import train, encode, decode, optimize, assert_roundtrip
from pdca.core import canonical, profile_hash
from pdca.providers import parse_atoms, propose_atoms, build_prompt


DATA = [
    {"product_name": "Anker Nano Charger", "stock": "low_stock",
     "price": {"value": 3990, "currency": "JPY"},
     "specifications": [{"name": "size", "value": "約77 × 37 × 25mm"}],
     "features": [], "input": {"url": "https://example.invalid/catalog"},
     "text": "製品の仕様は予告なく変更されることがあります。"}
    for _ in range(12)
]
DATA[6]["stock"] = "out_of_stock"
DATA[6]["extra"] = {"source": True, "color": None}


class CodecTests(unittest.TestCase):
    def test_realistic(self):
        p = train(DATA)
        self.assertEqual(canonical(decode(encode(DATA, p))), canonical(DATA))
        self.assertTrue(p["schemas"])
        self.assertTrue(p["defaults"])

    def test_unknown_keys_and_types(self):
        p = train(DATA)
        future = [dict(x) for x in DATA]
        future[0]["unknown_new"] = {"$pdca": "default", "deep": ["§A", None, {}, []]}
        future[1]["product_name"] = None
        future[2]["specifications"] = None
        future[3]["specifications"] = [{"name": "X", "new_key": "y"}]
        future[4]["price"] = False
        future[5]["stock"] = "in_stock"
        future[7] = {"$pdca": "literal", "value": "$"}
        assert_roundtrip(future, p)

    def test_escape_and_atoms(self):
        data = [{"text": "hello §ABCDE § world §A long shared phrase."},
                {"text": "hello §ABCDE § world §A long shared phrase."},
                {"text": "hello §ABCDE § world §A long shared phrase."}]
        p = train(data)
        assert_roundtrip(data, p)

    def test_external_profile_and_hash_mismatch(self):
        p = train(DATA)
        q = encode(DATA, p, embed_profile=False)
        self.assertEqual(canonical(decode(q, p)), canonical(DATA))
        with self.assertRaises(ValueError):
            decode(q)
        wrong = train([{"different": 1}, {"different": 2}, {"different": 3}])
        with self.assertRaises(ValueError):
            decode(q, wrong)

    def test_reject_corrupted_payload(self):
        p = train(DATA)
        q = encode(DATA, p)
        q["payload"] = {"not": "expected"}
        with self.assertRaises(ValueError):
            decode(q)

    def test_missing_null_float_bool(self):
        data = [None, False, True, 0, 1, 1.0, 0.0, "", "§§", [], {}, {"x": None}, {"x": 0}]
        assert_roundtrip(data, train(data, atoms=False))
        self.assertNotEqual(canonical(1), canonical(1.0))

    def test_roundtrip_random(self):
        rng = random.Random(137)
        atoms = ["", "aaa", "§", "🦊", "保証がございます。", "","a\nb"]
        def item(depth=0):
            if depth >= 3:
                return rng.choice(atoms + [None, False, True, 0, 42, 1.5])
            kind = rng.randint(0, 3)
            if kind == 0:
                return {rng.choice(["x", "y", "§", "$pdca", "z"]): item(depth+1) for _ in range(rng.randint(0, 3))}
            if kind == 1:
                return [item(depth+1) for _ in range(rng.randint(0, 3))]
            return rng.choice(atoms + [None, False, True, 0, 42, 1.5])
        train_data = [item() for _ in range(70)]
        profile = train(train_data)
        assert_roundtrip(train_data, profile)
        new_data = [item() for _ in range(70)]
        assert_roundtrip(new_data, profile)

    def test_optimize(self):
        data = DATA * 2
        p, metrics = optimize(data)
        self.assertIn(metrics["metric"], ("bytes", "tokens"))
        self.assertIsInstance(metrics["saved"], int)
        assert_roundtrip(data, p)

    def test_untrusted_ai_suggestions(self):
        class FakeResult:
            output_text = json.dumps({"atoms": ["never shows up", "製品の仕様は予告なく変更されることがあります。", "💡" * 999]})
        class FakeResponses:
            def create(self, **kwargs):
                return FakeResult()
        class FakeClient:
            responses = FakeResponses()
        result = propose_atoms(DATA, "openai", "gpt-6", client=FakeClient())
        self.assertEqual(result, ["製品の仕様は予告なく変更されることがあります。"])
        with self.assertRaises(ValueError):
            parse_atoms('not json')
        with self.assertRaises(ValueError):
            propose_atoms(DATA, "unknown", "test")

    def test_malformed_profile(self):
        p = train(DATA)
        p["atoms"] = ["§invalid"]
        with self.assertRaises(ValueError):
            encode(DATA, p)


if __name__ == "__main__":
    unittest.main()