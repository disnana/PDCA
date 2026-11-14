"""Regression tests for atom substitution touching previously emitted codes."""
import json
import random
import unittest

from pdca.core import _pack_string, _unpack_string, decode, encode, train


class AtomSubstitutionTests(unittest.TestCase):
    def test_generated_atom_code_is_not_replaced(self):
        # The first atom produces §A. A later atom must not replace its A.
        atoms = ["hello", "A"]
        self.assertEqual(_pack_string("hello", atoms), "§A")
        self.assertEqual(_unpack_string(_pack_string("hello", atoms), atoms), "hello")

    def test_escaped_marker_and_generated_reference(self):
        atoms = ["repeat", "A"]
        original = "§repeat A §§repeat"
        self.assertEqual(_unpack_string(_pack_string(original, atoms), atoms), original)

    def test_overlapping_atoms_and_unicode(self):
        atoms = ["東京都", "東京", "都", "京", "A", "aaa"]
        for s in ["東京都東京", "東京東京都", "aaaAaaaa", "🦊§東京都A"]:
            self.assertEqual(_unpack_string(_pack_string(s, atoms), atoms), s)

    def test_full_roundtrip_with_single_character_atom(self):
        data = [{"text": "hello"}, {"text": "§hello A"}, {"text": "Ahello"}]
        profile = train(data, atoms=False)
        profile["atoms"] = ["hello", "A"]
        self.assertEqual(decode(encode(data, profile)), data)

    def test_existing_packet_decode_unchanged(self):
        # A pre-fix wire payload can be decoded unchanged (decoder not modified).
        data = [{"text": "hello A"}]
        profile = train(data, atoms=False)
        profile["atoms"] = ["hello", "A"]
        packet = encode(data, profile)
        packet["payload"][0]["text"] = "§A §B"
        self.assertEqual(decode(json.loads(json.dumps(packet, ensure_ascii=False))), data)

    def test_generated_codes_are_never_reexamined(self):
        # Every code in the 62-character alphabet must remain immutable.
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        atoms = ["segment" + f"{i:03d}" for i in range(len(alphabet))]
        for i, atom in enumerate(atoms):
            self.assertEqual(_pack_string(atom, atoms), "§" + alphabet[i])
        self.assertEqual(_unpack_string(_pack_string("".join(atoms), atoms), atoms), "".join(atoms))

    def test_random_unicode_and_atom_overlaps(self):
        rng = random.Random(20261114)
        fragments = ["§", "A", "B", "東京", "🦊", "aa", "abc", "hello", "\n", "0"]
        atoms = ["hello", "A", "abc", "aa", "東京", "B", "0"]
        for _ in range(500):
            value = "".join(rng.choice(fragments) for _ in range(rng.randint(0, 20)))
            self.assertEqual(_unpack_string(_pack_string(value, atoms), atoms), value)


if __name__ == "__main__":
    unittest.main()
