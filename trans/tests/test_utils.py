"""Unit tests for utils.py."""
import os
import tempfile
import unicodedata
import unittest

from trans import utils


class UtilsTests(unittest.TestCase):

    def test_open_normalize(self):
        self.assertRaises(ValueError, utils.OpenNormalize,
                          "foo.txt", True, "wb")

    def test_open_read_normalize(self):
        text = "우체국\n"
        self.assertEqual(unicodedata.normalize("NFC", text), text)
        with tempfile.TemporaryDirectory() as tempdir:
            tmp_tsv = os.path.join(tempdir, "tmp.tsv")
            with open(tmp_tsv, "w") as w:
                w.write(text)
            with utils.OpenNormalize(tmp_tsv, normalize=True, mode="r") as f:
                normalized_text = list(f)[0]
            self.assertEqual(8, len(normalized_text))
            self.assertEqual(unicodedata.normalize("NFD", text),
                             normalized_text)

    def test_open_read(self):
        text = "우체국\n"
        self.assertEqual(unicodedata.normalize("NFC", text), text)
        with tempfile.TemporaryDirectory() as tempdir:
            tmp_tsv = os.path.join(tempdir, "tmp.tsv")
            with open(tmp_tsv, "w") as w:
                w.write(text)
            with utils.OpenNormalize(
                    tmp_tsv, normalize=False, mode="r") as f:
                unnormalized_text = list(f)[0]
            self.assertEqual(4, len(unnormalized_text))
            self.assertEqual(text, unnormalized_text)

    def test_open_write_normalize(self):
        text = "우체국\n"
        normalized_text = unicodedata.normalize("NFD", text)
        self.assertNotEqual(text, normalized_text)
        with tempfile.TemporaryDirectory() as tempdir:
            tmp_tsv = os.path.join(tempdir, "tmp.tsv")
            with utils.OpenNormalize(tmp_tsv, normalize=True, mode="w") as w:
                w.write(normalized_text)
            with open(tmp_tsv, "r") as f:
                written_text = f.read()
            self.assertEqual(4, len(written_text))
            self.assertEqual(text, written_text)

    def test_open_write(self):
        text = "우체국\n"
        normalized_text = unicodedata.normalize("NFD", text)
        self.assertNotEqual(text, normalized_text)
        with tempfile.TemporaryDirectory() as tempdir:
            tmp_tsv = os.path.join(tempdir, "tmp.tsv")
            with utils.OpenNormalize(
                    tmp_tsv, normalize=False, mode="w") as w:
                w.write(normalized_text)
            with open(tmp_tsv, "r") as f:
                written_text = f.read()
            self.assertEqual(8, len(written_text))
            self.assertEqual(normalized_text, written_text)

    def test_tokenizer_character_mode(self):
        tokenizer = utils.Tokenizer.from_cli("none")

        self.assertEqual(["a", "b"], tokenizer.tokenize("ab"))
        self.assertEqual("ab", tokenizer.untokenize(["a", "b"]))

    def test_tokenizer_separator_mode(self):
        tokenizer = utils.Tokenizer(" ")

        self.assertEqual(["a", "d͡ʒ", "e"], tokenizer.tokenize("a d͡ʒ e"))
        self.assertEqual("a d͡ʒ e", tokenizer.untokenize(["a", "d͡ʒ", "e"]))

    def test_tokenizer_rejects_empty_separator_tokens(self):
        tokenizer = utils.Tokenizer(" ")

        with self.assertRaisesRegex(ValueError, "Empty token"):
            tokenizer.tokenize("a  b")


if __name__ == "__main__":
    UtilsTests().run()
