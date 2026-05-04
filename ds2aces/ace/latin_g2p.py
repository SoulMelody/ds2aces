"""
ACE Studio G2P (Grapheme-to-Phoneme) ONNX inference reimplementation.

Supports: English (en), French (fr), Italian (it), Portuguese (pt), Spanish (es).

Architecture:
  - Each language has a separate g2p_xx.onnx model under MLG2PService.
  - Input: char_ids (int64 tensor, shape [1, seq_len]) — character indices.
  - Output: phoneme id sequence (int64 tensor) — decoded via phoneme vocabulary.
  - Characters are mapped through a grapheme vocabulary (P="padding", E="end", then a-z).
  - The word is lowercased, appended with "E" (end marker), each char looked up in the
    grapheme vocab to get an integer id, then fed as a 1D tensor named "char_ids".
  - Output ids are mapped back through the phoneme vocabulary.
  - <PAD> and <EOS> tokens in the output are stripped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Vocabulary definitions (extracted from HyperParameters constructors)
# ---------------------------------------------------------------------------

# Grapheme vocab is shared across all languages:
#   index 0 -> "P" (padding), index 1 -> "E" (end), index 2 -> "U" (unknown),
#   index 3..28 -> 'a'..'z'.
# Note: in the binary, the vocab init passes L"Uabcdefghijklmnopqrstuvwxyz"
# (a wide string) cast to const char*. QString::QString(const char*) reads it
# as UTF-8/Latin-1 and stops at the first NUL byte after 'U' (since UTF-16LE
# encoding of 'a' starts with 0x00), leaving just "U" — the unknown-grapheme
# placeholder used as the QHash default value (id 2).
GRAPHEME_VOCAB: list[str] = ["P", "E", "U"] + list("abcdefghijklmnopqrstuvwxyz")
GRAPHEME_TO_ID: dict[str, int] = {ch: i for i, ch in enumerate(GRAPHEME_VOCAB)}

# Phoneme vocabularies per language (index 0 = <PAD>, index 1 = <EOS>, rest are phonemes)
PHONEME_VOCABS: dict[str, list[str]] = {
    "en": [
        "<PAD>", "<EOS>",
        "aa", "ae", "ah", "ao", "aw", "ay",
        "b", "ch", "d", "dh", "eh", "er", "ey",
        "f", "g", "hh", "ih", "iy", "jh",
        "k", "l", "m", "n", "ng", "ow", "oy",
        "p", "r", "s", "sh", "t", "th", "uh", "uw",
        "v", "w", "y", "z", "zh",
        "mv", "nv", "ngv", "dx", "dr", "tr",
    ],
    "fr": [
        "<PAD>", "<EOS>",
        "i", "y", "u", "e", "two", "o", "ee", "nin", "oo",
        "a", "aa", "eh", "onp", "eenp", "ninnp", "anp",
        "p", "b", "t", "d", "k", "g", "qm", "tss", "dzz",
        "m", "n", "jj", "nn",
        "f", "v", "s", "z", "ss", "zz", "rr",
        "j", "l", "hh", "w",
    ],
    "it": [
        "<PAD>", "<EOS>",
        "i", "u", "e", "o", "ee", "oo", "a",
        "p", "b", "t", "d", "k", "g", "ts", "dz", "tss", "dzz",
        "m", "n", "jj", "nn", "r", "for",
        "f", "v", "s", "z", "ss",
        "j", "w", "l", "ll",
    ],
    "pt": [
        "<PAD>", "<EOS>",
        "i", "u", "ii", "e", "o", "ee", "oo", "six", "a", "two",
        "inp", "unp", "enp", "onp", "sixnp", "eenp", "anp",
        "uj", "ej", "oj", "eej", "ooj", "aj",
        "iw", "ew", "eew", "oow", "aw",
        "ujnp", "ejnp", "ojnp", "sixjnp", "ownp", "awnp", "sixwnp",
        "p", "b", "t", "d", "k", "g", "tss", "dzz",
        "m", "n", "jj", "for",
        "f", "v", "s", "z", "ss", "zz", "x",
        "j", "w", "l", "ll",
    ],
    "es": [
        "<PAD>", "<EOS>",
        "a", "e", "i", "o", "u", "y",
        "aa", "ee", "ii", "iisl", "mm", "oo", "qq", "uu", "uusl", "vv", "yy",
        "eh", "ehsl", "ae", "ea",
        "one", "two", "thr", "thrsl", "six", "svn", "eit", "nin",
        "nd", "aj", "ej", "oj", "uj", "aw", "ew", "ow",
        "ja", "je", "jo", "ju", "wa", "we", "wi", "wo",
        "mv", "nv", "nnv",
        "b", "blz", "c", "d", "ddt", "dlz", "f", "g", "glz", "h", "hsl",
        "j", "jsl", "k", "l", "ldt", "lsl", "m", "n", "ndt",
        "p", "psl", "q", "r", "rdt", "rsl", "rst",
        "s", "sdt", "ssl", "t", "tdt", "v", "w", "x", "xsl", "z", "zdt", "zsl",
        "bb", "bbsl", "cc", "dd", "ff", "gg", "ggsl", "ggsz",
        "hh", "hhsl", "jj", "jjsl", "jjsz", "kk", "kksl", "ll", "llsl", "mmsl",
        "nn", "nnsl", "oosl", "pp", "rr", "rrsl", "ss", "tt", "ww", "xx", "xxsl", "zz",
        "for", "fve", "qm", "qmsl", "ls", "zl", "nz", "nzl", "xcsl",
        "ilsl", "iilsl", "eqsl",
        "ts", "dz", "tss", "dzz", "tssl", "dzsl", "tkk", "kp", "gb", "nnm",
    ],
}


# ---------------------------------------------------------------------------
# G2P engine
# ---------------------------------------------------------------------------

PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
PAD_GRAPHEME = "P"
END_GRAPHEME = "E"  # appended to every input word before encoding (sub_14080ABD0)

# HyperParameters::MAX_WORD_LENGTH — the ONNX models always receive a
# fixed-size [1, 20] char_ids tensor; positions past the actual encoded
# sequence are zero-filled (id 0 = "P", the padding grapheme).
MAX_WORD_LENGTH = 20


@dataclass
class MLG2PProvider:
    """One language-specific G2P provider, mirroring MLG2P*Provider in ACE Studio.

    Mirrors the inference path in sub_14080ABD0:
      1. word.toLower(); seq = word.mid(0, len-1) + "E"
      2. for each char -> grapheme_vocab.indexOf(char) -> int64 char_ids[i]
         (unknown chars fall back to id 2 in the binary's QHash; we use 2 for "a")
      3. ONNX session.run({"char_ids": char_ids[None, :]}) -> output ids
      4. map ids back via phoneme_vocab; stop at <EOS>; drop <PAD>.
    """

    language: str
    model_path: Path
    grapheme_vocab: list[str] = field(default_factory=lambda: list(GRAPHEME_VOCAB))
    phoneme_vocab: list[str] = field(default_factory=list)
    _session: Optional["ort.InferenceSession"] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.phoneme_vocab:
            self.phoneme_vocab = list(PHONEME_VOCABS[self.language])

    @property
    def session(self) -> "ort.InferenceSession":
        if self._session is None:
            if ort is None:
                raise RuntimeError("onnxruntime is not installed")
            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        return self._session

    def _encode(self, word: str) -> np.ndarray:
        word = word.lower()
        # Mirrors QString::mid(0, MAX_WORD_LENGTH - 1) + append("E"):
        # take up to (MAX-1) chars from the word, then append the end marker.
        seq = word[: MAX_WORD_LENGTH - 1] + END_GRAPHEME
        ids: list[int] = []
        # Decompiled fallback id for missing keys is 2 (the value past the EOF
        # marker in the QHash default), which corresponds to 'a' here.
        unk_id = 2
        for ch in seq:
            ids.append(GRAPHEME_TO_ID.get(ch, unk_id))
        return np.asarray(ids, dtype=np.int64)

    def _decode(self, output_ids: np.ndarray) -> list[str]:
        phonemes: list[str] = []
        for idx in output_ids.flatten().tolist():
            if idx < 0 or idx >= len(self.phoneme_vocab):
                continue
            tok = self.phoneme_vocab[idx]
            if tok == EOS_TOKEN:
                break
            if tok == PAD_TOKEN:
                continue
            phonemes.append(tok)
        return phonemes

    def predict(self, word: str) -> list[str]:
        if not word:
            return []
        ids = self._encode(word)
        # Right-pad with 0 (the "P" grapheme id) to MAX_WORD_LENGTH.
        char_ids = np.zeros(MAX_WORD_LENGTH, dtype=np.int64)
        char_ids[: ids.shape[0]] = ids
        sess = self.session
        input_name = sess.get_inputs()[0].name  # expected: "char_ids"
        outputs = sess.run(None, {input_name: char_ids})
        out_ids = np.asarray(outputs[0], dtype=np.int64)
        return self._decode(out_ids)


class MLG2PService:
    """Top-level service that lazily loads each language model on first use.

    Mirrors MLG2PService in ACE Studio: providers are built on demand from
    g2p_<lang>.onnx in the model directory.
    """

    LANGUAGES = ("en", "fr", "it", "pt", "es")

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self._providers: dict[str, MLG2PProvider] = {}

    def get_provider(self, language: str) -> MLG2PProvider:
        if language not in self.LANGUAGES:
            raise ValueError(f"Unsupported language: {language!r}")
        prov = self._providers.get(language)
        if prov is None:
            model_path = self.model_dir / f"g2p_{language}.onnx"
            if not model_path.is_file():
                raise FileNotFoundError(model_path)
            prov = MLG2PProvider(language=language, model_path=model_path)
            self._providers[language] = prov
        return prov

    def predict(self, language: str, word: str) -> list[str]:
        return self.get_provider(language).predict(word)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ACE Studio G2P inference")
    parser.add_argument(
        "--model-dir", required=True,
        help="Directory containing g2p_{en,fr,it,pt,es}.onnx",
    )
    parser.add_argument("--lang", required=True, choices=MLG2PService.LANGUAGES)
    parser.add_argument("words", nargs="+")
    args = parser.parse_args()

    service = MLG2PService(args.model_dir)
    for w in args.words:
        print(f"{w} -> {' '.join(service.predict(args.lang, w))}")
