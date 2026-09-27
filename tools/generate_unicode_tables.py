#!/usr/bin/env python3
"""Generate the Unicode tables the Linux engine's tokenizer embeds.

The Qwen2 pre-tokenizer splits on \\p{L}, \\p{N} and \\s, and its normalizer is
NFC. This writes those properties, from this Python's unicodedata, to a small
little-endian u32 blob that qwen_image_cuda/src/unicode.cplus embeds with
#include_bytes. Development tool only.

Layout (every value a u32):
  magic "QIUT", unicode version as (major << 16 | minor << 8 | micro)
  letters:      count, then count (first, last) ranges
  numbers:      count, then count (first, last) ranges
  whitespace:   count, then count (first, last) ranges
  combining:    count, then count (first, last, class) ranges, class > 0
  decompositions: count, then count (codepoint, pool offset, length) sorted by
                  codepoint (full canonical decomposition, Hangul excluded),
                  then pool size and the pool
  compositions: count, then count (first, second, composite) sorted by
                (first, second): primary composites only
"""

from __future__ import annotations

import argparse
import struct
import sys
import unicodedata
from pathlib import Path

MAX = 0x110000


def ranges(predicate) -> list[tuple[int, int]]:
    result = []
    start = None
    for cp in range(MAX):
        if predicate(cp):
            if start is None:
                start = cp
        elif start is not None:
            result.append((start, cp - 1))
            start = None
    if start is not None:
        result.append((start, MAX - 1))
    return result


def is_hangul_syllable(cp: int) -> bool:
    return 0xAC00 <= cp <= 0xD7A3


def full_decomposition(cp: int) -> list[int]:
    mapping = unicodedata.decomposition(chr(cp))
    if not mapping or mapping.startswith("<"):
        return [cp]
    result = []
    for part in mapping.split():
        result.extend(full_decomposition(int(part, 16)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("qwen_image_cuda/data/unicode.bin"))
    arguments = parser.parse_args()

    letters = ranges(lambda cp: unicodedata.category(chr(cp)).startswith("L"))
    numbers = ranges(lambda cp: unicodedata.category(chr(cp)).startswith("N"))
    # Unicode White_Space, the \s of the Rust and Oniguruma regex engines.
    white = {0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
             0x2028, 0x2029, 0x202F, 0x205F, 0x3000}
    whitespace = ranges(lambda cp: cp in white)

    combining = []
    for cp in range(MAX):
        c = unicodedata.combining(chr(cp))
        if c:
            if combining and combining[-1][1] == cp - 1 and combining[-1][2] == c:
                combining[-1] = (combining[-1][0], cp, c)
            else:
                combining.append((cp, cp, c))

    decompositions = []
    pool: list[int] = []
    for cp in range(MAX):
        if is_hangul_syllable(cp):
            continue
        full = full_decomposition(cp)
        if full != [cp]:
            decompositions.append((cp, len(pool), len(full)))
            pool.extend(full)

    compositions = []
    for cp in range(MAX):
        if is_hangul_syllable(cp):
            continue
        mapping = unicodedata.decomposition(chr(cp))
        if not mapping or mapping.startswith("<"):
            continue
        parts = [int(p, 16) for p in mapping.split()]
        if len(parts) != 2:
            continue
        if unicodedata.normalize("NFC", chr(parts[0]) + chr(parts[1])) == chr(cp):
            compositions.append((parts[0], parts[1], cp))
    compositions.sort()

    major, minor, micro = (int(x) for x in unicodedata.unidata_version.split("."))
    values = [int.from_bytes(b"QIUT", "little"), (major << 16) | (minor << 8) | micro]
    for table in (letters, numbers, whitespace):
        values.append(len(table))
        for first, last in table:
            values += [first, last]
    values.append(len(combining))
    for first, last, c in combining:
        values += [first, last, c]
    values.append(len(decompositions))
    for cp, offset, length in decompositions:
        values += [cp, offset, length]
    values.append(len(pool))
    values += pool
    values.append(len(compositions))
    for first, second, composite in compositions:
        values += [first, second, composite]

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(struct.pack(f"<{len(values)}I", *values))
    print(f"unicode {unicodedata.unidata_version}: letters {len(letters)}, numbers {len(numbers)}, "
          f"whitespace {len(whitespace)}, combining {len(combining)}, decompositions {len(decompositions)} "
          f"(pool {len(pool)}), compositions {len(compositions)}; {len(values) * 4} bytes", file=sys.stderr)


if __name__ == "__main__":
    main()
