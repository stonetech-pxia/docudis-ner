// Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.

import 'dart:convert';

import 'package:docudis_ner_ffi/docudis_ner_ffi.dart';
import 'package:test/test.dart';

void main() {
  test('UTF-8 offsets map to UTF-16 across ASCII, accents, CJK and emoji', () {
    const text = 'aé张😀b';
    final offsets = Utf8ToUtf16Offsets(text);
    // Every character boundary agrees with an encode of the prefix.
    for (var u16 = 0; u16 <= text.length; u16++) {
      if (u16 > 0 &&
          u16 < text.length &&
          text.codeUnitAt(u16) >= 0xDC00 &&
          text.codeUnitAt(u16) <= 0xDFFF) {
        continue; // inside the surrogate pair
      }
      final u8 = utf8.encode(text.substring(0, u16)).length;
      expect(offsets[u8], u16, reason: 'utf8 $u8');
    }
  });

  test('an offset inside a character or past the end is an error', () {
    final offsets = Utf8ToUtf16Offsets('é');
    expect(() => offsets[1], throwsRangeError);
    expect(() => offsets[3], throwsRangeError);
    expect(Utf8ToUtf16Offsets('')[0], 0);
  });
}
