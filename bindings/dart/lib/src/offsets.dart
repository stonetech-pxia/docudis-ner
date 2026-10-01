// Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.

/// Maps half-open UTF-8 byte offsets of a string to UTF-16 code-unit offsets,
/// for every detection of one text after a single pass over it.
class Utf8ToUtf16Offsets {
  Utf8ToUtf16Offsets(String text) : _utf16At = _table(text);

  /// `_utf16At[b]` is the UTF-16 offset of UTF-8 byte `b`, or -1 when `b`
  /// falls inside a character.
  final List<int> _utf16At;

  static List<int> _table(String text) {
    final out = <int>[];
    var utf16 = 0;
    for (final rune in text.runes) {
      final bytes = rune < 0x80
          ? 1
          : rune < 0x800
          ? 2
          : rune < 0x10000
          ? 3
          : 4;
      out.add(utf16);
      for (var i = 1; i < bytes; i++) {
        out.add(-1);
      }
      utf16 += rune < 0x10000 ? 1 : 2;
    }
    out.add(utf16);
    return out;
  }

  int operator [](int utf8Offset) {
    if (utf8Offset < 0 || utf8Offset >= _utf16At.length) {
      throw RangeError.range(utf8Offset, 0, _utf16At.length - 1, 'offset');
    }
    final value = _utf16At[utf8Offset];
    if (value < 0) {
      throw RangeError('UTF-8 offset $utf8Offset is not a character boundary');
    }
    return value;
  }
}
