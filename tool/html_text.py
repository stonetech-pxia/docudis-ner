# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""HTML to plain text for registry announcements.

Copied from docudis-android tool/fetch_public_samples.py, which keeps its own copy for the test sets.
"""
import html
import re


def strip_tags(x):
    x = re.sub(r'<(script|style)[\s\S]*?</\1>', ' ', x)
    x = re.sub(r'<br\s*/?>|</p>|</div>|</tr>|</dd>|</dt>|</h\d>|</li>', '\n', x, flags=re.I)
    x = html.unescape(re.sub(r'<[^>]+>', ' ', x))
    x = re.sub(r'[ \t]+', ' ', x)
    x = re.sub(r' *\n *', '\n', x)
    x = re.sub(r' ([,.;:)])', r'\1', x)
    return re.sub(r'\n{2,}', '\n', x).strip()
