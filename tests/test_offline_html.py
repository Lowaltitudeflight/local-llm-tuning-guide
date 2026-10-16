#!/usr/bin/env python3
"""The offline promise of tools/config-generator.html is enforced, not assumed.

The README and tools/README.md both claim the page runs with no network and
no build step. This test fails the build the moment someone adds an external
resource. Plain anchor links (<a href="https://...">) are navigation, not
resource loading, and are allowed.

Run with:  python -m unittest discover tests
"""

import os
import re
import unittest

HTML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tools", "config-generator.html"
)

# Each pattern catches one way of pulling a resource over the network.
RESOURCE_PATTERNS = [
    (r"<script[^>]+src\s*=\s*[\"'](?!data:)", "external <script src>"),
    (r"<link[^>]+href\s*=\s*[\"'](?!data:|#)", "external <link href>"),
    (r"<img[^>]+src\s*=\s*[\"'](?!data:)", "external <img src>"),
    (r"<iframe[^>]+src\s*=\s*[\"'](?!data:)", "external <iframe src>"),
    (r"<source[^>]+src\s*=\s*[\"'](?!data:)", "external <source src>"),
    (r"@import\s+url\(", "CSS @import"),
    (r"\bfetch\s*\(\s*[\"']https?:", "fetch() to an absolute URL"),
    (r"XMLHttpRequest", "XMLHttpRequest"),
    (r"import\s*\(\s*[\"']https?:", "dynamic import of an absolute URL"),
]


class TestOfflineHTML(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            os.path.exists(HTML_PATH),
            "tools/config-generator.html is missing",
        )
        with open(HTML_PATH, encoding="utf-8") as fh:
            self.html = fh.read()

    def test_no_external_resources(self):
        offenders = [
            label
            for pattern, label in RESOURCE_PATTERNS
            if re.search(pattern, self.html, re.IGNORECASE)
        ]
        self.assertFalse(
            offenders,
            "config-generator.html must stay fully offline; found: "
            + ", ".join(offenders),
        )

    def test_no_inline_event_handlers_reaching_out(self):
        # src="http..." inside a string is already covered above; this guards
        # against attributes like onclick="location='http://...'" being used
        # to load something. Navigation is fine, loading is not.
        self.assertIsNone(
            re.search(r"\.src\s*=\s*[\"']https?:", self.html, re.IGNORECASE),
            "assigning an absolute URL to .src loads an external resource",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
