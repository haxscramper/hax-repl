from __future__ import annotations

import re


MACRO_PATTERN = re.compile(r"\$\((.*?)\)")


class MacroExpander:
    """POC macro expander placeholder.

    For phases 0-2 we only parse macros and return their body as-is.
    """

    def expand(self, prompt: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            return match.group(1)

        return MACRO_PATTERN.sub(_replace, prompt)
