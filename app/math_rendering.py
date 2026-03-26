"""Helpers for detecting and normalizing equation markup in chat content."""

from __future__ import annotations

import re
from typing import Any

INLINE_MATH_ALIAS_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
BLOCK_MATH_ALIAS_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
INLINE_DOLLAR_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$", re.DOTALL)
BLOCK_DOLLAR_MATH_RE = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.DOTALL)
LATEX_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")
UNESCAPED_DOLLAR_RE = re.compile(r"(?<!\\)\$")
LIST_PREFIX_RE = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+)(.*)$")
MATH_STRUCTURE_RE = re.compile(r"[{}_^=]")
PLAIN_WORD_RE = re.compile(r"(?<!\\)\b[A-Za-z]{2,}\b")
TERMINAL_PUNCTUATION_RE = re.compile(r"^(.*?)([.,;:])?$", re.DOTALL)
STRONG_LATEX_COMMANDS = {
    "alpha",
    "approx",
    "beta",
    "cdot",
    "chi",
    "cos",
    "delta",
    "Delta",
    "displaystyle",
    "dfrac",
    "epsilon",
    "eta",
    "exp",
    "frac",
    "gamma",
    "Gamma",
    "geq",
    "hbar",
    "hslash",
    "infty",
    "int",
    "iota",
    "kappa",
    "lambda",
    "Lambda",
    "left",
    "leq",
    "lim",
    "log",
    "mu",
    "nabla",
    "neq",
    "nu",
    "Omega",
    "omega",
    "oint",
    "partial",
    "phi",
    "Phi",
    "pi",
    "Pi",
    "prod",
    "psi",
    "Psi",
    "rho",
    "right",
    "sigma",
    "Sigma",
    "sin",
    "sqrt",
    "sum",
    "tan",
    "tau",
    "theta",
    "Theta",
    "tfrac",
    "varphi",
    "vartheta",
    "xi",
    "Xi",
    "zeta",
}


def split_markdown_code_segments(content: str) -> list[tuple[str, bool]]:
    """Split markdown into plain-text and code segments."""

    segments: list[tuple[str, bool]] = []
    index = 0
    plain_start = 0

    while index < len(content):
        if content.startswith("```", index):
            closing = content.find("```", index + 3)
            if closing == -1:
                index += 3
                continue

            if plain_start < index:
                segments.append((content[plain_start:index], False))

            closing += 3
            segments.append((content[index:closing], True))
            index = closing
            plain_start = index
            continue

        if content[index] == "`":
            run_length = 1
            while index + run_length < len(content) and content[index + run_length] == "`":
                run_length += 1

            # Triple-backtick fences are handled above; longer runs are left as plain text.
            if run_length >= 3:
                index += run_length
                continue

            delimiter = "`" * run_length
            closing = content.find(delimiter, index + run_length)
            if closing == -1:
                index += run_length
                continue

            if plain_start < index:
                segments.append((content[plain_start:index], False))

            closing += run_length
            segments.append((content[index:closing], True))
            index = closing
            plain_start = index
            continue

        index += 1

    if plain_start < len(content):
        segments.append((content[plain_start:], False))

    return segments


def normalize_math_aliases(segment: str) -> str:
    """Normalize LaTeX alias delimiters to Streamlit-supported markdown math."""

    normalized = INLINE_MATH_ALIAS_RE.sub(lambda match: f"${match.group(1)}$", segment)

    def _block_replacement(match: re.Match[str]) -> str:
        body = match.group(1).strip("\n")
        prefix = "" if match.start() == 0 or normalized[match.start() - 1] == "\n" else "\n"
        suffix = "" if match.end() == len(normalized) or normalized[match.end()] == "\n" else "\n"
        return f"{prefix}$$\n{body}\n$${suffix}"

    return BLOCK_MATH_ALIAS_RE.sub(_block_replacement, normalized)


def _extract_list_prefix(line: str) -> tuple[str, str]:
    """Return markdown list prefix and remaining line body."""

    match = LIST_PREFIX_RE.match(line)
    if match is None:
        return "", line
    return match.group(1), match.group(2)


def _contains_strong_latex_syntax(text: str) -> bool:
    """Return True when text contains strong signals of LaTeX math syntax."""

    commands = {match.group(1) for match in LATEX_COMMAND_RE.finditer(text)}
    if commands.intersection(STRONG_LATEX_COMMANDS):
        return True
    return False


def _find_first_math_command_start(text: str) -> int | None:
    """Return the index of the first strong LaTeX command in the text."""

    for match in LATEX_COMMAND_RE.finditer(text):
        if match.group(1) in STRONG_LATEX_COMMANDS:
            return match.start()
    return None


def _looks_like_equation_fragment(text: str, math_start: int | None = None) -> bool:
    """Return True when text contains an equation-like LaTeX fragment."""

    candidate_start = _find_first_math_command_start(text) if math_start is None else math_start
    if candidate_start is None:
        return False

    candidate = text[candidate_start:].strip()
    if not candidate.startswith("\\"):
        return False

    strong_commands = [
        match.group(1)
        for match in LATEX_COMMAND_RE.finditer(candidate)
        if match.group(1) in STRONG_LATEX_COMMANDS
    ]
    if not strong_commands:
        return False
    if len(strong_commands) >= 2:
        return True
    return MATH_STRUCTURE_RE.search(candidate) is not None


def repair_latex_delimiters_in_line(line: str) -> str:
    """Repair missing inline-math delimiters for equation-like lines."""

    prefix, body = _extract_list_prefix(line)
    if not body.strip():
        return line

    dollar_matches = list(UNESCAPED_DOLLAR_RE.finditer(body))
    math_start = _find_first_math_command_start(body)

    if len(dollar_matches) == 1 and math_start is not None and _looks_like_equation_fragment(body, math_start):
        dollar_index = dollar_matches[0].start()
        if dollar_index <= math_start:
            return prefix + body + "$"
        return prefix + body[:math_start] + "$" + body[math_start:]

    if not dollar_matches and math_start == 0 and _looks_like_equation_fragment(body, math_start):
        return prefix + f"${body}$"

    return line


def repair_latex_delimiters(content: str) -> str:
    """Repair broken or missing inline-math delimiters on equation-like lines."""

    repaired_lines: list[str] = []
    for line in content.splitlines(keepends=True):
        line_body = line.rstrip("\r\n")
        newline = line[len(line_body):]
        repaired_lines.append(repair_latex_delimiters_in_line(line_body) + newline)

    if not repaired_lines and content:
        return repair_latex_delimiters_in_line(content)
    return "".join(repaired_lines)


def _looks_like_adjacent_math_fragment(text: str) -> bool:
    """Return True when plain text next to a math block looks like equation continuation."""

    stripped = text.strip()
    if not stripped:
        return False
    if PLAIN_WORD_RE.search(stripped):
        return False
    if _contains_strong_latex_syntax(stripped):
        return True
    return False


def _wrap_math_fragment(fragment: str) -> str:
    """Wrap a plain-text fragment in inline-math delimiters, preserving outer spacing."""

    leading_len = len(fragment) - len(fragment.lstrip())
    trailing_len = len(fragment) - len(fragment.rstrip())
    leading = fragment[:leading_len]
    trailing = fragment[len(fragment) - trailing_len:] if trailing_len else ""
    core = fragment[leading_len: len(fragment) - trailing_len if trailing_len else len(fragment)]

    punct_match = TERMINAL_PUNCTUATION_RE.match(core)
    if punct_match is None:
        return fragment

    body = punct_match.group(1)
    punctuation = punct_match.group(2) or ""
    if not body:
        return fragment
    return f"{leading}${body}${punctuation}{trailing}"


def repair_adjacent_math_fragments_in_line(line: str) -> str:
    """Wrap bare equation fragments that appear next to valid inline math blocks."""

    matches = list(INLINE_DOLLAR_MATH_RE.finditer(line))
    if not matches:
        return line

    rebuilt: list[str] = []
    cursor = 0
    for match in matches:
        fragment = line[cursor:match.start()]
        if _looks_like_adjacent_math_fragment(fragment):
            rebuilt.append(_wrap_math_fragment(fragment))
        else:
            rebuilt.append(fragment)
        rebuilt.append(match.group(0))
        cursor = match.end()

    tail = line[cursor:]
    if _looks_like_adjacent_math_fragment(tail):
        rebuilt.append(_wrap_math_fragment(tail))
    else:
        rebuilt.append(tail)

    return "".join(rebuilt)


def repair_adjacent_math_fragments(content: str) -> str:
    """Repair bare equation fragments adjacent to valid inline math blocks."""

    repaired_lines: list[str] = []
    for line in content.splitlines(keepends=True):
        line_body = line.rstrip("\r\n")
        newline = line[len(line_body):]
        repaired_lines.append(repair_adjacent_math_fragments_in_line(line_body) + newline)

    if not repaired_lines and content:
        return repair_adjacent_math_fragments_in_line(content)
    return "".join(repaired_lines)


def normalize_math_markdown(content: str) -> str:
    """Normalize supported math aliases while preserving markdown code segments."""

    normalized_segments = [
        (
            segment
            if is_code
            else repair_adjacent_math_fragments(
                repair_latex_delimiters(normalize_math_aliases(segment))
            )
        )
        for segment, is_code in split_markdown_code_segments(content)
    ]
    return "".join(normalized_segments)


def has_supported_math_markup(segment: str) -> bool:
    """Return whether a plain-text markdown segment contains supported math syntax."""

    patterns = (
        INLINE_MATH_ALIAS_RE,
        BLOCK_MATH_ALIAS_RE,
        INLINE_DOLLAR_MATH_RE,
        BLOCK_DOLLAR_MATH_RE,
    )
    return any(match.group(1).strip() for pattern in patterns for match in pattern.finditer(segment))


def content_contains_math_markup(content: Any) -> bool:
    """Return True when string content includes supported math markup outside code."""

    if not isinstance(content, str) or not content:
        return False

    return any(
        not is_code
        and (
            has_supported_math_markup(segment)
            or repair_latex_delimiters(normalize_math_aliases(segment)) != segment
        )
        for segment, is_code in split_markdown_code_segments(content)
    )
