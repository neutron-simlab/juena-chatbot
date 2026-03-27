"""Tests for the chunking abstraction and enrichment layer."""

from juena.indexing.chunking import IndexChunk, chunk_file, _detect_language, _path_tokens


PYTHON_SOURCE = '''\
def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"


class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
'''

MARKDOWN_SOURCE = """\
# My Project

Some intro text.

## Installation

Run `pip install myproject`.

## Usage

Import and call `greet`.
"""


def test_detect_language_python():
    assert _detect_language("src/main.py") == "python"


def test_detect_language_markdown():
    assert _detect_language("README.md") == "markdown"


def test_detect_language_fortran():
    assert _detect_language("src/solver.f90") == "fortran"


def test_detect_language_c():
    assert _detect_language("src/main.c") == "c"


def test_detect_language_unknown():
    assert _detect_language("data.json") == "other"


def test_path_tokens():
    tokens = _path_tokens("src/utils/helpers.py")
    assert "src" in tokens
    assert "utils" in tokens
    assert "helpers" in tokens


def test_chunk_file_python_returns_index_chunks():
    chunks = chunk_file(PYTHON_SOURCE, "src/main.py", chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 0
    assert all(isinstance(c, IndexChunk) for c in chunks)
    assert all(c.language == "python" for c in chunks)
    assert all(c.file_path == "src/main.py" for c in chunks)
    assert all(c.content_hash for c in chunks)


def test_chunk_file_python_enriches_symbols():
    chunks = chunk_file(PYTHON_SOURCE, "src/main.py", chunk_size=2000, chunk_overlap=0)
    symbols = [c.symbol for c in chunks if c.symbol]
    assert len(symbols) > 0


def test_chunk_file_markdown():
    chunks = chunk_file(MARKDOWN_SOURCE, "README.md", is_doc=True, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 0
    assert all(c.is_doc for c in chunks)
    assert all(c.language == "markdown" for c in chunks)


def test_chunk_file_fortran_falls_back_to_generic():
    fortran_source = """\
      SUBROUTINE SOLVE(X, Y, N)
      IMPLICIT NONE
      INTEGER N
      DOUBLE PRECISION X(N), Y(N)
      INTEGER I
      DO I = 1, N
        Y(I) = X(I) * 2.0D0
      END DO
      RETURN
      END
"""
    chunks = chunk_file(fortran_source, "src/solver.f90", chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 0
    assert all(c.language == "fortran" for c in chunks)


def test_chunk_file_c_falls_back_to_generic():
    c_source = """\
#include <stdio.h>

int main(int argc, char *argv[]) {
    printf("Hello, world!\\n");
    return 0;
}
"""
    chunks = chunk_file(c_source, "src/main.c", chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 0
    assert all(c.language == "c" for c in chunks)


def test_chunk_indices_are_sequential():
    chunks = chunk_file(PYTHON_SOURCE, "src/main.py", chunk_size=50, chunk_overlap=10)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_path_tokens_populated():
    chunks = chunk_file(PYTHON_SOURCE, "src/utils/main.py", chunk_size=2000, chunk_overlap=0)
    for c in chunks:
        assert "src" in c.path_tokens
        assert "utils" in c.path_tokens
        assert "main" in c.path_tokens
