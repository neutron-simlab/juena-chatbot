"""Tests for the chunking abstraction and enrichment layer."""

from juena.indexing import chunking as chunking_module
from juena.indexing.chunking import IndexChunk, chunk_file, _detect_language


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


def test_detect_language_rst():
    assert _detect_language("docs/index.rst") == "rst"


def test_detect_language_fortran():
    assert _detect_language("src/solver.f90") == "fortran"


def test_detect_language_c():
    assert _detect_language("src/main.c") == "c"


def test_detect_language_unknown():
    assert _detect_language("data.json") == "other"



def test_chunk_file_python_returns_index_chunks():
    chunks = chunk_file(PYTHON_SOURCE, "src/main.py", chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 0
    assert all(isinstance(c, IndexChunk) for c in chunks)
    assert all(c.language == "python" for c in chunks)
    assert all(c.file_path == "src/main.py" for c in chunks)
    assert all(c.content_hash for c in chunks)



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


def test_parser_chunks_are_hard_capped(monkeypatch):
    oversized_text = "\n".join(f"line_{i} = {i}" for i in range(100))

    def fake_chonkie_chunk(source: str, language: str, chunk_size: int):
        return [{
            "text": oversized_text,
            "start_index": 0,
            "end_index": len(oversized_text),
            "token_count": len(oversized_text),
        }]

    monkeypatch.setattr(chunking_module, "_chonkie_chunk", fake_chonkie_chunk)

    chunks = chunk_file(oversized_text, "src/main.py", chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))




def test_start_index_is_exact_for_every_chunk():
    """The old implementation recovered offsets with source.find(), which is
    wrong whenever the splitter strips whitespace -- which it does by default."""
    source = "\n\n".join(f"def fn_{i}():\n    return {i}\n" for i in range(40))

    chunks = chunk_file(source, "src/many.py", chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert source[chunk.start_index:chunk.end_index] == chunk.text, (
            f"chunk {chunk.chunk_index} offsets do not locate its own text"
        )


def test_start_index_is_exact_for_non_chonkie_languages():
    source = "\n\n".join(
        f"void fn_{i}(void) {{\n    int x = {i};\n}}\n" for i in range(40)
    )

    chunks = chunk_file(source, "src/many.c", chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert source[chunk.start_index:chunk.end_index] == chunk.text


def test_duplicate_text_chunks_keep_distinct_exact_offsets():
    repeated = "same repeated paragraph with enough content to split"
    source = "\n\n".join([repeated] * 20)

    chunks = chunk_file(source, "notes.txt", chunk_size=80, chunk_overlap=0)

    assert len(chunks) > 1
    assert len({chunk.start_index for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        assert source[chunk.start_index:chunk.end_index] == chunk.text


def test_c_uses_language_aware_separators():
    """A C file must not be split with the generic prose separators."""
    from juena.indexing.chunking import _make_splitter

    c_splitter = _make_splitter("c", 100, 0)
    generic_splitter = _make_splitter("fortran", 100, 0)

    assert c_splitter._separators != generic_splitter._separators


def test_fortran_has_no_language_aware_splitter():
    """Fortran has no Language member; it must fall back, not crash."""
    chunks = chunk_file("      PROGRAM MAIN\n      END\n", "solver.f90")
    assert all(c.language == "fortran" for c in chunks)
