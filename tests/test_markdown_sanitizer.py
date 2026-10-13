"""Tests for the markdown sanitizer on the agent editor's create path."""

from pathlib import Path

import pytest

pytest.importorskip("pydantic_ai")

from codewiki.src.be.agent_tools.str_replace_editor import (
    EditTool,
    sanitize_llm_markdown,
)


def test_closed_think_block_is_stripped():
    text = "<think>reasoning about the module</think>\n# Title\n\nBody.\n"
    assert sanitize_llm_markdown(text) == "# Title\n\nBody.\n"


def test_stray_closing_think_keeps_the_tail():
    text = (
        "<OVERVIEW_READ>\n"
        "I'll read the documentation files before writing the overview."
        "</think>I'll read the documentation files for the top-level modules first.\n\n"
        "# Overview\n\nThe repository is a spec-first API.\n"
    )
    out = sanitize_llm_markdown(text)
    assert out.startswith("# Overview")
    assert "think" not in out.lower()
    assert "I'll read" not in out


def test_invented_leading_tool_tags_are_stripped():
    text = "<OVERVIEW_READ>\n\n# Overview\n\nBody.\n"
    out = sanitize_llm_markdown(text)
    assert out == "# Overview\n\nBody.\n"


def test_narration_before_first_heading_is_dropped():
    text = "Let me look at the module pages first.\n\n# Title\n\nBody.\n"
    assert sanitize_llm_markdown(text) == "# Title\n\nBody.\n"


def test_plain_page_is_untouched():
    text = "# Title\n\nBody with <em>inline html</em> and normal prose.\n"
    assert sanitize_llm_markdown(text) == text


def test_content_without_heading_is_untouched():
    # No markdown heading anywhere: nothing to anchor the narration strip,
    # so the page passes through unchanged.
    text = "A page that never uses headings.\n\nPlain body.\n"
    assert sanitize_llm_markdown(text) == text


def test_front_matter_style_preamble_is_kept():
    # A preamble that is not narration chatter (no opener words, no tags)
    # must survive.
    text = "Some intentional intro paragraph.\n\n# Title\n\nBody.\n"
    assert sanitize_llm_markdown(text) == text


def _editor(tmp_path: Path) -> EditTool:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(exist_ok=True)
    return EditTool({"file_history": "{}"}, absolute_docs_path=str(docs_dir))


def test_narration_detector_matches_chatter_not_documents():
    from codewiki.src.be.documentation_generator import _looks_like_narration

    # The observed failure mode: narration instead of the overview.
    assert _looks_like_narration(
        "I'll read the module documentation files referenced by the "
        "`docs_path` fields before writing the overview...."
    )
    assert _looks_like_narration("Let me look at the child pages first.")
    # Real content, however small, is not narration.
    assert not _looks_like_narration("overview")
    assert not _looks_like_narration("[store](store.md)")
    assert not _looks_like_narration("# Overview\n\nThe repository provides...")
    # Narration-shaped openers in a long structured document are fine.
    assert not _looks_like_narration(
        "I will describe the system.\n\n- point one\n- point two\n\n```go\nx()\n```"
    )


def test_create_sanitizes_markdown_but_not_other_files(tmp_path: Path):
    editor = _editor(tmp_path)
    editor.create_file(
        tmp_path / "docs" / "overview.md",
        "I'll write the overview now.\n\n# Overview\n\nBody.\n",
    )
    editor.create_file(tmp_path / "docs" / "data.json", '{"a": "<think>x</think>"}')
    assert (tmp_path / "docs" / "overview.md").read_text() == "# Overview\n\nBody.\n"
    assert (
        tmp_path / "docs" / "data.json"
    ).read_text() == '{"a": "<think>x</think>"}'
