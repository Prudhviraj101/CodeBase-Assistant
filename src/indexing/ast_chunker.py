"""
AST-aware code chunker using Tree-sitter.

Splits source files at function/class boundaries so each chunk is a
complete logical unit. Carries three categories of metadata per chunk:
  - Identity:  file_path, start_line, end_line, language
  - Semantics: chunk_type, name, imports, calls
  - Content:   docstring, signature, source
"""

from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken
from tree_sitter import Language, Parser, Node

from src.config import LANGUAGE_MAP, MAX_CHUNK_TOKENS, MIN_CHUNK_CHARS

logger = logging.getLogger(__name__)

# Cache parsed languages and tokenizer
_language_cache: dict[str, Language] = {}
_tokenizer = tiktoken.get_encoding("cl100k_base")


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class CodeChunk:
    """A single semantically-coherent code chunk with metadata."""

    # Identity
    file_path: str
    start_line: int
    end_line: int
    language: str

    # Semantics
    chunk_type: str  # "function", "class", "method", "module", "block"
    name: str  # identifier (function/class name) or ""
    imports: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    # Content
    source: str = ""
    docstring: str = ""
    signature: str = ""

    @property
    def token_count(self) -> int:
        return len(_tokenizer.encode(self.source))

    def to_embedding_text(self) -> str:
        """Build the text that will be embedded — includes context."""
        parts = [f"# {self.chunk_type}: {self.name}" if self.name else ""]
        if self.signature:
            parts.append(self.signature)
        if self.docstring:
            parts.append(f'"""{self.docstring}"""')
        parts.append(self.source)
        return "\n".join(p for p in parts if p)

    def to_metadata(self) -> dict:
        """Return a flat dict suitable for ChromaDB metadata storage."""
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "imports": ",".join(self.imports),
            "calls": ",".join(self.calls),
            "docstring": self.docstring[:500],  # cap for metadata size
            "signature": self.signature,
        }


# ── Language loading ─────────────────────────────────────────────────────


def _get_language(ext: str) -> Language | None:
    """Load and cache a Tree-sitter language for the given file extension."""
    if ext not in LANGUAGE_MAP:
        return None
    if ext in _language_cache:
        return _language_cache[ext]

    cfg = LANGUAGE_MAP[ext]
    try:
        mod = importlib.import_module(cfg["module"])
        if "sub" in cfg:
            # TypeScript has .language_typescript() and .language_tsx()
            lang_fn = getattr(mod, f"language_{cfg['sub']}", None)
            if lang_fn is None:
                lang_fn = getattr(mod, "language")
            lang = Language(lang_fn())
        else:
            lang = Language(mod.language())
        _language_cache[ext] = lang
        return lang
    except Exception as exc:
        logger.warning("Failed to load tree-sitter language for %s: %s", ext, exc)
        return None


# ── Node inspection helpers ──────────────────────────────────────────────


def _node_text(node: Node, source_bytes: bytes) -> str:
    """Extract the source text for a node."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_name(node: Node, source_bytes: bytes) -> str:
    """Try to extract the identifier name from a node."""
    # Look for a direct 'name' or 'identifier' child
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier", "property_identifier"):
            return _node_text(child, source_bytes)
    # For decorated definitions, look one level deeper
    for child in node.children:
        if child.type in ("function_definition", "class_definition",
                          "function_declaration", "class_declaration",
                          "method_declaration", "function_item"):
            return _node_name(child, source_bytes)
    return ""


def _node_signature(node: Node, source_bytes: bytes) -> str:
    """Extract just the first line (signature) of a function/class."""
    text = _node_text(node, source_bytes)
    first_line = text.split("\n")[0].strip()
    return first_line


def _extract_calls(node: Node, source_bytes: bytes) -> list[str]:
    """Recursively find all function call names within a node."""
    calls: set[str] = set()

    def _walk(n: Node):
        if n.type == "call" or n.type == "call_expression":
            # Get the function being called
            if n.children:
                func_node = n.children[0]
                call_name = _node_text(func_node, source_bytes).split("(")[0].strip()
                # Only keep simple names, not long chained expressions
                if len(call_name) < 80 and "\n" not in call_name:
                    calls.add(call_name)
        for child in n.children:
            _walk(child)

    _walk(node)
    return sorted(calls)


def _extract_imports(source: str, language: str) -> list[str]:
    """Extract import statements from source code using regex."""
    imports: list[str] = []
    if language in ("python",):
        for m in re.finditer(r"(?:from\s+([\w.]+)\s+)?import\s+([\w., ]+)", source):
            imports.append(m.group(0).strip())
    elif language in ("javascript", "typescript", "tsx", "jsx"):
        for m in re.finditer(r"import\s+.+?from\s+['\"](.+?)['\"]", source):
            imports.append(m.group(0).strip())
    elif language in ("java",):
        for m in re.finditer(r"import\s+([\w.]+);", source):
            imports.append(m.group(1))
    elif language in ("go",):
        for m in re.finditer(r'"([\w./]+)"', source):
            imports.append(m.group(1))
    elif language in ("rust",):
        for m in re.finditer(r"use\s+([\w:]+)", source):
            imports.append(m.group(1))
    return imports


def _extract_docstring(node: Node, source_bytes: bytes) -> str:
    """Try to extract a docstring from the first child of a node."""
    for child in node.children:
        if child.type in ("expression_statement", "comment", "block"):
            inner = child
            # Python: expression_statement → string
            if child.type == "expression_statement" and child.children:
                inner = child.children[0]
            if inner.type in ("string", "string_literal", "template_string", "comment"):
                text = _node_text(inner, source_bytes).strip("\"'` \n")
                return text[:500]
        # Also check for block → expression_statement pattern (class body)
        if child.type == "block" and child.children:
            first = child.children[0]
            if first.type == "expression_statement" and first.children:
                inner = first.children[0]
                if inner.type == "string":
                    return _node_text(inner, source_bytes).strip("\"'` \n")[:500]
    return ""


# ── Extension → language name mapping ────────────────────────────────────

_EXT_TO_LANG_NAME: dict[str, str] = {
    ".py": "python",
    ".js": "javascript", ".jsx": "jsx",
    ".ts": "typescript", ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp",
}


# ── Main chunking ────────────────────────────────────────────────────────


def chunk_file(file_path: Path, source_code: str | None = None) -> list[CodeChunk]:
    """
    Parse a source file using Tree-sitter and split it into semantic chunks.

    Each chunk corresponds to a top-level function, class, or (for large
    classes) individual methods. Tiny fragments are merged.

    Falls back to line-based splitting if the file cannot be parsed.
    """
    ext = file_path.suffix
    lang = _get_language(ext)
    lang_name = _EXT_TO_LANG_NAME.get(ext, ext.lstrip("."))

    if source_code is None:
        source_code = file_path.read_text(encoding="utf-8", errors="replace")

    source_bytes = source_code.encode("utf-8")
    rel_path = str(file_path)

    # Fallback: if no tree-sitter grammar, do line-based splitting
    if lang is None:
        return _line_based_chunks(rel_path, source_code, lang_name)

    parser = Parser(lang)
    tree = parser.parse(source_bytes)

    top_node_types = set(LANGUAGE_MAP.get(ext, {}).get("top_nodes", []))
    file_imports = _extract_imports(source_code, lang_name)

    chunks: list[CodeChunk] = []
    covered_ranges: list[tuple[int, int]] = []

    def _process_node(node: Node, parent_name: str = "") -> None:
        """Recursively process an AST node into chunks."""
        if node.type in top_node_types:
            text = _node_text(node, source_bytes)
            token_count = len(_tokenizer.encode(text))
            name = _node_name(node, source_bytes)
            chunk_type = _classify_node(node.type)

            if token_count <= MAX_CHUNK_TOKENS:
                # Fits in one chunk — emit it whole
                chunk = CodeChunk(
                    file_path=rel_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    language=lang_name,
                    chunk_type=chunk_type,
                    name=name,
                    imports=file_imports,
                    calls=_extract_calls(node, source_bytes),
                    source=text,
                    docstring=_extract_docstring(node, source_bytes),
                    signature=_node_signature(node, source_bytes),
                )
                chunks.append(chunk)
                covered_ranges.append((node.start_byte, node.end_byte))
            else:
                # Too big — recursively split children (e.g. class → methods)
                has_child_chunks = False
                for child in node.children:
                    if child.type in top_node_types or child.type in (
                        "method_definition", "function_definition",
                        "function_declaration", "method_declaration",
                        "function_item",
                    ):
                        child_name = _node_name(child, source_bytes)
                        full_name = f"{name}.{child_name}" if name and child_name else (child_name or name)
                        child_text = _node_text(child, source_bytes)
                        child_tokens = len(_tokenizer.encode(child_text))

                        if child_tokens <= MAX_CHUNK_TOKENS:
                            chunk = CodeChunk(
                                file_path=rel_path,
                                start_line=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                                language=lang_name,
                                chunk_type=_classify_node(child.type),
                                name=full_name,
                                imports=file_imports,
                                calls=_extract_calls(child, source_bytes),
                                source=child_text,
                                docstring=_extract_docstring(child, source_bytes),
                                signature=_node_signature(child, source_bytes),
                            )
                            chunks.append(chunk)
                            covered_ranges.append((child.start_byte, child.end_byte))
                            has_child_chunks = True
                        else:
                            # Still too big — line-based fallback for this node
                            sub_chunks = _line_based_chunks(
                                rel_path, child_text, lang_name,
                                start_line_offset=child.start_point[0],
                            )
                            chunks.extend(sub_chunks)
                            covered_ranges.append((child.start_byte, child.end_byte))
                            has_child_chunks = True

                if not has_child_chunks:
                    # No recognized children — line-based fallback
                    sub_chunks = _line_based_chunks(
                        rel_path, text, lang_name,
                        start_line_offset=node.start_point[0],
                    )
                    chunks.extend(sub_chunks)
                    covered_ranges.append((node.start_byte, node.end_byte))
        else:
            # Not a top-level node — recurse into children
            for child in node.children:
                _process_node(child, parent_name)

    # Process all top-level children
    for child in tree.root_node.children:
        _process_node(child)

    # Gather any uncovered module-level code (imports, constants, etc.)
    _add_uncovered_regions(
        chunks, covered_ranges, source_bytes, rel_path,
        lang_name, file_imports, tree.root_node.end_byte,
    )

    # Merge tiny chunks
    chunks = _merge_tiny_chunks(chunks)

    return chunks


def _classify_node(node_type: str) -> str:
    """Map a tree-sitter node type to a human-readable chunk_type."""
    if "class" in node_type or "struct" in node_type or "impl" in node_type:
        return "class"
    if "method" in node_type:
        return "method"
    if "function" in node_type:
        return "function"
    if "interface" in node_type or "trait" in node_type:
        return "interface"
    if "enum" in node_type:
        return "enum"
    if "namespace" in node_type or "module" in node_type:
        return "module"
    if "type" in node_type:
        return "type"
    return "block"


def _add_uncovered_regions(
    chunks: list[CodeChunk],
    covered: list[tuple[int, int]],
    source_bytes: bytes,
    file_path: str,
    language: str,
    imports: list[str],
    total_bytes: int,
) -> None:
    """Add module-level code that wasn't captured by any top-level node."""
    covered_sorted = sorted(covered)
    gaps: list[tuple[int, int]] = []
    pos = 0
    for start, end in covered_sorted:
        if start > pos:
            gaps.append((pos, start))
        pos = max(pos, end)
    if pos < total_bytes:
        gaps.append((pos, total_bytes))

    for gap_start, gap_end in gaps:
        text = source_bytes[gap_start:gap_end].decode("utf-8", errors="replace").strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        start_line = source_bytes[:gap_start].count(b"\n") + 1
        end_line = source_bytes[:gap_end].count(b"\n") + 1
        chunks.append(CodeChunk(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            language=language,
            chunk_type="module",
            name="",
            imports=imports,
            calls=[],
            source=text,
        ))


def _line_based_chunks(
    file_path: str,
    source: str,
    language: str,
    start_line_offset: int = 0,
) -> list[CodeChunk]:
    """
    Fallback: split source into chunks of roughly MAX_CHUNK_TOKENS tokens
    at line boundaries.
    """
    lines = source.split("\n")
    chunks: list[CodeChunk] = []
    current_lines: list[str] = []
    current_tokens = 0
    chunk_start = start_line_offset

    for i, line in enumerate(lines):
        line_tokens = len(_tokenizer.encode(line))
        if current_tokens + line_tokens > MAX_CHUNK_TOKENS and current_lines:
            text = "\n".join(current_lines)
            if len(text.strip()) >= MIN_CHUNK_CHARS:
                chunks.append(CodeChunk(
                    file_path=file_path,
                    start_line=chunk_start + 1,
                    end_line=chunk_start + len(current_lines),
                    language=language,
                    chunk_type="block",
                    name="",
                    source=text,
                ))
            current_lines = [line]
            current_tokens = line_tokens
            chunk_start = start_line_offset + i
        else:
            current_lines.append(line)
            current_tokens += line_tokens

    # Last chunk
    if current_lines:
        text = "\n".join(current_lines)
        if len(text.strip()) >= MIN_CHUNK_CHARS:
            chunks.append(CodeChunk(
                file_path=file_path,
                start_line=chunk_start + 1,
                end_line=chunk_start + len(current_lines),
                language=language,
                chunk_type="block",
                name="",
                source=text,
            ))

    return chunks


def _merge_tiny_chunks(chunks: list[CodeChunk]) -> list[CodeChunk]:
    """Merge adjacent chunks that are smaller than MIN_CHUNK_CHARS."""
    if not chunks:
        return chunks

    merged: list[CodeChunk] = [chunks[0]]
    for chunk in chunks[1:]:
        prev = merged[-1]
        # Merge if same file, adjacent lines, and previous is tiny
        if (
            prev.file_path == chunk.file_path
            and len(prev.source) < MIN_CHUNK_CHARS
            and prev.end_line >= chunk.start_line - 2  # allow 1-line gap
        ):
            prev.source = prev.source + "\n" + chunk.source
            prev.end_line = chunk.end_line
            prev.calls = sorted(set(prev.calls + chunk.calls))
            if chunk.name and not prev.name:
                prev.name = chunk.name
                prev.chunk_type = chunk.chunk_type
        else:
            merged.append(chunk)

    return merged
