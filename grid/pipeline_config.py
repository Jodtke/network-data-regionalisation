from __future__ import annotations

"""Minimal configuration helpers shared by grid preprocessing scripts.

The workflow uses small YAML-like scenario files to keep paths, target years,
and method switches outside the code. These helpers provide only the subset of
YAML features needed by the pipeline, with explicit errors for unsupported
structures so that publication runs remain reproducible.
"""

import ast
from pathlib import Path
from typing import Any


def load_yaml_like(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    if not text.strip():
        return {}

    try:
        import yaml  # type: ignore
    except ImportError:
        data = _load_simple_yaml(text)
    else:
        data = yaml.safe_load(text)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return []
        if "," in txt:
            return [part.strip() for part in txt.split(",") if part.strip()]
        return [txt]
    return [value]


def resolve_path(value: Any, *, base_dir: str | Path | None = None) -> Path | None:
    if value in (None, ""):
        return None

    path = value if isinstance(value, Path) else Path(str(value))
    path = Path(str(path).replace("/", "\\")).expanduser()
    if path.is_absolute():
        return path

    if base_dir is None:
        return path.resolve()
    return (Path(base_dir) / path).resolve()


def _load_simple_yaml(text: str) -> dict[str, Any]:
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped = _strip_yaml_comment(raw_line).rstrip()
        if not stripped.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, stripped.lstrip()))

    if not lines:
        return {}

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for idx, (indent, content) in enumerate(lines):
        while indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        next_indent = lines[idx + 1][0] if idx + 1 < len(lines) else None
        next_content = lines[idx + 1][1] if idx + 1 < len(lines) else None

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Unexpected list item outside list context: {content}")
            item_text = content[2:].strip()
            if not item_text:
                child = _new_container(indent, next_indent, next_content)
                parent.append(child)
                stack.append((indent, child))
                continue
            if item_text.endswith(":") and ":" not in item_text[:-1]:
                child_dict: dict[str, Any] = {}
                key = item_text[:-1].strip()
                child_dict[key] = _new_container(indent, next_indent, next_content)
                parent.append(child_dict)
                stack.append((indent, child_dict[key]))
                continue
            parent.append(_parse_scalar(item_text))
            continue

        if ":" not in content:
            raise ValueError(f"Expected 'key: value' YAML entry, got: {content}")

        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not isinstance(parent, dict):
            raise ValueError(f"Unexpected mapping entry inside list context: {content}")

        if value == "":
            child = _new_container(indent, next_indent, next_content)
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)

    return root


def _new_container(
    indent: int,
    next_indent: int | None,
    next_content: str | None,
) -> Any:
    if next_indent is not None and next_indent > indent and next_content is not None:
        if next_content.startswith("- "):
            return []
    return {}


def _parse_scalar(value: str) -> Any:
    txt = value.strip()
    if txt == "":
        return ""

    lower = txt.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"null", "none", "~"}:
        return None

    if txt.startswith("[") and txt.endswith("]"):
        inner = txt[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in _split_inline_list(inner)]

    if (txt.startswith("'") and txt.endswith("'")) or (
        txt.startswith('"') and txt.endswith('"')
    ):
        return ast.literal_eval(txt)

    try:
        return int(txt)
    except ValueError:
        pass

    try:
        return float(txt)
    except ValueError:
        return txt


def _split_inline_list(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False

    for char in value:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double

        if char == "," and not in_single and not in_double:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _strip_yaml_comment(line: str) -> str:
    current: list[str] = []
    in_single = False
    in_double = False

    for char in line:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            break
        current.append(char)

    return "".join(current)
