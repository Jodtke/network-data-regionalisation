from __future__ import annotations

"""Small Excel reader for the TYNDP workbook layouts used in this pipeline.

Only a limited subset of workbook functionality is needed for preprocessing:
cell ranges, row iteration, and typed scalar extraction from sheets with merged
or formatted headers. Keeping this reader local avoids depending on a full Excel
stack in scripts that only need reproducible extraction of published scenario
tables.
"""

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"m": MAIN_NS, "r": REL_NS, "p": PKG_REL_NS}
CELL_RE = re.compile(r"([A-Z]+)(\d+)")


def column_number(column_letters: str) -> int:
    value = 0
    for char in column_letters:
        value = value * 26 + ord(char.upper()) - 64
    return value


def parse_cell_reference(reference: str) -> tuple[int, int]:
    match = CELL_RE.match(reference)
    if not match:
        raise ValueError(f"Invalid cell reference: {reference}")
    return int(match.group(2)), column_number(match.group(1))


def normalize_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("â‚¬", "eur").replace("€", "eur").replace("co₂", "co2")
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("â‚¬", "eur").replace("€", "eur").replace("co₂", "co2")
    text = re.sub(r"[^a-z0-9/]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unspecified"


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    parsed = safe_float(value)
    if parsed is None:
        return default
    return int(round(parsed))


def format_number(value: Any) -> str:
    parsed = safe_float(value)
    if parsed is None:
        return "" if value in (None, "") else str(value)
    return f"{parsed:.12g}"


@dataclass(frozen=True)
class XlsxSheet:
    name: str
    rows: dict[int, dict[int, str]]
    dimension: str | None = None

    def cell(self, row: int, column: int, default: str = "") -> str:
        return self.rows.get(row, {}).get(column, default)

    def row(self, row: int) -> dict[int, str]:
        return self.rows.get(row, {})

    @property
    def max_row(self) -> int:
        return max(self.rows, default=0)

    @property
    def max_column(self) -> int:
        return max((max(row) for row in self.rows.values() if row), default=0)

    def nonempty_columns(self, rows: Iterable[int], *, min_column: int = 1) -> list[int]:
        columns: set[int] = set()
        for row_number in rows:
            for column, value in self.row(row_number).items():
                if column >= min_column and str(value).strip():
                    columns.add(column)
        return sorted(columns)


def _sheet_path(target: str) -> str:
    clean = target.lstrip("/")
    if clean.startswith("xl/"):
        return clean
    return f"xl/{clean}"


def _read_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values: list[str] = []
    for item in root.findall("m:si", NS):
        values.append("".join(text.text or "" for text in item.iter(f"{{{MAIN_NS}}}t")))
    return values


def _read_sheet_targets(workbook: zipfile.ZipFile) -> dict[str, str]:
    workbook_xml = ET.fromstring(workbook.read("xl/workbook.xml"))
    rels_xml = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels_xml.findall("p:Relationship", NS)
    }
    targets: dict[str, str] = {}
    for sheet in workbook_xml.find("m:sheets", NS).findall("m:sheet", NS):
        name = sheet.attrib["name"]
        rel_id = sheet.attrib[f"{{{REL_NS}}}id"]
        targets[name] = _sheet_path(rel_targets[rel_id])
    return targets


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.iter(f"{{{MAIN_NS}}}t"))

    raw_value = cell.find("m:v", NS)
    if raw_value is None:
        return ""
    text = raw_value.text or ""
    if cell_type == "s":
        try:
            return shared_strings[int(text)]
        except (IndexError, ValueError):
            return text
    if cell_type == "b":
        return "TRUE" if text == "1" else "FALSE"
    return text


def read_sheet(path: Path, sheet_name: str) -> XlsxSheet:
    with zipfile.ZipFile(path) as workbook:
        shared_strings = _read_shared_strings(workbook)
        targets = _read_sheet_targets(workbook)
        if sheet_name not in targets:
            raise KeyError(f"{path.name} does not contain sheet '{sheet_name}'.")
        root = ET.fromstring(workbook.read(targets[sheet_name]))
        dimension_node = root.find("m:dimension", NS)
        dimension = dimension_node.attrib.get("ref") if dimension_node is not None else None
        rows: dict[int, dict[int, str]] = {}
        for row_node in root.findall(".//m:sheetData/m:row", NS):
            values: dict[int, str] = {}
            row_number = int(row_node.attrib.get("r", "0"))
            for cell in row_node.findall("m:c", NS):
                reference = cell.attrib.get("r", "")
                if not reference:
                    continue
                parsed_row, column = parse_cell_reference(reference)
                row_number = parsed_row or row_number
                value = _cell_value(cell, shared_strings)
                if value != "":
                    values[column] = value
            if values:
                rows[row_number] = values
        return XlsxSheet(name=sheet_name, rows=rows, dimension=dimension)


def list_sheet_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as workbook:
        return list(_read_sheet_targets(workbook))


def find_table_header_row(sheet: XlsxSheet, *, date_label: str = "date", hour_label: str = "hour") -> int:
    wanted_date = normalize_key(date_label)
    wanted_hour = normalize_key(hour_label)
    for row_number, values in sorted(sheet.rows.items()):
        if normalize_key(values.get(1)) == wanted_date and normalize_key(values.get(2)) == wanted_hour:
            return row_number
    raise ValueError(f"Could not find Date/Hour header row in sheet '{sheet.name}'.")


def iter_hourly_rows(sheet: XlsxSheet, *, header_row: int | None = None) -> Iterable[tuple[int, str, int, dict[int, str]]]:
    start = (header_row if header_row is not None else find_table_header_row(sheet)) + 1
    timestep = 0
    for row_number in range(start, sheet.max_row + 1):
        values = sheet.row(row_number)
        date = str(values.get(1, "")).strip()
        hour = safe_int(values.get(2))
        if not date or hour is None:
            continue
        timestep += 1
        yield timestep, date, hour, values
