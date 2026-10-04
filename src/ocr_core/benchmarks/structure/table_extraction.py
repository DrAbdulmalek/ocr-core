"""
Table Extraction Benchmark
============================
Benchmarks table detection and extraction from medical documents.
Measures cell accuracy, row completeness, and structural preservation.

Author: Dr. Abdulmalek
Version: 1.0.0
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TableCell:
    row: int
    col: int
    expected: str
    predicted: str = ""
    is_correct: bool = False


@dataclass
class TableBenchmarkResult:
    table_id: str
    table_type: str  # "lab_report", "prescription", "vital_signs", "icd_codes"
    total_cells: int
    correct_cells: int
    cell_accuracy: float
    rows_complete: int
    total_rows: int
    row_completeness: float
    header_preserved: bool
    structure_preserved: bool  # merged cells, spanning
    latency_ms: float
    cell_errors: list[dict] = field(default_factory=list)


class TableExtractionBenchmark:
    """
    Benchmark for table extraction quality in medical documents.

    Medical table types:
    - Lab reports (test name, value, unit, reference range)
    - Prescriptions (drug, dosage, frequency, duration)
    - Vital signs (parameter, value, time, trend)
    - ICD/code tables (code, description, category)
    """

    # Sample golden table data for medical documents
    SAMPLE_TABLES = {
        "lab_report_001": {
            "type": "lab_report",
            "rows": [
                ["Test Name", "Value", "Unit", "Reference Range"],
                ["Hemoglobin", "12.5", "g/dL", "13.5-17.5"],
                ["WBC", "7500", "cells/uL", "4500-11000"],
                ["Platelets", "250000", "cells/uL", "150000-400000"],
                ["Glucose (Fasting)", "95", "mg/dL", "70-100"],
                ["Creatinine", "0.9", "mg/dL", "0.7-1.3"],
                ["TSH", "2.5", "mIU/L", "0.4-4.0"],
            ],
            "expected_cells": 28,
            "language": "mixed",
        },
        "prescription_001": {
            "type": "prescription",
            "rows": [
                ["Drug Name", "Dosage", "Frequency", "Duration"],
                ["Metformin", "500mg", "Twice daily", "3 months"],
                ["Amlodipine", "5mg", "Once daily", "Ongoing"],
                ["Omeprazole", "20mg", "Before breakfast", "2 weeks"],
                ["Atorvastatin", "10mg", "At bedtime", "Ongoing"],
            ],
            "expected_cells": 16,
            "language": "english",
        },
        "vital_signs_001": {
            "type": "vital_signs",
            "rows": [
                ["Parameter", "08:00", "12:00", "16:00", "20:00"],
                ["BP (mmHg)", "120/80", "130/85", "125/82", "118/76"],
                ["HR (bpm)", "72", "78", "75", "70"],
                ["SpO2 (%)", "98", "97", "98", "99"],
                ["Temp (C)", "36.5", "37.0", "36.8", "36.6"],
                ["RR (/min)", "16", "18", "17", "15"],
            ],
            "expected_cells": 30,
            "language": "english",
        },
        "arabic_lab_001": {
            "type": "lab_report",
            "rows": [
                ["اسم التحليل", "القيمة", "الوحدة", "المعدل الطبيعي"],
                ["هيموغلوبين", "11.8", "g/dL", "12.0-15.5"],
                ["كريات الدم البيضاء", "6800", "خلية/ميكرولتر", "4500-11000"],
                ["صفيحات الدم", "220000", "خلية/ميكرولتر", "150000-400000"],
                ["سكر الدم", "110", "mg/dL", "70-100"],
            ],
            "expected_cells": 16,
            "language": "arabic",
        },
    }

    def __init__(self, ocr_func=None):
        """
        Args:
            ocr_func: Optional function(image_path) -> str for real OCR testing.
                      If None, uses text-based comparison only.
        """
        self.ocr_func = ocr_func

    def evaluate_table(self, table_id: str, predicted_rows: list[list[str]]) -> TableBenchmarkResult:
        """
        Evaluate a single table extraction.

        Args:
            table_id: Key from SAMPLE_TABLES
            predicted_rows: Extracted table rows from OCR
        """
        if table_id not in self.SAMPLE_TABLES:
            raise ValueError(f"Unknown table: {table_id}")

        expected_table = self.SAMPLE_TABLES[table_id]
        expected_rows = expected_table["rows"]
        table_type = expected_table["type"]

        start = time.time()

        # Build cell-level comparison
        cells: list[TableCell] = []
        cell_errors = []
        correct_count = 0
        total_expected = 0

        max_rows = max(len(expected_rows), len(predicted_rows))

        for r in range(max_rows):
            exp_row = expected_rows[r] if r < len(expected_rows) else []
            pred_row = predicted_rows[r] if r < len(predicted_rows) else []
            max_cols = max(len(exp_row), len(pred_row))

            for c in range(max_cols):
                exp_val = exp_row[c].strip() if c < len(exp_row) else ""
                pred_val = pred_row[c].strip() if c < len(pred_row) else ""

                if exp_val:  # Only count expected cells
                    total_expected += 1
                    is_correct = self._cell_match(exp_val, pred_val)
                    if is_correct:
                        correct_count += 1
                    else:
                        cell_errors.append({
                            "row": r, "col": c,
                            "expected": exp_val, "predicted": pred_val,
                            "type": self._classify_error(exp_val, pred_val)
                        })

                cells.append(TableCell(
                    row=r, col=c,
                    expected=exp_val,
                    predicted=pred_val,
                    is_correct=exp_val == "" or (exp_val and self._cell_match(exp_val, pred_val))
                ))

        latency = (time.time() - start) * 1000

        # Row completeness: rows where all expected cells are non-empty in prediction
        rows_complete = 0
        total_data_rows = len(expected_rows) - 1  # Exclude header
        for r in range(1, len(expected_rows)):
            exp_row = expected_rows[r]
            pred_row = predicted_rows[r] if r < len(predicted_rows) else []
            all_present = all(
                any(self._cell_match(exp_row[c], pred_row[c2]) if c2 < len(pred_row) else False
                    for c2 in range(len(pred_row)))
                for c in range(len(exp_row)) if exp_row[c].strip()
            )
            if all_present:
                rows_complete += 1

        # Header preservation
        header_preserved = False
        if expected_rows and predicted_rows:
            exp_header = [c.strip() for c in expected_rows[0]]
            pred_header = [c.strip() for c in predicted_rows[0]] if predicted_rows else []
            header_preserved = len(exp_header) > 0 and len(pred_header) >= len(exp_header) * 0.7

        cell_accuracy = correct_count / max(total_expected, 1)
        row_completeness = rows_complete / max(total_data_rows, 1)

        return TableBenchmarkResult(
            table_id=table_id,
            table_type=table_type,
            total_cells=total_expected,
            correct_cells=correct_count,
            cell_accuracy=cell_accuracy,
            rows_complete=rows_complete,
            total_rows=total_data_rows,
            row_completeness=row_completeness,
            header_preserved=header_preserved,
            structure_preserved=header_preserved and row_completeness > 0.8,
            latency_ms=latency,
            cell_errors=cell_errors
        )

    def run_all(self) -> dict[str, Any]:
        """Run benchmark on all sample tables with simulated predictions."""
        results = []
        for table_id in self.SAMPLE_TABLES:
            # Simulate OCR output with realistic noise
            expected_rows = self.SAMPLE_TABLES[table_id]["rows"]
            noisy_rows = self._simulate_ocr_noise(expected_rows)
            result = self.evaluate_table(table_id, noisy_rows)
            results.append(result)

        return self._aggregate(results)

    def run_with_func(self, table_images: dict[str, Path]) -> dict[str, Any]:
        """Run benchmark using actual OCR function on table images."""
        if not self.ocr_func:
            raise ValueError("ocr_func required for image-based benchmarking")

        results = []
        for table_id, image_path in table_images.items():
            predicted_text = self.ocr_func(str(image_path))
            predicted_rows = self._parse_table_text(predicted_text)
            result = self.evaluate_table(table_id, predicted_rows)
            results.append(result)

        return self._aggregate(results)

    def _cell_match(self, expected: str, predicted: str) -> bool:
        """Flexible cell matching (numeric tolerance, whitespace tolerance)."""
        if not expected.strip():
            return True
        if not predicted.strip():
            return False

        exp = expected.strip()
        pred = predicted.strip()

        # Numeric tolerance (5%)
        try:
            exp_num = float(exp.replace(",", ""))
            pred_num = float(pred.replace(",", ""))
            if exp_num != 0:
                return abs(exp_num - pred_num) / abs(exp_num) < 0.05
        except (ValueError, TypeError):
            pass

        # Exact after normalization
        return exp.lower() == pred.lower()

    def _classify_error(self, expected: str, predicted: str) -> str:
        """Classify the type of cell error."""
        if not predicted.strip():
            return "missing"
        try:
            float(expected.replace(",", ""))
            float(predicted.replace(",", ""))
            return "numeric_mismatch"
        except (ValueError, TypeError):
            pass
        # Levenshtein-based
        if len(expected) > 3 and len(predicted) > 3:
            common = sum(1 for a, b in zip(expected, predicted) if a == b)
            if common / max(len(expected), len(predicted)) > 0.7:
                return "minor_typo"
        return "major_error"

    def _simulate_ocr_noise(self, rows: list[list[str]]) -> list[list[str]]:
        """Simulate realistic OCR noise on table data."""
        import random
        noisy = []
        for r, row in enumerate(rows):
            noisy_row = []
            for c, cell in enumerate(row):
                if random.random() < 0.05:  # 5% chance of error
                    if cell and cell[0].isdigit():
                        # Numeric noise
                        noisy_row.append(cell[:-1] + str(random.randint(0, 9)))
                    elif len(cell) > 3:
                        # Character noise
                        idx = random.randint(0, len(cell) - 1)
                        noisy_row.append(cell[:idx] + cell[idx+1:])
                    else:
                        noisy_row.append(cell)
                else:
                    noisy_row.append(cell)
            noisy.append(noisy_row)
        return noisy

    def _parse_table_text(self, text: str) -> list[list[str]]:
        """Parse tabular text output into rows of cells."""
        rows = []
        for line in text.strip().split("\n"):
            # Try tab-separated, pipe-separated, then whitespace
            if "\t" in line:
                rows.append(line.split("\t"))
            elif "|" in line:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if cells:
                    rows.append(cells)
            else:
                rows.append(line.split())
        return rows

    def _aggregate(self, results: list[TableBenchmarkResult]) -> dict[str, Any]:
        """Aggregate results across all tables."""
        if not results:
            return {"tables": [], "summary": {}}

        return {
            "tables": [
                {
                    "table_id": r.table_id,
                    "type": r.table_type,
                    "cell_accuracy": round(r.cell_accuracy, 4),
                    "row_completeness": round(r.row_completeness, 4),
                    "header_preserved": r.header_preserved,
                    "structure_preserved": r.structure_preserved,
                    "latency_ms": round(r.latency_ms, 2),
                    "error_count": len(r.cell_errors),
                }
                for r in results
            ],
            "summary": {
                "avg_cell_accuracy": round(
                    sum(r.cell_accuracy for r in results) / len(results), 4
                ),
                "avg_row_completeness": round(
                    sum(r.row_completeness for r in results) / len(results), 4
                ),
                "header_preservation_rate": round(
                    sum(1 for r in results if r.header_preserved) / len(results), 4
                ),
                "structure_preservation_rate": round(
                    sum(1 for r in results if r.structure_preserved) / len(results), 4
                ),
                "total_cell_errors": sum(len(r.cell_errors) for r in results),
                "total_cells": sum(r.total_cells for r in results),
                "tables_evaluated": len(results),
            }
        }
