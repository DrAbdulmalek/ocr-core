"""
report.py — Report generation for benchmark results.
توليد التقارير لنتائج المعايير.

Supports Markdown, JSON, and HTML output formats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EngineResult:
    """Results for a single OCR engine."""
    engine_name: str
    total_cases: int = 0
    avg_cer: float = 0.0
    avg_wer: float = 0.0
    avg_medical_accuracy: float = 0.0
    avg_latency: float = 0.0  # seconds per image
    throughput: float = 0.0  # images per second
    per_language: dict[str, dict] = field(default_factory=dict)
    per_specialty: dict[str, dict] = field(default_factory=dict)
    per_difficulty: dict[str, dict] = field(default_factory=dict)
    case_results: list[dict] = field(default_factory=list)
    errors: int = 0


@dataclass
class BenchmarkReport:
    """Complete benchmark report across all engines."""
    timestamp: str = ""
    duration: float = 0.0  # Total benchmark time in seconds
    engine_results: dict[str, EngineResult] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """Generates benchmark reports in multiple formats."""

    def __init__(self, output_dir: str | None = None):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to write reports to. Defaults to reports/.
        """
        if output_dir is None:
            output_dir = Path.cwd() / "reports"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(
        self,
        report: BenchmarkReport,
        formats: list[str] | None = None,
    ) -> list[Path]:
        """
        Generate reports in all specified formats.

        Args:
            report: The benchmark report data
            formats: List of formats to generate. Defaults to ["json", "markdown", "html"]

        Returns:
            List of paths to generated report files
        """
        if formats is None:
            formats = ["json", "markdown", "html"]

        generated = []
        for fmt in formats:
            if fmt == "json":
                path = self.generate_json(report)
            elif fmt in ("markdown", "md"):
                path = self.generate_markdown(report)
            elif fmt == "html":
                path = self.generate_html(report)
            else:
                raise ValueError(f"Unknown format: {fmt}")
            generated.append(path)

        return generated

    def generate_json(self, report: BenchmarkReport) -> Path:
        """Generate a JSON report."""
        timestamp = report.timestamp or datetime.now().isoformat()
        filename = f"benchmark_{timestamp.replace(':', '-')}.json"
        filepath = self.output_dir / filename

        data = {
            "timestamp": timestamp,
            "duration_seconds": report.duration,
            "metadata": report.metadata,
            "engines": {},
            "summary": report.summary,
        }

        for engine_name, result in report.engine_results.items():
            data["engines"][engine_name] = {
                "total_cases": result.total_cases,
                "avg_cer": round(result.avg_cer, 4),
                "avg_wer": round(result.avg_wer, 4),
                "avg_medical_accuracy": round(result.avg_medical_accuracy, 4),
                "avg_latency_seconds": round(result.avg_latency, 2),
                "throughput_ips": round(result.throughput, 2),
                "per_language": result.per_language,
                "per_specialty": result.per_specialty,
                "per_difficulty": result.per_difficulty,
                "errors": result.errors,
            }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        return filepath

    def generate_markdown(self, report: BenchmarkReport) -> Path:
        """Generate a Markdown report."""
        timestamp = report.timestamp or datetime.now().isoformat()
        filename = f"benchmark_{timestamp.replace(':', '-')}.md"
        filepath = self.output_dir / filename

        lines = []
        lines.append("# Medical OCR Benchmark Report / تقرير معايير تقييم OCR الطبي")
        lines.append("")
        lines.append(f"**Date / التاريخ:** {timestamp}")
        lines.append(f"**Duration / المدة:** {report.duration:.1f} seconds")
        lines.append("")

        # Summary table
        lines.append("## Summary / ملخص")
        lines.append("")
        lines.append("| Engine / المحرك | Cases | CER | WER | Medical Acc | Latency (s) | Throughput |")
        lines.append("|---|---|---|---|---|---|---|")

        for name, result in report.engine_results.items():
            cer_status = "✅" if result.avg_cer < 0.15 else ("⚠️" if result.avg_cer < 0.25 else "❌")
            lines.append(
                f"| {name} | {result.total_cases} | {result.avg_cer:.4f} {cer_status} "
                f"| {result.avg_wer:.4f} | {result.avg_medical_accuracy:.1%} "
                f"| {result.avg_latency:.2f} | {result.throughput:.2f} img/s |"
            )

        lines.append("")

        # Per-engine details
        for name, result in report.engine_results.items():
            lines.append(f"## {name} - Detailed Results / نتائج مفصلة")
            lines.append("")

            # Per language
            if result.per_language:
                lines.append("### By Language / حسب اللغة")
                lines.append("")
                lines.append("| Language | CER | WER | Medical Acc |")
                lines.append("|---|---|---|---|")
                for lang, metrics in sorted(result.per_language.items()):
                    lines.append(
                        f"| {lang} | {metrics.get('cer', 0):.4f} "
                        f"| {metrics.get('wer', 0):.4f} "
                        f"| {metrics.get('medical_accuracy', 0):.1%} |"
                    )
                lines.append("")

            # Per difficulty
            if result.per_difficulty:
                lines.append("### By Difficulty / حسب الصعوبة")
                lines.append("")
                lines.append("| Difficulty | CER | WER | Medical Acc |")
                lines.append("|---|---|---|---|")
                for diff, metrics in sorted(result.per_difficulty.items()):
                    lines.append(
                        f"| {diff} | {metrics.get('cer', 0):.4f} "
                        f"| {metrics.get('wer', 0):.4f} "
                        f"| {metrics.get('medical_accuracy', 0):.1%} |"
                    )
                lines.append("")

            # Per specialty
            if result.per_specialty:
                lines.append("### By Specialty / حسب التخصص")
                lines.append("")
                lines.append("| Specialty | CER | WER | Medical Acc |")
                lines.append("|---|---|---|---|")
                for spec, metrics in sorted(result.per_specialty.items()):
                    lines.append(
                        f"| {spec} | {metrics.get('cer', 0):.4f} "
                        f"| {metrics.get('wer', 0):.4f} "
                        f"| {metrics.get('medical_accuracy', 0):.1%} |"
                    )
                lines.append("")

        # Warnings
        lines.append("## CI Threshold Warnings / تحذيرات أطراف CI")
        lines.append("")
        warnings_found = False
        for name, result in report.engine_results.items():
            if result.avg_cer > 0.15:
                lines.append(f"- ⚠️ **{name}**: CER {result.avg_cer:.4f} exceeds 0.15 threshold")
                warnings_found = True
            if result.avg_wer > 0.25:
                lines.append(f"- ⚠️ **{name}**: WER {result.avg_wer:.4f} exceeds 0.25 threshold")
                warnings_found = True
        if not warnings_found:
            lines.append("✅ All engines within CI thresholds.")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return filepath

    def generate_html(self, report: BenchmarkReport) -> Path:
        """Generate an HTML report with styling."""
        timestamp = report.timestamp or datetime.now().isoformat()
        filename = f"benchmark_{timestamp.replace(':', '-')}.html"
        filepath = self.output_dir / filename

        html_parts = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append('<html lang="en" dir="ltr">')
        html_parts.append("<head>")
        html_parts.append('<meta charset="UTF-8">')
        html_parts.append(f"<title>Medical OCR Benchmark - {timestamp[:10]}</title>")
        html_parts.append("<style>")
        html_parts.append("""
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }
        h2 { color: #333; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }
        th { background: #1a73e8; color: white; padding: 12px; text-align: left; }
        td { padding: 10px 12px; border-bottom: 1px solid #eee; }
        tr:hover { background: #f0f7ff; }
        .good { color: #0d904f; font-weight: bold; }
        .warn { color: #e37400; font-weight: bold; }
        .fail { color: #d93025; font-weight: bold; }
        .summary-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .metric { display: inline-block; text-align: center; padding: 15px; min-width: 150px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #1a73e8; }
        .metric-label { font-size: 12px; color: #666; text-transform: uppercase; }
        .warning { background: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 4px; margin: 10px 0; }
        """)
        html_parts.append("</style>")
        html_parts.append("</head>")
        html_parts.append("<body>")
        html_parts.append("<h1>🏥 Medical OCR Benchmark Report</h1>")
        html_parts.append(f"<p>Date: {timestamp} | Duration: {report.duration:.1f}s</p>")

        # Summary table
        html_parts.append('<div class="summary-card">')
        html_parts.append("<h2>Engine Comparison / مقارنة المحركات</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Engine</th><th>Cases</th><th>CER</th><th>WER</th><th>Medical Acc</th><th>Latency</th><th>Throughput</th></tr>")

        for name, result in report.engine_results.items():
            cer_class = "good" if result.avg_cer < 0.15 else ("warn" if result.avg_cer < 0.25 else "fail")
            html_parts.append(
                f'<tr><td><strong>{name}</strong></td><td>{result.total_cases}</td>'
                f'<td class="{cer_class}">{result.avg_cer:.4f}</td>'
                f'<td>{result.avg_wer:.4f}</td>'
                f'<td>{result.avg_medical_accuracy:.1%}</td>'
                f'<td>{result.avg_latency:.2f}s</td>'
                f'<td>{result.throughput:.2f} img/s</td></tr>'
            )

        html_parts.append("</table>")
        html_parts.append("</div>")

        # Per-engine details
        for name, result in report.engine_results.items():
            html_parts.append('<div class="summary-card">')
            html_parts.append(f"<h2>{name} - Details</h2>")

            if result.per_language:
                html_parts.append("<h3>By Language</h3><table>")
                html_parts.append("<tr><th>Language</th><th>CER</th><th>WER</th><th>Medical Acc</th></tr>")
                for lang, m in sorted(result.per_language.items()):
                    html_parts.append(
                        f"<tr><td>{lang}</td><td>{m.get('cer', 0):.4f}</td>"
                        f"<td>{m.get('wer', 0):.4f}</td><td>{m.get('medical_accuracy', 0):.1%}</td></tr>"
                    )
                html_parts.append("</table>")

            if result.per_difficulty:
                html_parts.append("<h3>By Difficulty</h3><table>")
                html_parts.append("<tr><th>Difficulty</th><th>CER</th><th>WER</th><th>Medical Acc</th></tr>")
                for diff, m in sorted(result.per_difficulty.items()):
                    html_parts.append(
                        f"<tr><td>{diff}</td><td>{m.get('cer', 0):.4f}</td>"
                        f"<td>{m.get('wer', 0):.4f}</td><td>{m.get('medical_accuracy', 0):.1%}</td></tr>"
                    )
                html_parts.append("</table>")

            html_parts.append("</div>")

        html_parts.append("</body></html>")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

        return filepath
