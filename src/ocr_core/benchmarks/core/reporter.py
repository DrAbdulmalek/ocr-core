"""Report Generator — Markdown, JSON, and HTML outputs.

Takes benchmark results and renders them into human-readable formats
suitable for CI artifacts, pull request comments, or documentation.
"""

import json
import os
from datetime import datetime, timezone


class BenchmarkReporter:
    """Generate benchmark reports in multiple formats.

    Supported formats:
        - Markdown (.md)
        - JSON (.json)
        - HTML (.html)
    """

    def __init__(self, results: dict):
        """Initialize the reporter with benchmark results.

        Args:
            results: A dict as returned by ``BenchmarkRunner.run_all()``
                or ``BenchmarkSuite.summary()``.
        """
        self.results = results
        self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def to_json(self) -> dict:
        """Convert results to a JSON-serializable dict.

        Returns:
            The results dict with added metadata.
        """
        output = dict(self.results)
        output["_metadata"] = {
            "generated_at": self.timestamp,
            "generator": "medical-ocr-benchmarks",
            "version": "1.0.0",
        }
        return output

    def to_markdown(self) -> str:
        """Render benchmark results as a Markdown report.

        Returns:
            A formatted Markdown string.
        """
        lines: list[str] = []
        lines.append("# 🏥 Medical OCR Benchmark Report")
        lines.append("")
        lines.append(f"**Generated:** {self.timestamp}")
        lines.append("**Version:** 1.0.0")
        lines.append("")

        # Summary section
        if "summary" in self.results:
            summary = self.results["summary"]
            lines.append("## Summary")
            lines.append("")
            if isinstance(summary, dict):
                lines.append("| Metric | Mean | Median | Std Dev | Min | Max |")
                lines.append("|--------|------|--------|---------|-----|-----|")
                for metric_name, stats in summary.get("metrics_summary", {}).items():
                    if isinstance(stats, dict) and "mean" in stats:
                        lines.append(
                            f"| {metric_name} | {stats.get('mean', 'N/A')} | "
                            f"{stats.get('median', 'N/A')} | {stats.get('stdev', 'N/A')} | "
                            f"{stats.get('min', 'N/A')} | {stats.get('max', 'N/A')} |"
                        )
                lines.append("")

        # Individual results
        if "results" in self.results:
            results_list = self.results["results"]
            if results_list:
                lines.append("## Benchmark Results")
                lines.append("")

                for result in results_list:
                    name = result.get("benchmark_name", "unknown")
                    dataset = result.get("dataset", "N/A")
                    metrics = result.get("metrics", {})

                    lines.append(f"### {name} — `{dataset}`")
                    lines.append("")

                    if "error" in metrics:
                        lines.append(f"⚠️ **Error:** {metrics['error']}")
                    else:
                        lines.append("| Metric | Value |")
                        lines.append("|--------|-------|")
                        for key, val in metrics.items():
                            if isinstance(val, float):
                                val_str = f"{val:.6f}"
                            elif isinstance(val, dict):
                                val_str = json.dumps(val, ensure_ascii=False)
                            else:
                                val_str = str(val)
                            lines.append(f"| {key} | {val_str} |")

                    lines.append("")

        # Comparison section
        if "comparison" in self.results:
            comparison = self.results["comparison"]
            lines.append("## Comparison")
            lines.append("")
            lines.append(f"**Winner:** {self.results.get('winner', 'N/A')}")
            lines.append("")
            lines.append("| Metric | Value A | Value B | Delta | Delta % |")
            lines.append("|--------|---------|---------|-------|---------|")

            for metric, data in comparison.items():
                if isinstance(data, dict):
                    val_a = data.get("value_a", "N/A")
                    val_b = data.get("value_b", "N/A")
                    delta = data.get("delta", "—")
                    delta_pct = data.get("delta_percent", "—")

                    def fmt(v):
                        if isinstance(v, float):
                            return f"{v:.6f}"
                        return str(v)

                    lines.append(
                        f"| {metric} | {fmt(val_a)} | {fmt(val_b)} | {fmt(delta)} | {fmt(delta_pct)} |"
                    )

            lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Render benchmark results as an HTML document.

        Returns:
            A complete HTML document string with embedded CSS.
        """
        md = self.to_markdown()

        # Simple markdown-to-HTML conversion for the report
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "  <meta charset='utf-8'>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1'>",
            "  <title>Medical OCR Benchmark Report</title>",
            "  <style>",
            "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #24292f; background: #ffffff; line-height: 1.6; }",
            "    h1 { border-bottom: 2px solid #d0d7de; padding-bottom: 0.3em; }",
            "    h2 { border-bottom: 1px solid #d0d7de; padding-bottom: 0.3em; margin-top: 2em; }",
            "    h3 { margin-top: 1.5em; }",
            "    table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
            "    th, td { border: 1px solid #d0d7de; padding: 8px 12px; text-align: left; }",
            "    th { background-color: #f6f8fa; font-weight: 600; }",
            "    tr:nth-child(even) { background-color: #f6f8fa; }",
            "    code { background-color: #eff1f3; padding: 0.2em 0.4em; border-radius: 3px; font-size: 85%; }",
            "    .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 85%; font-weight: 600; }",
            "    .badge-good { background-color: #dafbe1; color: #116329; }",
            "    .badge-warn { background-color: #fff8c5; color: #9a6700; }",
            "    .badge-error { background-color: #ffebe9; color: #82071e; }",
            "    .timestamp { color: #57606a; font-size: 0.9em; }",
            "    .error { color: #82071e; font-weight: 600; }",
            "  </style>",
            "</head>",
            "<body>",
            self._markdown_to_html(md),
            "</body>",
            "</html>",
        ]

        return "\n".join(html_parts)

    @staticmethod
    def _markdown_to_html(md: str) -> str:
        """Convert basic Markdown to HTML.

        Handles headings, tables, bold, code, and paragraphs.

        Args:
            md: Markdown source string.

        Returns:
            HTML string.
        """
        lines = md.split("\n")
        html_lines: list[str] = []
        in_table = False
        table_headers: list[str] = []

        for line in lines:
            stripped = line.strip()

            # Headings
            if stripped.startswith("### "):
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                html_lines.append(f"<h3>{_inline_html(stripped[4:])}</h3>")
                continue
            if stripped.startswith("## "):
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                html_lines.append(f"<h2>{_inline_html(stripped[3:])}</h2>")
                continue
            if stripped.startswith("# "):
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                html_lines.append(f"<h1>{_inline_html(stripped[2:])}</h1>")
                continue

            # Table separator
            if stripped.startswith("|---") or stripped.startswith("|:"):
                if not in_table:
                    continue
                # Skip separator line, headers already captured
                continue

            # Table row
            if stripped.startswith("|"):
                if not in_table:
                    # First table row — headers
                    cells = [c.strip() for c in stripped.split("|")[1:-1]]
                    table_headers = cells
                    html_lines.append("<table>")
                    html_lines.append("<thead><tr>")
                    for cell in cells:
                        html_lines.append(f"  <th>{_inline_html(cell)}</th>")
                    html_lines.append("</tr></thead>")
                    html_lines.append("<tbody>")
                    in_table = True
                else:
                    cells = [c.strip() for c in stripped.split("|")[1:-1]]
                    html_lines.append("<tr>")
                    for i, cell in enumerate(cells):
                        css_class = ""
                        val = cell
                        # Apply color badges for certain patterns
                        try:
                            if "error" in (table_headers[i] if i < len(table_headers) else "").lower():
                                float(val)
                                css_class = ' class="error"'
                        except (ValueError, IndexError):
                            pass
                        html_lines.append(f"  <td{css_class}>{_inline_html(val)}</td>")
                    html_lines.append("</tr>")
                continue

            # Close table if leaving table context
            if in_table:
                html_lines.append("</tbody></table>")
                in_table = False

            # Empty line
            if not stripped:
                continue

            # Regular paragraph
            html_lines.append(f"<p>{_inline_html(stripped)}</p>")

        if in_table:
            html_lines.append("</tbody></table>")

        return "\n".join(html_lines)

    def save(self, output_path: str, format: str = "markdown") -> str:
        """Save the report to a file.

        Args:
            output_path: File path to write the report to. The file
                extension is overridden if format is specified.
            format: One of 'markdown', 'json', 'html'.

        Returns:
            The absolute path of the saved file.

        Raises:
            ValueError: If the format is not recognized.
        """
        # Override extension based on format
        ext_map = {
            "markdown": ".md",
            "json": ".json",
            "html": ".html",
        }
        if format not in ext_map:
            raise ValueError(
                f"Unknown format '{format}'. Supported: {', '.join(ext_map.keys())}"
            )

        base, _ = os.path.splitext(output_path)
        final_path = base + ext_map[format]

        # Ensure directory exists
        os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)

        if format == "markdown":
            content = self.to_markdown()
        elif format == "json":
            content = json.dumps(self.to_json(), indent=2, ensure_ascii=False, default=str)
        elif format == "html":
            content = self.to_html()
        else:
            content = str(self.results)

        with open(final_path, "w", encoding="utf-8") as f:
            f.write(content)

        return os.path.abspath(final_path)


def _inline_html(text: str) -> str:
    """Convert inline Markdown formatting to HTML.

    Handles **bold**, `code`, and plain text.

    Args:
        text: Markdown text.

    Returns:
        HTML string with inline formatting.
    """
    import re

    # Escape HTML entities
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)

    # Inline code
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)

    return text
