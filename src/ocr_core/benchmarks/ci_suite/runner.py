"""
runner.py — Benchmark runner for medical OCR evaluation.
مشغل المعايير لتقييم جودة OCR الطبي.

Supports running benchmarks across multiple OCR engines and generating reports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ocr_core.benchmarks.ci_suite.ci import ThresholdChecker
from ocr_core.benchmarks.ci_suite.dataset import DatasetManager, TestCase
from ocr_core.benchmarks.ci_suite.metrics import calculate_all_metrics
from ocr_core.benchmarks.ci_suite.report import (
    BenchmarkReport,
    EngineResult,
    ReportGenerator,
)

console = Console()


@dataclass
class OCRPredictResult:
    """Result from an OCR engine prediction."""
    text: str
    confidence: float = 0.0
    latency: float = 0.0  # seconds


class OCREngine:
    """Abstract base for OCR engines."""

    name: str = "base"

    def predict(self, image_path: str) -> OCRPredictResult:
        raise NotImplementedError

    def is_available(self) -> bool:
        return False

    def warmup(self) -> None:
        pass


class PaddleOCREngine(OCREngine):
    """PaddleOCR engine wrapper."""

    name = "paddleocr"
    _instance = None

    def __init__(self):
        self._ocr = None

    def is_available(self) -> bool:
        try:
            import paddleocr  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
                use_textline_orientation=True,
            )
        return self._ocr

    def predict(self, image_path: str) -> OCRPredictResult:
        start = time.time()
        try:
            ocr = self._get_ocr()
            result = ocr.ocr(image_path, cls=True)
            texts = []
            total_conf = 0.0
            count = 0
            if result and result[0]:
                for line in result[0]:
                    texts.append(line[1][0])
                    total_conf += line[1][1]
                    count += 1
            avg_conf = total_conf / count if count > 0 else 0.0
            return OCRPredictResult(
                text="\n".join(texts),
                confidence=avg_conf,
                latency=time.time() - start,
            )
        except Exception as e:
            console.print(f"[yellow]PaddleOCR error: {e}[/yellow]")
            return OCRPredictResult(text="", confidence=0.0, latency=time.time() - start)

    def warmup(self) -> None:
        """Pre-load PaddleOCR models."""
        if self.is_available():
            self._get_ocr()


class TesseractEngine(OCREngine):
    """Tesseract OCR engine wrapper."""

    name = "tesseract"

    def is_available(self) -> bool:
        try:
            import pytesseract  # noqa: F401
            return True
        except ImportError:
            return False

    def predict(self, image_path: str) -> OCRPredictResult:
        start = time.time()
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang="eng+ara")
            return OCRPredictResult(
                text=text,
                confidence=0.0,  # Tesseract doesn't return confidence easily
                latency=time.time() - start,
            )
        except Exception as e:
            console.print(f"[yellow]Tesseract error: {e}[/yellow]")
            return OCRPredictResult(text="", confidence=0.0, latency=time.time() - start)


class EasyOCREngine(OCREngine):
    """EasyOCR engine wrapper."""

    name = "easyocr"
    _instance = None

    def __init__(self):
        self._reader = None

    def is_available(self) -> bool:
        try:
            import easyocr  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en", "ar"], gpu=False, verbose=False)
        return self._reader

    def predict(self, image_path: str) -> OCRPredictResult:
        start = time.time()
        try:
            reader = self._get_reader()
            results = reader.readtext(image_path)
            texts = [r[1] for r in results]
            confs = [r[2] for r in results]
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            return OCRPredictResult(
                text="\n".join(texts),
                confidence=avg_conf,
                latency=time.time() - start,
            )
        except Exception as e:
            console.print(f"[yellow]EasyOCR error: {e}[/yellow]")
            return OCRPredictResult(text="", confidence=0.0, latency=time.time() - start)

    def warmup(self) -> None:
        if self.is_available():
            self._get_reader()


class MockOCREngine(OCREngine):
    """Mock OCR engine for testing without real images."""

    name = "mock"

    def is_available(self) -> bool:
        return True

    def predict(self, image_path: str) -> OCRPredictResult:
        time.sleep(0.01)  # Simulate minimal latency
        # Return empty string since we don't have real images
        return OCRPredictResult(text="", confidence=0.0, latency=0.01)


class BenchmarkRunner:
    """Main benchmark runner that tests OCR engines against test cases."""

    ENGINES = {
        "paddleocr": PaddleOCREngine,
        "tesseract": TesseractEngine,
        "easyocr": EasyOCREngine,
        "mock": MockOCREngine,
    }

    def __init__(
        self,
        data_dir: str | None = None,
        output_dir: str | None = None,
        engines: list[str] | None = None,
        thresholds_path: str | None = None,
        baselines_path: str | None = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            data_dir: Path to benchmark data directory
            output_dir: Path to output reports directory
            engines: List of engine names to benchmark. None = all available.
            thresholds_path: Path to CI thresholds config
            baselines_path: Path to baseline snapshots config
        """
        self.dataset = DatasetManager(data_dir)
        self.dataset.load()

        self.report_gen = ReportGenerator(output_dir)
        self.ci_checker = ThresholdChecker(thresholds_path, baselines_path)

        self.engine_instances: dict[str, OCREngine] = {}
        self.selected_engines = engines or list(self.ENGINES.keys())

    def _init_engines(self, skip_unavailable: bool = True) -> list[str]:
        """Initialize selected OCR engines."""
        available = []
        for name in self.selected_engines:
            if name not in self.ENGINES:
                console.print(f"[yellow]Unknown engine: {name}[/yellow]")
                continue

            engine_class = self.ENGINES[name]
            instance = engine_class()

            if skip_unavailable and not instance.is_available():
                console.print(
                    f"[yellow]Engine '{name}' not available (missing dependencies). "
                    f"Skipping.[/yellow]"
                )
                continue

            self.engine_instances[name] = instance
            available.append(name)

        if not available:
            console.print("[red]No OCR engines available![/red]")
        return available

    def _run_single_engine(
        self,
        engine: OCREngine,
        cases: list[TestCase],
        images_available: bool = False,
    ) -> EngineResult:
        """Run benchmark for a single engine across all test cases."""
        result = EngineResult(engine_name=engine.name)
        total_start = time.time()

        # Track metrics per category
        lang_metrics: dict[str, list] = {}
        spec_metrics: dict[str, list] = {}
        diff_metrics: dict[str, list] = {}

        for case in cases:
            if images_available and case.image_path:
                pred = engine.predict(case.image_path)
            else:
                # No real images - return empty prediction
                pred = OCRPredictResult(text="", confidence=0.0, latency=0.001)

            metrics = calculate_all_metrics(case.ground_truth, pred.text)

            result.case_results.append({
                "case_id": case.id,
                "cer": metrics.cer,
                "wer": metrics.wer,
                "medical_accuracy": metrics.medical_accuracy,
                "latency": pred.latency,
                "confidence": pred.confidence,
            })

            result.total_cases += 1

            # Accumulate per-language metrics
            lang_metrics.setdefault(case.language, []).append(metrics)
            spec_metrics.setdefault(case.specialty, []).append(metrics)
            diff_metrics.setdefault(case.difficulty, []).append(metrics)

        total_time = time.time() - total_start

        # Calculate averages
        if result.case_results:
            result.avg_cer = sum(r["cer"] for r in result.case_results) / len(result.case_results)
            result.avg_wer = sum(r["wer"] for r in result.case_results) / len(result.case_results)
            result.avg_medical_accuracy = sum(
                r["medical_accuracy"] for r in result.case_results
            ) / len(result.case_results)
            result.avg_latency = sum(r["latency"] for r in result.case_results) / len(result.case_results)
            result.throughput = len(result.case_results) / total_time if total_time > 0 else 0

        # Calculate per-category averages
        for lang, metrics_list in lang_metrics.items():
            n = len(metrics_list)
            result.per_language[lang] = {
                "cer": sum(m.cer for m in metrics_list) / n,
                "wer": sum(m.wer for m in metrics_list) / n,
                "medical_accuracy": sum(m.medical_accuracy for m in metrics_list) / n,
            }

        for spec, metrics_list in spec_metrics.items():
            n = len(metrics_list)
            result.per_specialty[spec] = {
                "cer": sum(m.cer for m in metrics_list) / n,
                "wer": sum(m.wer for m in metrics_list) / n,
                "medical_accuracy": sum(m.medical_accuracy for m in metrics_list) / n,
            }

        for diff, metrics_list in diff_metrics.items():
            n = len(metrics_list)
            result.per_difficulty[diff] = {
                "cer": sum(m.cer for m in metrics_list) / n,
                "wer": sum(m.wer for m in metrics_list) / n,
                "medical_accuracy": sum(m.medical_accuracy for m in metrics_list) / n,
            }

        return result

    def run(
        self,
        language: str | None = None,
        specialty: str | None = None,
        difficulty: str | None = None,
        skip_unavailable: bool = True,
        check_ci: bool = False,
        formats: list[str] | None = None,
        images_available: bool = False,
    ) -> BenchmarkReport:
        """
        Run the full benchmark suite.

        Args:
            language: Filter test cases by language
            specialty: Filter by specialty
            difficulty: Filter by difficulty
            skip_unavailable: Skip engines that aren't installed
            check_ci: Run CI threshold checks after benchmarks
            formats: Report output formats (json, markdown, html)
            images_available: Whether real test images are available

        Returns:
            Complete BenchmarkReport
        """
        import datetime

        start_time = time.time()

        # Initialize engines
        available_engines = self._init_engines(skip_unavailable)

        # Filter test cases
        cases = self.dataset.filter(
            language=language,
            specialty=specialty,
            difficulty=difficulty,
        )

        if not cases:
            console.print("[red]No test cases matching filters![/red]")
            return BenchmarkReport()

        console.print("\n[bold]Medical OCR Benchmark Suite[/bold]")
        console.print(f"  Test cases: {len(cases)}")
        console.print(f"  Engines: {', '.join(available_engines)}")
        console.print(f"  Dataset stats: {self.dataset.get_stats()}")
        console.print()

        report = BenchmarkReport(
            timestamp=datetime.datetime.now().isoformat(),
            metadata={
                "languages": [c.language for c in cases],
                "specialties": list(set(c.specialty for c in cases)),
                "difficulties": list(set(c.difficulty for c in cases)),
                "total_cases": len(cases),
                "engines": available_engines,
            },
        )

        # Run each engine
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            for engine_name in available_engines:
                engine = self.engine_instances[engine_name]
                task = progress.add_task(
                    f"Running {engine_name}...",
                    total=len(cases),
                )

                console.print(f"[bold cyan]Benchmarking: {engine_name}[/bold cyan]")

                # Warm up engine
                engine.warmup()

                result = self._run_single_engine(engine, cases, images_available)
                report.engine_results[engine_name] = result

                progress.update(task, completed=len(cases))

                # Print summary
                console.print(
                    f"  CER: {result.avg_cer:.4f} | "
                    f"WER: {result.avg_wer:.4f} | "
                    f"Medical: {result.avg_medical_accuracy:.1%} | "
                    f"Latency: {result.avg_latency:.2f}s | "
                    f"Throughput: {result.throughput:.2f} img/s"
                )

        report.duration = time.time() - start_time
        console.print(f"\n[green]Benchmark completed in {report.duration:.1f}s[/green]")

        # Generate reports
        console.print("\n[bold]Generating reports...[/bold]")
        generated = self.report_gen.generate_all(report, formats)
        for path in generated:
            console.print(f"  📝 {path}")

        # CI check
        if check_ci:
            console.print("\n[bold]CI Threshold Check[/bold]")
            ci_result = self.ci_checker.check(report.engine_results)
            self.ci_checker.print_ci_result(ci_result)
            report.summary = {"ci_result": ci_result.summary}

        return report


@click.command()
@click.option("--data-dir", default=None, help="Path to benchmark data directory")
@click.option("--output-dir", default=None, help="Path to output reports directory")
@click.option("--engines", default=None, help="Comma-separated engine list (e.g., paddleocr,easyocr)")
@click.option("--language", default=None, type=click.Choice(["english", "arabic", "mixed"]), help="Filter by language")
@click.option("--specialty", default=None, help="Filter by specialty")
@click.option("--difficulty", default=None, type=click.Choice(["easy", "medium", "hard"]), help="Filter by difficulty")
@click.option("--skip-unavailable/--include-unavailable", default=True, help="Skip unavailable engines")
@click.option("--check-ci/--no-ci", default=True, help="Run CI threshold checks")
@click.option("--format", "formats", default="json,markdown,html", help="Output formats (json,markdown,html)")
@click.option("--images", is_flag=True, default=False, help="Real test images are available")
def main(
    data_dir: str | None,
    output_dir: str | None,
    engines: str | None,
    language: str | None,
    specialty: str | None,
    difficulty: str | None,
    skip_unavailable: bool,
    check_ci: bool,
    formats: str,
    images: bool,
):
    """Run medical OCR benchmarks. / تشغيل معايير تقييم OCR الطبي"""
    engine_list = engines.split(",") if engines else None
    format_list = formats.split(",") if formats else None

    runner = BenchmarkRunner(
        data_dir=data_dir,
        output_dir=output_dir,
        engines=engine_list,
    )

    runner.run(
        language=language,
        specialty=specialty,
        difficulty=difficulty,
        skip_unavailable=skip_unavailable,
        check_ci=check_ci,
        formats=format_list,
        images_available=images,
    )


if __name__ == "__main__":
    main()
