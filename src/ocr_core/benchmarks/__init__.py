"""Medical OCR Benchmarks — unified benchmark suite (ported from omni
packages/benchmark_core, MIT).

Layout:
- core/, ocr/, structure/, extended/, postprocessor/, datasets/ — the omni
  benchmark suite (verbatim, imports rewritten to ocr_core.benchmarks.*).
- ci_suite/ — the leaner CI-gate benchmark package (omni
  packages/benchmark_core/src/benchmarks + threshold_checker + ab_testing),
  namespaced under ci_suite/ to avoid module-name collision with the suite
  tree above. Carries its own unit tests.
- Orchestration CLIs (run_all_benchmarks, scanner_fixer_benchmark) were NOT
  ported — tied to the omni application context.
"""
__version__ = "1.0.0"
