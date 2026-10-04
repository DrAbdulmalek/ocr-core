#!/usr/bin/env python3
"""
Threshold Checker - Medical OCR Benchmarks (v2.0 - Full Integration)
Author: DrAbdulmalek
Description: Reads benchmark results and automatically decides:
             DEPLOY (if model passes thresholds) or RETRAIN (if regression detected)
Features: Auto-deploy to HF, auto-retrain trigger, Telegram notifications, A/B test integration
"""

import csv
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('threshold_check.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark test result"""
    model_name: str
    test_type: str  # 'printed' or 'handwritten'
    cer: float      # Character Error Rate (0.0 - 1.0)
    wer: float      # Word Error Rate (0.0 - 1.0)
    timestamp: str
    dataset_size: int


@dataclass
class ThresholdConfig:
    """Threshold configuration"""
    printed_cer_threshold: float = 0.05
    handwritten_cer_threshold: float = 0.12
    printed_wer_threshold: float = 0.10
    handwritten_wer_threshold: float = 0.20
    min_improvement_percent: float = 2.0  # Minimum 2% improvement for A/B deploy


class ResultsReader:
    """Reads benchmark results from JSON or CSV files"""

    @staticmethod
    def from_json(file_path: str) -> list[BenchmarkResult]:
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)
        results = []
        for item in data:
            results.append(BenchmarkResult(
                model_name=item.get('model_name', 'unknown'),
                test_type=item.get('test_type', 'printed'),
                cer=float(item.get('cer', 0.0)),
                wer=float(item.get('wer', 0.0)),
                timestamp=item.get('timestamp', datetime.now().isoformat()),
                dataset_size=int(item.get('dataset_size', 0))
            ))
        return results

    @staticmethod
    def from_csv(file_path: str) -> list[BenchmarkResult]:
        results = []
        with open(file_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(BenchmarkResult(
                    model_name=row.get('model_name', 'unknown'),
                    test_type=row.get('test_type', 'printed'),
                    cer=float(row.get('cer', 0.0)),
                    wer=float(row.get('wer', 0.0)),
                    timestamp=row.get('timestamp', datetime.now().isoformat()),
                    dataset_size=int(row.get('dataset_size', 0))
                ))
        return results


class DecisionEngine:
    """Compares results against thresholds and makes deployment decisions"""

    def __init__(self, config: ThresholdConfig):
        self.config = config
        self.decisions: list[dict] = []

    def evaluate(self, results: list[BenchmarkResult]) -> tuple[bool, list[dict]]:
        all_passed = True
        for result in results:
            decision = self._evaluate_single(result)
            self.decisions.append(decision)
            if not decision['passed']:
                all_passed = False
        return all_passed, self.decisions

    def _evaluate_single(self, result: BenchmarkResult) -> dict:
        if result.test_type.lower() == 'handwritten':
            cer_threshold = self.config.handwritten_cer_threshold
            wer_threshold = self.config.handwritten_wer_threshold
        else:
            cer_threshold = self.config.printed_cer_threshold
            wer_threshold = self.config.printed_wer_threshold

        cer_passed = result.cer <= cer_threshold
        wer_passed = result.wer <= wer_threshold
        overall_passed = cer_passed and wer_passed

        return {
            'model_name': result.model_name,
            'test_type': result.test_type,
            'cer': result.cer,
            'cer_threshold': cer_threshold,
            'cer_passed': cer_passed,
            'wer': result.wer,
            'wer_threshold': wer_threshold,
            'wer_passed': wer_passed,
            'passed': overall_passed,
            'action': 'DEPLOY' if overall_passed else 'RETRAIN',
            'timestamp': result.timestamp
        }


class ReportGenerator:
    """Generates Markdown reports from decisions"""

    @staticmethod
    def generate(decisions: list[dict], all_passed: bool, ab_result: dict | None = None) -> str:
        status_emoji = "✅" if all_passed else "❌"
        status_text = "PASSED - Ready for Deploy" if all_passed else "FAILED - Retrain Required"

        report = "# Threshold Check Report\n\n"
        report += "## 📊 Overall Status\n"
        report += f"{status_emoji} **{status_text}**\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"

        if ab_result:
            report += "## ⚖️ A/B Test Result\n\n"
            report += "| Metric | Current Model | New Model | Improvement |\n"
            report += "|--------|--------------|-----------|-------------|\n"
            report += f"| CER | {ab_result['current_cer']:.2%} | {ab_result['new_cer']:.2%} | {ab_result['improvement_percent']:+.1f}% |\n"
            report += f"| Decision | {ab_result['decision']} | | {ab_result['reason']} |\n\n---\n\n"

        report += "## 📈 Detailed Results\n\n"
        report += "| Model | Test Type | CER | Threshold | WER | Threshold | Status | Action |\n"
        report += "|-------|-----------|-----|-----------|-----|-----------|--------|--------|\n"

        for d in decisions:
            cer_status = "✅" if d['cer_passed'] else "❌"
            wer_status = "✅" if d['wer_passed'] else "❌"
            action_emoji = "🚀" if d['action'] == 'DEPLOY' else "🔄"
            report += f"| {d['model_name']} | {d['test_type']} | {d['cer']:.2%} {cer_status} | {d['cer_threshold']:.0%} | {d['wer']:.2%} {wer_status} | {d['wer_threshold']:.0%} | {'✅' if d['passed'] else '❌'} | {action_emoji} {d['action']} |\n"

        total = len(decisions)
        passed = sum(1 for d in decisions if d['passed'])
        failed = total - passed

        report += "\n---\n\n## 📉 Summary\n"
        report += f"- **Total Tests:** {total}\n"
        report += f"- **Passed:** {passed} ({passed/total*100:.1f}%)\n"
        report += f"- **Failed:** {failed} ({failed/total*100:.1f}%)\n\n"

        report += "---\n\n## 🔔 Next Steps\n\n"
        if all_passed:
            report += "✅ **All benchmarks passed the threshold checks.**\n\n"
            report += "**Recommended Actions:**\n"
            report += "1. Merge the model update to production\n"
            report += "2. Deploy model to Hugging Face Model Hub\n"
            report += "3. Update the model version in omni-medical-suite\n"
            report += "4. Send Telegram notification\n"
        else:
            report += "❌ **One or more benchmarks failed the threshold checks.**\n\n"
            report += "**Recommended Actions:**\n"
            report += "1. Trigger retraining pipeline with latest ground truth data\n"
            report += "2. Investigate regression causes (check recent data changes)\n"
            report += "3. Review failed test cases in detail\n"
            report += "4. Hold deployment until retraining completes\n"

        return report


class TelegramNotifier:
    """Sends notifications via Telegram"""

    @staticmethod
    def send_notification(message: str, bot_token: str, chat_id: str):
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        try:
            subprocess.run(
                ['curl', '-s', '-X', 'POST', url,
                 '-H', 'Content-Type: application/json',
                 '-d', json.dumps(payload)],
                check=True,
                capture_output=True,
                timeout=30
            )
            logger.info("Telegram notification sent")
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")


class ModelDeployer:
    """Deploys model to Hugging Face Model Hub"""

    @staticmethod
    def deploy_to_hf(model_path: str, repo_id: str, hf_token: str):
        logger.info(f"Deploying model to {repo_id}...")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True)
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                token=hf_token,
                commit_message=f"Deploy model {datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            logger.info(f"Model deployed successfully to {repo_id}")
            return True
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False


class SignalEmitter:
    """Emits signals for GitHub Actions workflows"""

    @staticmethod
    def emit_deploy_signal():
        signal_file = Path('.deploy_signal')
        signal_file.write_text('DEPLOY_APPROVED')
        logger.info("Deploy signal emitted")
        if 'GITHUB_OUTPUT' in os.environ:
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write('decision=deploy\n')

    @staticmethod
    def emit_retrain_signal():
        signal_file = Path('.retrain_signal')
        signal_file.write_text('RETRAIN_REQUIRED')
        logger.info("Retrain signal emitted")
        if 'GITHUB_OUTPUT' in os.environ:
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write('decision=retrain\n')


def main():
    logger.info("=" * 60)
    logger.info("Starting Threshold Checker v2.0...")
    logger.info("=" * 60)

    # 1. Load configuration
    config_path = Path('config/thresholds.json')
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        config = ThresholdConfig(**config_data)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = ThresholdConfig()
        logger.info("Using default thresholds")

    # 2. Read results
    results_path = Path('data/benchmark_results.json')
    if not results_path.exists():
        results_path = Path('data/benchmark_results.csv')

    if not results_path.exists():
        logger.error("No results file found")
        sys.exit(1)

    logger.info(f"Reading results from {results_path}")

    if results_path.suffix == '.json':
        results = ResultsReader.from_json(str(results_path))
    else:
        results = ResultsReader.from_csv(str(results_path))

    logger.info(f"Loaded {len(results)} benchmark results")

    # 3. Evaluate
    engine = DecisionEngine(config)
    all_passed, decisions = engine.evaluate(results)

    # 4. Try A/B test if new model results exist
    ab_result = None
    try:
        from src.ab_testing.ab_tester import ABTester
        holdout_path = Path('data/holdout/holdout_dataset.json')
        if holdout_path.exists():
            tester = ABTester(str(holdout_path))
            current_cer = 0.048  # baseline from last known good model
            new_results = [d for d in decisions if d['test_type'] == 'handwritten']
            if new_results:
                new_cer = new_results[0]['cer']
                ab_result = tester.run_ab_test(current_cer, new_cer)
                logger.info(f"A/B Test: improvement={ab_result['improvement_percent']:.1f}%, decision={ab_result['decision']}")
                if ab_result['decision'] == 'ROLLBACK':
                    all_passed = False
                    logger.info("A/B test failed - new model is worse than current")
    except ImportError:
        logger.info("A/B tester not available, skipping comparison")
    except Exception as e:
        logger.warning(f"A/B test could not run: {e}")

    # 5. Generate report
    report = ReportGenerator.generate(decisions, all_passed, ab_result)
    report_path = Path('threshold_report.md')
    report_path.write_text(report, encoding='utf-8')
    logger.info(f"Report saved to {report_path}")

    # 6. Send Telegram notification
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat = os.getenv('TELEGRAM_CHAT_ID')

    if telegram_token and telegram_chat:
        if all_passed:
            message = "✅ **[Omni-Medical-Suite]** Threshold Check PASSED!\n\nModel is ready for deployment to Hugging Face."
            if ab_result:
                message += f"\n\nA/B Test: {ab_result['improvement_percent']:+.1f}% improvement"
        else:
            message = "❌ **[Omni-Medical-Suite]** Threshold Check FAILED!\n\nAuto-triggering retraining pipeline."
            if ab_result:
                message += f"\n\nReason: {ab_result['reason']}"
        TelegramNotifier.send_notification(message, telegram_token, telegram_chat)

    # 7. Take action
    if all_passed:
        model_path = os.getenv('MODEL_PATH', './model')
        hf_repo = os.getenv('HF_MODEL_REPO', 'DrAbdulmalek/medical-ocr-model')
        hf_token = os.getenv('HF_TOKEN')

        if hf_token and Path(model_path).exists():
            ModelDeployer.deploy_to_hf(model_path, hf_repo, hf_token)

        SignalEmitter.emit_deploy_signal()
        logger.info("DECISION: DEPLOY")
        sys.exit(0)
    else:
        SignalEmitter.emit_retrain_signal()
        logger.info("DECISION: RETRAIN")
        sys.exit(1)


if __name__ == '__main__':
    main()
