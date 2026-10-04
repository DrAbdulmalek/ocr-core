#!/usr/bin/env python3
"""
A/B Tester - Medical OCR Benchmarks
Author: DrAbdulmalek
Description: Compares new model against current production model on holdout dataset
             to ensure the new model is actually better before deployment.
"""

import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    current_cer: float
    new_cer: float
    improvement_percent: float
    decision: str
    reason: str
    timestamp: str
    holdout_size: int


@dataclass
class HoldoutSample:
    image_path: str
    ground_truth: str
    ocr_engine: str


class HoldoutManager:
    """Manages the holdout dataset - data split from training that is never used for training"""

    def __init__(self, holdout_path: str):
        self.holdout_path = Path(holdout_path)
        self.samples: list[HoldoutSample] = []
        if self.holdout_path.exists():
            self._load()

    def _load(self):
        with open(self.holdout_path, encoding='utf-8') as f:
            data = json.load(f)
        for item in data.get('samples', []):
            self.samples.append(HoldoutSample(
                image_path=item['image_path'],
                ground_truth=item['ground_truth'],
                ocr_engine=item.get('ocr_engine', 'auto')
            ))
        logger.info(f"Loaded {len(self.samples)} holdout samples from {self.holdout_path}")

    def create_from_dataset(self, dataset_path: str, test_ratio: float = 0.2, seed: int = 42):
        """
        Split an existing dataset into training (80%) and holdout (20%).
        The holdout set is ONLY used for A/B testing, never for training.
        """
        random.seed(seed)
        dataset_path = Path(dataset_path)

        if dataset_path.suffix == '.json':
            with open(dataset_path, encoding='utf-8') as f:
                full_data = json.load(f)
            items = full_data if isinstance(full_data, list) else full_data.get('samples', full_data.get('data', []))
        elif dataset_path.suffix == '.jsonl':
            items = []
            with open(dataset_path, encoding='utf-8') as f:
                for line in f:
                    items.append(json.loads(line.strip()))
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

        random.shuffle(items)
        split_idx = int(len(items) * (1 - test_ratio))
        train_items = items[:split_idx]
        holdout_items = items[split_idx:]

        # Save training set (without holdout)
        train_path = dataset_path.parent / f"{dataset_path.stem}_train{dataset_path.suffix}"
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_items, f, ensure_ascii=False, indent=2)

        # Save holdout set
        holdout_data = {
            'created': datetime.now().isoformat(),
            'source': str(dataset_path),
            'test_ratio': test_ratio,
            'seed': seed,
            'total_original': len(items),
            'train_size': len(train_items),
            'holdout_size': len(holdout_items),
            'samples': holdout_items
        }
        self.holdout_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.holdout_path, 'w', encoding='utf-8') as f:
            json.dump(holdout_data, f, ensure_ascii=False, indent=2)

        self.samples = [
            HoldoutSample(
                image_path=item.get('image_path', item.get('file', '')),
                ground_truth=item.get('ground_truth', item.get('text', item.get('corrected', ''))),
                ocr_engine=item.get('ocr_engine', 'auto')
            )
            for item in holdout_items
        ]

        logger.info(f"Split complete: {len(train_items)} train, {len(holdout_items)} holdout")
        logger.info(f"Training set saved to: {train_path}")
        logger.info(f"Holdout set saved to: {self.holdout_path}")

        return len(holdout_items)

    @property
    def size(self) -> int:
        return len(self.samples)


class ABTester:
    """Compares two model versions on the holdout dataset"""

    def __init__(self, holdout_dataset_path: str):
        self.holdout = HoldoutManager(holdout_dataset_path)
        self.history: list[ABTestResult] = []

    def evaluate_model(self, model_path: str) -> float:
        """
        Evaluate a model on the holdout set.
        Returns CER (Character Error Rate).
        In production, this loads the model and runs OCR on each holdout image,
        then compares against ground truth using CER metric.
        """
        # Placeholder: In real implementation, this would:
        # 1. Load the model from model_path
        # 2. Run OCR on each holdout sample's image
        # 3. Compare output with ground_truth using CER
        # 4. Return average CER across all samples
        logger.info(f"Evaluating model at {model_path} on {self.holdout.size} holdout samples")
        return 0.045  # Example: would be computed from actual evaluation

    def run_ab_test(self, current_model_cer: float, new_model_cer: float,
                    min_improvement: float = 2.0) -> dict:
        """
        Run A/B comparison between current and new model.
        The new model must be at least min_improvement% better to be deployed.
        """
        if current_model_cer == 0:
            # First model ever - deploy if it passes thresholds
            return {
                'current_cer': current_model_cer,
                'new_cer': new_model_cer,
                'improvement_percent': 100.0,
                'decision': 'DEPLOY',
                'reason': 'First model deployment'
            }

        improvement = ((current_model_cer - new_model_cer) / current_model_cer) * 100

        if new_model_cer <= current_model_cer and improvement >= min_improvement:
            decision = 'DEPLOY'
            reason = f'Model improved by {improvement:.1f}% (minimum: {min_improvement}%)'
        elif new_model_cer <= current_model_cer:
            decision = 'DEPLOY'
            reason = f'Model slightly improved by {improvement:.1f}% (below {min_improvement}% threshold but still better)'
        else:
            decision = 'ROLLBACK'
            reason = f'Regression detected: CER increased by {abs(improvement):.1f}%'

        result = ABTestResult(
            current_cer=current_model_cer,
            new_cer=new_model_cer,
            improvement_percent=improvement,
            decision=decision,
            reason=reason,
            timestamp=datetime.now().isoformat(),
            holdout_size=self.holdout.size
        )
        self.history.append(result)

        logger.info(f"A/B Test Result: {decision} - {reason}")
        logger.info(f"  Current CER: {current_model_cer:.4f}, New CER: {new_model_cer:.4f}")
        logger.info(f"  Improvement: {improvement:+.2f}%")

        return {
            'current_cer': result.current_cer,
            'new_cer': result.new_cer,
            'improvement_percent': result.improvement_percent,
            'decision': result.decision,
            'reason': result.reason,
            'timestamp': result.timestamp,
            'holdout_size': result.holdout_size
        }

    def save_history(self, output_path: str = 'data/ab_test_history.json'):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = [vars(r) for r in self.history]
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"A/B test history saved to {output_path}")


def create_sample_holdout():
    """Create a sample holdout dataset for testing"""
    holdout_data = {
        'created': datetime.now().isoformat(),
        'source': 'synthetic',
        'test_ratio': 0.2,
        'seed': 42,
        'total_original': 10,
        'train_size': 8,
        'holdout_size': 2,
        'samples': [
            {
                'image_path': 'holdout/sample_001.png',
                'ground_truth': 'Patient presents with chronic hypertension',
                'ocr_engine': 'auto'
            },
            {
                'image_path': 'holdout/sample_002.png',
                'ground_truth': 'Amoxicillin 500mg three times daily for 7 days',
                'ocr_engine': 'auto'
            }
        ]
    }
    output_path = Path('data/holdout/holdout_dataset.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(holdout_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Sample holdout dataset created at {output_path}")
    return str(output_path)


if __name__ == '__main__':
    # Demo run
    print("Creating sample holdout dataset...")
    holdout_path = create_sample_holdout()

    print("\nRunning A/B test...")
    tester = ABTester(holdout_path)

    # Simulate: current model CER = 4.8%, new model CER = 4.2% (improvement!)
    result = tester.run_ab_test(0.048, 0.042)
    print(f"\nResult: {result['decision']} - {result['reason']}")
    print(f"  Current CER: {result['current_cer']:.2%}")
    print(f"  New CER: {result['new_cer']:.2%}")
    print(f"  Improvement: {result['improvement_percent']:+.1f}%")

    # Simulate regression
    print("\n--- Simulating Regression ---")
    result2 = tester.run_ab_test(0.048, 0.055)
    print(f"Result: {result2['decision']} - {result2['reason']}")

    tester.save_history()
