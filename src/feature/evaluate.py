import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from datetime import datetime
from src.feature.classifier import classify_email


def load_previous_run(history_path):
    """Loads the most recent evaluation run from history."""
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
            if history:
                return history[-1]
    return None


def save_current_run(history_path, run_data):
    """Appends the current run to the history file."""
    history = []
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

    history.append(run_data)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def run_evaluation():
    print("Starting Evaluation Pipeline with Diff Engine...\n")

    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / "data" / "golden_dataset.json"
    history_path = project_root / "data" / "run_history.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    passed = 0
    total = len(test_cases)
    current_results = {}

    # 1. Evaluate current state
    for case in test_cases:
        result = classify_email(case['email_text'], prompt_version="v2")
        is_pass = (result.category.value == case['expected_category'])

        current_results[case['id']] = {
            "expected": case['expected_category'],
            "actual": result.category.value,
            "passed": is_pass
        }

        if is_pass:
            passed += 1

    accuracy = (passed / total) * 100

    # 2. DIFF ENGINE LOGIC: Compare with previous run
    previous_run = load_previous_run(history_path)

    regressions = []
    improvements = []
    delta_str = "N/A (First Run)"

    if previous_run:
        prev_accuracy = previous_run["accuracy"]
        delta = accuracy - prev_accuracy
        delta_str = f"{delta:+.1f}%"

        prev_results = previous_run["results"]
        for case_id, current in current_results.items():
            if case_id in prev_results:
                prev_passed = prev_results[case_id]["passed"]
                curr_passed = current["passed"]

                if prev_passed and not curr_passed:
                    regressions.append(case_id)
                elif not prev_passed and curr_passed:
                    improvements.append(case_id)

    # 3. Save current state to history
    run_data = {
        "timestamp": datetime.now().isoformat(),
        "prompt_version": "v1.0.0",
        "total_cases": total,
        "passed": passed,
        "accuracy": accuracy,
        "results": current_results
    }
    save_current_run(history_path, run_data)

    # 4. Print Diff Report
    print("=" * 45)
    print("📋 EVALUATION REPORT & DIFF")
    print(f"Prompt Version: v2.0.0")
    print(f"Accuracy:       {accuracy:.1f}%")
    print(f"Delta vs Last:  {delta_str}")
    print("-" * 45)

    if previous_run:
        if regressions:
            print(f"🚨 REGRESSIONS ({len(regressions)}): {', '.join(regressions)}")
        else:
            print("✅ No regressions detected.")

        if improvements:
            print(f"🚀 IMPROVEMENTS ({len(improvements)}): {', '.join(improvements)}")
    else:
        print("ℹ️  Baseline established. Future runs will show regressions.")
    print("=" * 45)


if __name__ == "__main__":
    run_evaluation()