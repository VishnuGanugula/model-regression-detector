import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from datetime import datetime


def load_data():
    project_root = Path(__file__).resolve().parents[2]
    history_path = project_root / "data" / "run_history.json"
    dataset_path = project_root / "data" / "golden_dataset.json"

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = {item["id"]: item for item in json.load(f)}

    return history, dataset, project_root


def generate_html():
    print("Generating HTML Diff Report...")
    history, dataset, project_root = load_data()

    if not history:
        print("No history found. Run evaluate.py first.")
        return

    current_run = history[-1]
    prev_run = history[-2] if len(history) > 1 else None

    # Calculate Delta
    delta_str = "N/A"
    delta_color = "black"
    if prev_run:
        delta = current_run["accuracy"] - prev_run["accuracy"]
        delta_str = f"{delta:+.1f}%"
        if delta > 0:
            delta_color = "green"
        elif delta < 0:
            delta_color = "red"

    # Start HTML template
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM Regression Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; background-color: #f4f4f9; color: #333; }}
            h1 {{ border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
            .summary-container {{ display: flex; gap: 20px; margin-bottom: 30px; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 150px; text-align: center; }}
            .card h3 {{ margin: 0 0 10px 0; color: #666; font-size: 14px; text-transform: uppercase; }}
            .card p {{ margin: 0; font-size: 24px; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
            th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; font-weight: 600; }}
            .status-pass {{ color: #155724; background-color: #d4edda; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
            .status-fail {{ color: #721c24; background-color: #f8d7da; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
            .diff-improve {{ color: #155724; font-weight: bold; }}
            .diff-regress {{ color: #721c24; font-weight: bold; }}
            .diff-none {{ color: #6c757d; }}
        </style>
    </head>
    <body>
        <h1>LLM Evaluation & Regression Report</h1>

        <div class="summary-container">
            <div class="card">
                <h3>Prompt Version</h3>
                <p>{current_run['prompt_version']}</p>
            </div>
            <div class="card">
                <h3>Accuracy</h3>
                <p>{current_run['accuracy']:.1f}%</p>
            </div>
            <div class="card">
                <h3>Delta (vs Last)</h3>
                <p style="color: {delta_color}">{delta_str}</p>
            </div>
            <div class="card">
                <h3>Total Cases</h3>
                <p>{current_run['total_cases']}</p>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Input Text</th>
                    <th>Expected</th>
                    <th>Actual</th>
                    <th>Status</th>
                    <th>Diff (vs Last)</th>
                </tr>
            </thead>
            <tbody>
    """

    # Populate Table Rows
    curr_results = current_run["results"]
    prev_results = prev_run["results"] if prev_run else {}

    for case_id, result in curr_results.items():
        original_data = dataset.get(case_id, {})
        email_text = original_data.get("email_text", "N/A")
        if len(email_text) > 60:
            email_text = email_text[:57] + "..."

        is_pass = result["passed"]
        status_html = '<span class="status-pass">PASS</span>' if is_pass else '<span class="status-fail">FAIL</span>'

        # Diff Logic
        diff_html = '<span class="diff-none">-</span>'
        if case_id in prev_results:
            prev_pass = prev_results[case_id]["passed"]
            if prev_pass and not is_pass:
                diff_html = '<span class="diff-regress">🚨 Regression</span>'
            elif not prev_pass and is_pass:
                diff_html = '<span class="diff-improve">🚀 Improved</span>'

        html += f"""
                <tr>
                    <td>{case_id}</td>
                    <td>{email_text}</td>
                    <td>{result['expected']}</td>
                    <td>{result['actual']}</td>
                    <td>{status_html}</td>
                    <td>{diff_html}</td>
                </tr>
        """

    # Close HTML
    html += """
            </tbody>
        </table>
    </body>
    </html>
    """

    report_path = project_root / "data" / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report generated successfully: {report_path}")


if __name__ == "__main__":
    generate_html()