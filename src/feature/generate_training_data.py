import csv
import json
from pathlib import Path


def convert_kaggle_csv():
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "customer_support_tickets.csv"
    output_path = project_root / "data" / "training_data.jsonl"

    if not csv_path.exists():
        print(f"❌ Please place your Kaggle CSV file at: {csv_path}")
        return

    print("Processing Kaggle dataset...")
    count = 0

    with open(csv_path, mode="r", encoding="utf-8") as f_in, \
            open(output_path, mode="w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        for row in reader:
            # Adjust these keys based on the columns visible in your dataset viewer
            text = row.get("Ticket Description") or row.get("Ticket Subject") or row.get("text")
            ticket_type = (row.get("Ticket Type") or row.get("Category") or "").lower()

            # Map raw labels to your core categories
            category = "general"
            if "bill" in ticket_type or "charge" in ticket_type or "payment" in ticket_type:
                category = "billing"
            elif "tech" in ticket_type or "bug" in ticket_type or "software" in ticket_type:
                category = "technical"
            elif "account" in ticket_type or "password" in ticket_type or "login" in ticket_type:
                category = "account"

            if text:
                training_item = {
                    "email": text.strip(),
                    "category": category
                }
                f_out.write(json.dumps(training_item) + "\n")
                count += 1

                # Limit to 1,000 samples for efficient local training
                if count >= 1000:
                    break

    print(f"✅ Successfully converted {count} rows into {output_path}")


if __name__ == "__main__":
    convert_kaggle_csv()