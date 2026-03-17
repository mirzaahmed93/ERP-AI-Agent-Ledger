"""AgentLedger – Synthetic mock-data generator for ERP invoices & bank transactions.

Generates four payment scenarios:
  • perfect  – exact 1:1 match
  • short_pay – wire-fee deduction ($15/$20/$25)
  • typo     – mangled invoice-ID prefix
  • bulk     – single deposit covering 3 invoices
"""

import pandas as pd
from faker import Faker
import random
from datetime import timedelta

fake = Faker()
Faker.seed(42)  # Seed for reproducibility
random.seed(42)


def generate_mock_data(num_companies=50):
    erp_invoices = []
    bank_transactions = []

    transaction_id_counter = 10000

    for _ in range(num_companies):
        company_name = fake.company()
        payment_scenario = random.choice(["perfect", "short_pay", "typo", "bulk"])

        base_date = fake.date_between(start_date="-30d", end_date="today")

        if payment_scenario == "bulk":
            invoices = []
            total_amount = 0
            for i in range(3):
                inv_id = f"INV-{fake.unique.random_int(min=1000, max=9999)}"
                amt = round(random.uniform(500.0, 2000.0), 2)
                invoices.append(inv_id)
                total_amount += amt

                erp_invoices.append({
                    "invoice_id": inv_id,
                    "customer_name": company_name,
                    "amount_due": amt,
                    "due_date": base_date + timedelta(days=30),
                    "status": "OPEN",
                })

            bank_transactions.append({
                "txn_id": f"TXN-{transaction_id_counter}",
                "date": base_date + timedelta(days=random.randint(5, 25)),
                "description": f"WIRE TRANSFER {company_name} REF {invoices[0]} AND OTHERS",
                "amount": round(total_amount, 2),
                "type": "CREDIT",
            })

        else:
            inv_id = f"INV-{fake.unique.random_int(min=1000, max=9999)}"
            amt = round(random.uniform(1000.0, 5000.0), 2)

            erp_invoices.append({
                "invoice_id": inv_id,
                "customer_name": company_name,
                "amount_due": amt,
                "due_date": base_date + timedelta(days=30),
                "status": "OPEN",
            })

            payment_date = base_date + timedelta(days=random.randint(5, 25))

            if payment_scenario == "perfect":
                bank_transactions.append({
                    "txn_id": f"TXN-{transaction_id_counter}",
                    "date": payment_date,
                    "description": f"ACH PAYMENT {company_name} {inv_id}",
                    "amount": amt,
                    "type": "CREDIT",
                })

            elif payment_scenario == "short_pay":
                fee = random.choice([15.00, 20.00, 25.00])
                bank_transactions.append({
                    "txn_id": f"TXN-{transaction_id_counter}",
                    "date": payment_date,
                    "description": f"WIRE IN {company_name} REF: {inv_id}",
                    "amount": amt - fee,
                    "type": "CREDIT",
                })

            elif payment_scenario == "typo":
                typo_inv = inv_id.replace("INV-", "IN-")
                bank_transactions.append({
                    "txn_id": f"TXN-{transaction_id_counter}",
                    "date": payment_date,
                    "description": f"PAYMENT FROM {company_name} FOR {typo_inv}",
                    "amount": amt,
                    "type": "CREDIT",
                })

        transaction_id_counter += 1

    df_erp = pd.DataFrame(erp_invoices)
    df_bank = pd.DataFrame(bank_transactions)

    df_erp.to_csv("data/erp_accounts_receivable.csv", index=False)
    df_bank.to_csv("data/bank_statement.csv", index=False)
    print(
        f"✅ Successfully generated {len(df_erp)} ERP invoices "
        f"and {len(df_bank)} Bank Transactions."
    )


if __name__ == "__main__":
    generate_mock_data(num_companies=50)
