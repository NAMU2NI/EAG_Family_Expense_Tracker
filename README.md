# Expense Tracker

A personal finance dashboard built with Gradio that parses bank statements, auto-categorizes transactions, and visualizes spending with budget tracking.

## Features

- **Import bank statements** — Upload CSV or PDF files; supports multi-bank tracking
- **Auto-categorization** — 3-layer system: AI cache → merchant keyword cache → rule-based keyword lists (15+ categories)
- **AI categorization** — Uses GPT-4o-mini to resolve uncategorized ("Others") transactions in batches
- **Manual overrides** — Click any transaction to reassign its category; the app learns the mapping for future imports
- **Analytics dashboard** — Category/payment-type pie charts, budget vs. actual bar chart, KPI summary bar
- **Insight engine** — Automatic spending insights and recommendations based on budget performance

## Project Structure

```
expense-tracker/
├── app.py                  # Main application (Gradio UI + all logic)
├── requirements.txt        # Python dependencies
├── data/
│   ├── transactions.json   # Persisted transaction records
│   ├── categories.json     # Category keyword lists (editable)
│   ├── ai_cache.json       # GPT categorization cache (description → category)
│   └── merchant_cache.json # User-taught keyword → category mappings
└── Input/                  # Drop bank statement files here before importing
```

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure OpenAI key** (optional — needed for AI categorization)

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

**3. Run**

```bash
python app.py
```

The app opens at `http://localhost:7860`.

## Usage

### Import Transactions
1. Go to the **Import Transactions** tab
2. Upload a CSV or PDF bank statement
3. Enter the bank name (e.g., `HDFC`, `Axis`, `ICICI`)
4. Click **Import** — transactions are parsed and auto-categorized

### Review & Fix Categories
1. Open the **Transactions** tab
2. Use filters (category, payment type, bank, debit/credit) to narrow the list
3. Click a row to select it, pick the correct category from the dropdown, and click **Save**
4. Click **AI Categorize** to batch-resolve all remaining "Others" with GPT-4o-mini

### Analytics
- Switch to the **Analytics** tab for charts, budget comparisons, and auto-generated insights
- Use the bank filter to view spending across a specific bank or all banks combined

## Categorization Pipeline

| Priority | Source | Description |
|----------|--------|-------------|
| 1 | AI cache | Exact-match lookup from prior GPT calls |
| 2 | Merchant cache | User-taught keyword → category mappings |
| 3 | Keyword rules | Built-in lists per category (groceries, dining, transport, etc.) |
| — | Fallback | Assigned "Others" for manual or AI review |

## Default Categories & Budgets

| Category | Monthly Budget (₹) |
|---|---|
| Housing & Rent | 35,000 |
| Investments | 10,000 |
| Dining & Restaurants | 6,000 |
| Shopping | 5,000 |
| Transport & Fuel | 7,000 |
| Utilities | 4,000 |
| Healthcare & Medical | 3,000 |
| Entertainment & Sports | 3,000 |
| Insurance | 3,000 |
| Personal Care | 2,000 |
| Education | 2,000 |
| Groceries | 1,200 |
| Others | 1,000 |

Budgets and keyword lists are persisted in `data/categories.json` and can be edited directly.

## Dependencies

| Package | Purpose |
|---|---|
| `gradio` | Web UI |
| `pandas` | Data manipulation |
| `pdfplumber` | PDF statement parsing |
| `plotly` | Interactive charts |
| `openpyxl` | Excel/XLSX support |
| `openai` | GPT-4o-mini categorization |
| `python-dotenv` | `.env` key loading |
