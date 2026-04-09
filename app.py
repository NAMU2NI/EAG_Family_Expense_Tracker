import re
import os
import uuid
import json

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import pandas as pd
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_FILE           = "data/transactions.json"
CATEGORIES_FILE     = "data/categories.json"
AI_CACHE_FILE       = "data/ai_cache.json"
MERCHANT_CACHE_FILE = "data/merchant_cache.json"

_ai_cache: dict | None       = None
_merchant_cache: dict | None = None

# OpenAI key loaded from .env  (OPENAI_API_KEY=sk-...)
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# Regex helpers
_DATE_RE = re.compile(
    r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}'
    r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b',
    re.IGNORECASE,
)
_AMT_RE = re.compile(r'[\d,]+\.\d{2}')

DEFAULT_CATEGORIES = {
    "Groceries": [
        "grocery", "supermarket", "bigbasket", "dmart", "reliance fresh",
        "nature basket", "spar", "blinkit", "zepto", "instamart", "grofers",
        "food bazaar", "walmart", "costco", "more retail", "lulu", "smart bazaar",
        "heritage fresh", "nilgiris", "metro cash",
        # common UPI purpose words for groceries
        "curd", "milk", "butter", "ghee", "paneer", "vegetables", "veggies",
        "fruits", "guava", "banana", "provision", "kirana", "dairy",
    ],
    "Dining & Restaurants": [
        "restaurant", "cafe", "coffee", "swiggy", "zomato", "dominos",
        "mcdonald", "kfc", "subway", "pizza", "burger", "dining", "eatery",
        "dhaba", "biryani", "chaayos", "starbucks", "snack", "food kart",
        "foodkart", "canteen", "meals", "lunch", "dinner", "breakfast",
        "bakery", "juice", "hotel food", "tiffin", "mess", "barbeque",
        "barbeque nation", "the cake shop", "haldiram", "amul",
        # common UPI purpose words for food
        "tea", "chai", "vada", "idli", "dosa", "parotta", "puri",
        "cookies", "bites", "brewery", "drinks", "butter milk", "buttermilk",
        "cfe", "sweets", "mithai", "lassi", "shake", "ice cream",
    ],
    "Transport & Fuel": [
        "petrol", "diesel", "fuel", "ola", "uber", "rapido", "auto", "taxi",
        "metro", "bus", "irctc", "train", "flight", "airline", "toll",
        "parking", "fastag", "redbus", "makemytrip", "cars24", "car service",
        "car wash", "vehicle", "rto", "two wheeler", "bike service",
        "hp petrol", "iocl", "bpcl", "indian oil", "bharat petroleum",
        "bmtc", "ksrtc", "cab", "rickshaw",
    ],
    "Healthcare & Medical": [
        "hospital", "clinic", "pharmacy", "medicine", "doctor", "medical",
        "apollo", "medplus", "diagnostic", "lab test", "netmeds",
        "1mg", "pharmeasy", "dental", "eye care", "optical", "pathlab",
        "thyrocare", "lal path", "metropolis",
        # common UPI purpose words for health
        "physio", "physiotherapy", "ors", "consultation", "checkup",
        "dispensary", "nursing",
    ],
    "Education": [
        "school", "college", "university", "tuition", "coaching", "course",
        "books", "stationery", "fees", "exam", "udemy", "coursera",
        "byju", "unacademy", "vedantu", "whitehat", "skill"
    ],
    "Entertainment & Sports": [
        "movie", "cinema", "netflix", "amazon prime", "hotstar", "disney",
        "spotify", "gaming", "pvr", "inox", "concert", "event", "youtube",
        "badminton", "badmiton", "cricket", "tennis", "football", "gym",
        "fitness", "sports", "swimming", "yoga", "zumba", "hobby",
        "playzone", "fun world", "amusement", "bowling", "chess", "club",
        # common UPI purpose words
        "tournament", "match", "subscription", "google play", "playstore",
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "ajio", "nykaa", "meesho",
        "shoppers stop", "lifestyle", "zara", "h&m", "clothing", "apparel",
        "decathlon", "ikea", "pepperfry", "urban ladder", "croma", "reliance digital",
        "vijay sales", "boat", "samsung", "apple store"
    ],
    "Utilities": [
        "electricity", "water", "gas", "internet", "broadband", "recharge",
        "jio", "vodafone", "bsnl", "postpaid", "prepaid", "dth",
        "cable tv", "bbmp", "bescom", "tneb", "msedcl", "tata power",
        "adani electric", "mahanagar gas", "indane", "bharat gas",
        # common UPI purpose words
        "ebill", "bill pay", "mobile recharge", "data pack",
    ],
    "Housing & Rent": [
        "rent", "maintenance", "society", "housing", "apartment", "emi",
        "home loan", "property tax", "nobroker", "magicbricks", "99acres",
        "housing.com", "flat", "pg ", "hostel fee"
    ],
    "Insurance": [
        "insurance", "lic", "premium", "policy", "star healt", "star health",
        "hdfc life", "icici pru", "bajaj allianz", "max life", "sbi life",
        "term plan", "health cover", "mediclaim", "niva bupa", "care health"
    ],
    "Personal Care": [
        "salon", "spa", "beauty", "grooming", "haircut", "parlour",
        "lakme", "urban company", "naturals", "jawed habib", "vlcc",
        "mamaearth", "wow skin", "himalaya"
    ],
    "Investments": [
        "mutual fund", "sip", "fixed deposit", "ppf", "nps", "stocks",
        "zerodha", "groww", "upstox", "kuvera", "grip", "wint wealth",
        "jar app", "coin", "smallcase", "ipo", "demat", "trading",
        "nps contribution", "provident fund", "eba/nps"
    ],
    "Cash": [
        "atm withdrawal", "atm/", "/atm", "cash withdrawal", "cash deposit",
        "cash at", "cdm", "cash dispenser"
    ],
    "Bank Transfers": [
        "neft", "rtgs", "imps", "bil/neft", "bil/rtgs", "bil/inft", "mmt/imps",
        "fund transfer", "self transfer", "outward neft", "inward neft"
    ],
    "Farm Land":    [],
    "Others":       [],
}

# Monthly budget per category (₹).  0 = not budgeted / excluded from charts.
DEFAULT_BUDGETS = {
    "Housing & Rent":         35000,
    "Groceries":              1200,
    "Dining & Restaurants":    6000,
    "Transport & Fuel":        7000,
    "Utilities":               4000,
    "Healthcare & Medical":    3000,
    "Entertainment & Sports":  3000,
    "Personal Care":           2000,
    "Shopping":                5000,
    "Investments":            10000,
    "Insurance":               3000,
    "Education":               2000,
    "Others":                  1000
}

# ─── Data helpers ─────────────────────────────────────────────────────────────

def ensure_data_dir():
    os.makedirs("data", exist_ok=True)


def load_categories():
    ensure_data_dir()
    if os.path.exists(CATEGORIES_FILE):
        with open(CATEGORIES_FILE, "r") as f:
            saved = json.load(f)
        changed = False
        for cat, default_kws in DEFAULT_CATEGORIES.items():
            if cat not in saved:
                saved[cat] = default_kws
                changed = True
            else:
                existing = set(saved[cat])
                new_kws = [k for k in default_kws if k not in existing]
                if new_kws:
                    saved[cat] = saved[cat] + new_kws
                    changed = True
        if changed:
            save_categories(saved)
        return saved
    cats = DEFAULT_CATEGORIES.copy()
    save_categories(cats)
    return cats


def save_categories(cats):
    ensure_data_dir()
    with open(CATEGORIES_FILE, "w") as f:
        json.dump(cats, f, indent=2)


def load_transactions():
    ensure_data_dir()
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            if data:
                df = pd.DataFrame(data)
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                if "TxnType" not in df.columns:
                    df["TxnType"] = df.apply(
                        lambda r: get_txn_type(r.get("Description", ""), r.get("Category", "Others")), axis=1
                    )
                return df
    return pd.DataFrame(
        columns=["ID", "Date", "Description", "Debit", "Credit", "Type", "Amount", "Category", "TxnType", "Bank"]
    )


def save_transactions(df):
    ensure_data_dir()
    if df is not None and not df.empty:
        rec = df.copy()
        rec["Date"] = rec["Date"].astype(str)
        with open(DATA_FILE, "w") as f:
            json.dump(rec.to_dict("records"), f, indent=2)


# ─── AI cache ─────────────────────────────────────────────────────────────────

def load_ai_cache() -> dict:
    global _ai_cache
    if _ai_cache is None:
        ensure_data_dir()
        if os.path.exists(AI_CACHE_FILE):
            with open(AI_CACHE_FILE, "r", encoding="utf-8") as f:
                _ai_cache = json.load(f)
        else:
            _ai_cache = {}
    return _ai_cache


def save_ai_cache(cache: dict) -> None:
    global _ai_cache
    _ai_cache = cache
    ensure_data_dir()
    with open(AI_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ─── Merchant cache ───────────────────────────────────────────────────────────

def load_merchant_cache() -> dict:
    global _merchant_cache
    if _merchant_cache is None:
        ensure_data_dir()
        if os.path.exists(MERCHANT_CACHE_FILE):
            with open(MERCHANT_CACHE_FILE, "r", encoding="utf-8") as f:
                _merchant_cache = {k.lower(): v for k, v in json.load(f).items()}
        else:
            _merchant_cache = {}
    return _merchant_cache


def save_merchant_cache(cache: dict) -> None:
    global _merchant_cache
    _merchant_cache = cache
    ensure_data_dir()
    with open(MERCHANT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


_MERCHANT_NOISE = {
    "upi", "bank", "hdfc", "icici", "axis", "kotak", "sbi", "yes",
    "indusind", "federal", "airtel", "paytm", "phonepe", "gpay", "google",
    "neft", "rtgs", "imps", "inft", "mmt", "eba", "pos", "bil",
    "payment", "transfer", "received", "money", "fund", "pay", "cash",
    "account", "acct", "txn", "ref", "ibl", "ybl", "okicici", "oksbi",
    "okhdfcbank", "okaxis", "apl", "idfcfirst",
}


def extract_merchant_keyword(description: str) -> str | None:
    """
    Extract the most meaningful single keyword from a transaction description.

    UPI format:  UPI / <Merchant> / <upi-id@bank> / <Purpose> / <Bank> / <RefID>
    Strategy:
      1. For UPI: prefer slot-1 (merchant name); fall back to slot-3 (purpose).
      2. For others: take the first long alpha token that isn't a noise word.
    Returns a lowercase keyword of 4+ chars, or None.
    """
    raw = str(description).strip()
    parts = [p.strip() for p in raw.split("/")]
    prefix = parts[0].upper()

    candidates = []

    if prefix == "UPI" and len(parts) > 1:
        # Slot 1: merchant / person name — strip digits and special chars
        merchant = re.sub(r'[^a-zA-Z\s]', '', parts[1]).strip().lower()
        candidates += [w for w in merchant.split() if len(w) >= 4]
        # Slot 3: purpose / narration (skip if it looks like a UPI ID)
        if len(parts) > 3 and "@" not in parts[3]:
            purpose = re.sub(r'[^a-zA-Z\s]', '', parts[3]).strip().lower()
            candidates += [w for w in purpose.split() if len(w) >= 4]
    else:
        # Non-UPI: scan all alpha tokens across the whole description
        candidates = re.findall(r'[a-zA-Z]{4,}', raw.lower())

    for word in candidates:
        if word not in _MERCHANT_NOISE:
            return word
    return None


def update_merchant_cache(description: str, category: str) -> None:
    """Extract keyword from description and persist keyword→category mapping."""
    keyword = extract_merchant_keyword(description)
    if not keyword:
        return
    cache = load_merchant_cache()
    cache[keyword] = category
    save_merchant_cache(cache)


def ai_categorize_others(df, cats: dict):
    """
    Send all 'Others' transactions to GPT-4o-mini.
    Uses OPENAI_API_KEY from .env.
    Returns (updated_df, status_message).
    """
    api_key = _OPENAI_KEY
    if not api_key or not api_key.strip() or api_key.startswith("sk-your"):
        return df, (
            "⚠️ OpenAI API key not set. Add `OPENAI_API_KEY=sk-...` to your `.env` file "
            "and restart the app."
        )
    if df is None or df.empty:
        return df, "No transactions loaded."

    others_mask = df["Category"] == "Others"
    if not others_mask.any():
        return df, "✅ No uncategorized transactions — everything is already categorized."

    cache = load_ai_cache()
    unique_descs = df.loc[others_mask, "Description"].unique().tolist()
    uncached = [d for d in unique_descs if d not in cache]

    if uncached:
        try:
            from openai import OpenAI as _OpenAI
            client = _OpenAI(api_key=api_key.strip())
            cat_list = "\n".join(f"- {c}" for c in cats if c != "Others")

            BATCH = 40
            for i in range(0, len(uncached), BATCH):
                batch = uncached[i : i + BATCH]
                prompt = (
                    "You are an expense categorizer for an Indian family's bank transactions.\n\n"
                    f"Available categories:\n{cat_list}\n"
                    "- Others      (if truly none of the above fit)\n\n"
                    "Rules:\n"
                    "- UPI format: UPI/PayeeName/UPI-ID/Purpose/Bank/RefID\n"
                    "- Focus on PayeeName and Purpose segments for UPI transactions\n"
                    "- Return ONLY valid JSON: {\"<description>\": \"<category>\", ...}\n"
                    "- No markdown fences, no explanation.\n\n"
                    f"Transactions:\n{json.dumps(batch, ensure_ascii=False)}"
                )
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content.strip()
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', raw, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if not json_match:
                    raise ValueError(f"No JSON in AI response:\n{raw[:500]}")
                cache.update(json.loads(json_match.group(0)))

            save_ai_cache(cache)

        except Exception as exc:
            return df, f"❌ AI error: {exc}"

    updated = df.copy()
    changed = 0
    for idx in df.index[others_mask]:
        desc = updated.at[idx, "Description"]
        new_cat = cache.get(desc, updated.at[idx, "Category"])
        if new_cat != updated.at[idx, "Category"]:
            updated.at[idx, "Category"] = new_cat
            updated.at[idx, "TxnType"] = get_txn_type(desc, new_cat)
            changed += 1

    save_transactions(updated)
    still = int((updated["Category"] == "Others").sum())
    return updated, (
        f"✅ AI categorized **{changed}** transactions. "
        f"{still} still unresolved."
    )


# ─── Categorization ───────────────────────────────────────────────────────────

_NOISE_TOKENS = {
    "axis bank", "hdfc bank", "icici bank", "sbi", "yes bank", "indusind",
    "federal ba", "airtel pay", "paytm", "phonepe", "gpay", "google pay",
    "kotak", "rbl bank", "idfc", "au bank", "upiintent", "upi", "bil",
    "neft", "rtgs", "imps", "inft", "mmt", "eba", "pos", "nan",
}

def _enrich_description(desc: str) -> str:
    """Append the most informative fields from structured bank descriptions.

    For UPI transactions the format is:
      UPI / <merchant_or_person> / <upi_id@bank> / <purpose> / <bank> / <txn_id> / ...
    We extract slot-2 (merchant name) and slot-4 (purpose) directly — these are
    the richest signals for keyword-based categorisation.
    For non-UPI prefixes we fall back to filtering out noise tokens.
    """
    raw = str(desc).strip()
    parts = raw.split("/")
    if len(parts) < 2:
        return raw
    prefix = parts[0].upper()

    if prefix == "UPI":
        extras = []
        # Slot 2: merchant / person name
        merchant = parts[1].strip() if len(parts) > 1 else ""
        if merchant:
            extras.append(merchant)
        # Slot 4: payment purpose / narration
        purpose = parts[3].strip() if len(parts) > 3 else ""
        if purpose and "@" not in purpose:
            extras.append(purpose)
        return (raw + " " + " ".join(extras)).strip() if extras else raw

    # Non-UPI: filter out noise (bank names, UPI IDs, long transaction refs)
    if prefix not in ("BIL", "MMT", "EBA", "POS"):
        return raw
    meaningful = []
    for part in parts[1:]:
        part = part.strip()
        if not part or len(part) < 2:
            continue
        if "@" in part:
            continue
        if re.match(r'^[0-9A-Fa-f]{16,}$', part):
            continue
        if any(noise in part.lower() for noise in _NOISE_TOKENS):
            continue
        meaningful.append(part)
    if meaningful:
        return raw + " " + " ".join(meaningful)
    return raw


_P2P_PURPOSE = {"upi", "payment", "payment fr", "payment to", "send money", "received", "transfer", ""}


def categorize(description, cats):
    if not description:
        return "Others"

    # 1. Full-description AI cache (exact match — highest confidence)
    ai_cache = load_ai_cache()
    if description in ai_cache:
        return ai_cache[description]

    # 2. Merchant keyword cache (user-taught mappings)
    merchant_cache = load_merchant_cache()
    if merchant_cache:
        # First try the extracted keyword
        keyword = extract_merchant_keyword(description)
        if keyword and keyword in merchant_cache:
            return merchant_cache[keyword]
        # Then scan for any stored keyword appearing anywhere in the description
        desc_lower = description.lower()
        for kw, cat in merchant_cache.items():
            if len(kw) >= 4 and kw in desc_lower:
                return cat

    # 3. Rule-based keyword lists
    enriched = _enrich_description(description)
    desc_lower = enriched.lower()
    for cat_name, keywords in cats.items():
        if cat_name == "Others":
            continue
        for kw in keywords:
            kw_l = kw.lower()
            if len(kw_l) <= 4:
                if re.search(r'\b' + re.escape(kw_l) + r'\b', desc_lower):
                    return cat_name
            else:
                if kw_l in desc_lower:
                    return cat_name
    return "Others"


def get_txn_type(description: str, category: str) -> str:
    """Return the payment rail: Cash | UPI | UPI Others | Bank to Bank | Other"""
    cat = str(category)
    raw = str(description).strip()
    if cat == "Cash":
        return "Cash"
    if cat == "Bank Transfers":
        return "Bank to Bank"
    if raw.upper().startswith("UPI/"):
        # Legacy category value still in saved data → honour it directly
        if cat == "UPI Others":
            return "UPI Others"
        # Recognised spending category → identifiable merchant payment
        if cat != "Others":
            return "UPI"
        # Category is "Others" → distinguish P2P from unresolvable UPI
        if "@" in raw:
            parts = raw.split("/")
            purpose = parts[3].strip().lower() if len(parts) > 3 else ""
            if purpose in _P2P_PURPOSE or re.match(r'^[a-z0-9.]+@[a-z]+$', purpose):
                return "UPI"
        return "UPI Others"
    return "Other"


def recategorize_all(df, cats):
    if df is None or df.empty:
        return df
    df = df.copy()
    df["Category"] = df["Description"].apply(lambda d: categorize(d, cats))
    df["TxnType"]  = df.apply(lambda r: get_txn_type(r["Description"], r["Category"]), axis=1)
    return df


# ─── Parsers ──────────────────────────────────────────────────────────────────

def find_col(columns, keywords):
    for col in columns:
        if any(kw in col.lower().strip() for kw in keywords):
            return col
    return None


def clean_numeric(series):
    return (
        pd.to_numeric(
            series.astype(str)
            .str.replace(",", "").str.replace("Dr", "").str.replace("Cr", "").str.strip(),
            errors="coerce",
        ).fillna(0).abs()
    )


def build_result(raw, date_col, desc_col, debit_col, credit_col, amount_col, bank, cats):
    result = pd.DataFrame()
    result["Date"]        = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)
    result["Description"] = raw[desc_col].fillna("").astype(str).str.strip()

    if debit_col and credit_col:
        result["Debit"]  = clean_numeric(raw[debit_col])
        result["Credit"] = clean_numeric(raw[credit_col])
        result["Type"]   = result.apply(
            lambda r: "Credit" if r["Credit"] > 0 and r["Debit"] == 0 else "Debit", axis=1
        )
        result["Amount"] = result.apply(
            lambda r: r["Credit"] if r["Type"] == "Credit" else r["Debit"], axis=1
        )
    elif amount_col:
        raw_amt          = pd.to_numeric(
            raw[amount_col].astype(str).str.replace(",", "").str.strip(), errors="coerce"
        ).fillna(0)
        result["Amount"] = raw_amt.abs()
        result["Type"]   = raw_amt.apply(lambda x: "Credit" if x > 0 else "Debit")
        result["Debit"]  = result.apply(lambda r: r["Amount"] if r["Type"] == "Debit"   else 0, axis=1)
        result["Credit"] = result.apply(lambda r: r["Amount"] if r["Type"] == "Credit"  else 0, axis=1)
    else:
        return None, "Cannot find Amount / Debit / Credit columns."

    result["Bank"]     = bank
    result["Category"] = result["Description"].apply(lambda d: categorize(d, cats))
    result["TxnType"]  = result.apply(lambda r: get_txn_type(r["Description"], r["Category"]), axis=1)
    result["ID"]       = [str(uuid.uuid4())[:8] for _ in range(len(result))]
    result = result.dropna(subset=["Date"])
    result = result[result["Amount"] > 0]
    return result, None


_CSV_HEADER_KW = ["narration", "description", "particulars", "amount",
                  "debit", "credit", "withdrawal", "deposit", "balance",
                  "remarks", "txn", "transaction", "dr", "cr", "mode"]


def _find_header_row(raw_lines: list[str]) -> int:
    """Return the index of the row that looks like a transaction column-header row.

    A valid header MUST contain 'date' as one of its cells (every transaction
    table has a date column), plus at least 2 other recognised financial keywords.
    """
    for i, line in enumerate(raw_lines):
        cells = [c.strip().lower() for c in line.split(",")]
        # Must have a date column — every transaction table does
        has_date = any(c == "date" or c == "value date" or c == "txn date" for c in cells)
        if not has_date:
            continue
        hits      = sum(1 for c in cells if any(k in c for k in _CSV_HEADER_KW))
        non_empty = sum(1 for c in cells if c)
        if hits >= 2 and non_empty >= 3:
            return i
    return 0          # fall back to first row


_BANK_SIGNATURES = [
    # (keyword_to_search, display_name)
    ("hdfc",            "HDFC"),
    ("icici",           "ICICI"),
    ("axis bank",       "Axis"),
    ("axisbank",        "Axis"),
    ("kotak",           "Kotak"),
    ("state bank",      "SBI"),
    (" sbi ",           "SBI"),
    ("indusind",        "IndusInd"),
    ("yes bank",        "Yes Bank"),
    ("federal bank",    "Federal"),
    ("idfc",            "IDFC"),
    ("au small",        "AU Bank"),
    ("rbl bank",        "RBL"),
    ("bank of baroda",  "BOB"),
    ("canara bank",     "Canara"),
    ("punjab national", "PNB"),
    ("union bank",      "Union Bank"),
    ("paytm",           "Paytm"),
    ("airtel",          "Airtel"),
]


def detect_bank(file_path: str) -> str:
    """
    Try to infer the bank/account name from the file.
    Checks (in order): filename, first 30 lines of CSV, first-page PDF text.
    Returns a human-friendly bank name, or empty string if not detected.
    """
    name = os.path.basename(file_path).lower()
    ext  = os.path.splitext(file_path)[1].lower()

    def _match(text: str) -> str:
        t = text.lower()
        for sig, label in _BANK_SIGNATURES:
            if sig in t:
                return label
        return ""

    # 1. Filename
    found = _match(name)
    if found:
        return found

    # 2. Pre-header rows of CSV (first 30 lines)
    if ext == ".csv":
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(file_path, encoding=enc, errors="replace") as f:
                    head = "".join(f.readline() for _ in range(30))
                found = _match(head)
                if found:
                    return found
                break
            except Exception:
                continue

    # 3. First-page text of PDF
    if ext == ".pdf":
        try:
            with pdfplumber.open(file_path) as pdf:
                if pdf.pages:
                    text = pdf.pages[0].extract_text() or ""
                    found = _match(text)
                    if found:
                        return found
        except Exception:
            pass

    return ""


def parse_csv(file_path, bank, cats):
    # Read raw lines first so we can find the real header row
    raw = None
    enc_used = "utf-8"
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(file_path, encoding=enc, errors="replace") as f:
                raw = f.readlines()
            enc_used = enc
            break
        except Exception:
            continue
    if raw is None:
        return None, "Failed to read CSV file."

    header_row = _find_header_row(raw)

    df = None
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(
                file_path, encoding=enc,
                skiprows=header_row,
                on_bad_lines="skip",
            )
            break
        except Exception:
            continue
    if df is None:
        return None, "Failed to parse CSV after locating header."

    # Drop completely empty rows and columns
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.columns = [str(c).strip() for c in df.columns]

    cols = list(df.columns)
    date_col   = find_col(cols, ["date", "txn date", "value date", "transaction date", "posting date"])
    desc_col   = find_col(cols, ["narration", "description", "particulars", "details", "remarks", "memo"])
    debit_col  = find_col(cols, ["debit", "withdrawal", "dr amount", "debit amount", "withdra"])
    credit_col = find_col(cols, ["credit", "deposit", "cr amount", "credit amount", "deposi"])
    amount_col = find_col(cols, ["amount", "net amount", "transaction amount"])

    if not date_col:
        return None, f"Cannot find Date column. Columns found: {cols}"
    if not desc_col:
        return None, f"Cannot find Description column. Columns found: {cols}"

    result, err = build_result(df, date_col, desc_col, debit_col, credit_col, amount_col, bank, cats)
    if err:
        return None, err
    return result, f"Imported {len(result)} transactions"


def _parse_pdf_tables(file_path, bank, cats):
    HEADER_KW = ["date", "description", "amount", "amt", "debit", "credit",
                 "withdrawal", "deposit", "narration", "particulars", "remarks",
                 "txn", "transaction", "dr", "cr", "chq", "cheque", "ref"]
    AMT_KW    = ["debit", "credit", "withdrawal", "deposit", "amount", "amt",
                 "dr", "cr", "withdra", "deposi"]
    raw_rows, headers = [], None

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if not table:
                    continue
                for row in table:
                    if not row:
                        continue
                    # Collapse any intra-cell newlines (pdfplumber sometimes splits
                    # multi-line cells with \n instead of separate rows)
                    row_clean = [
                        " ".join(str(c).split()) if c else ""
                        for c in row
                    ]
                    non_empty = [c for c in row_clean if c]
                    kw_hits = sum(
                        1 for cell in non_empty
                        if any(k in cell.lower() for k in HEADER_KW)
                    )
                    if headers is None and len(non_empty) >= 3 and kw_hits >= 2:
                        headers = [h.lower().strip() for h in row_clean]
                        continue
                    if headers and any(row_clean):
                        raw_rows.append(row_clean)

    if not raw_rows or headers is None:
        return None, None

    # Identify key column indices so we can detect "continuation rows"
    # (rows where the description wraps to the next line with no amounts)
    amt_idxs = [i for i, h in enumerate(headers) if any(k in h for k in AMT_KW)]
    desc_idx = next(
        (i for i, h in enumerate(headers)
         if any(k in h for k in ["narration", "description", "particulars",
                                 "details", "remarks", "transaction"])),
        None,
    )

    # Merge continuation rows into the previous row's description
    merged: list[list[str]] = []
    for row in raw_rows:
        padded = (list(row) + [""] * len(headers))[: len(headers)]
        has_amount = any(padded[i].strip() for i in amt_idxs if i < len(padded))
        desc_text  = padded[desc_idx].strip() if (desc_idx is not None and desc_idx < len(padded)) else ""

        if merged and not has_amount and desc_text:
            # No amount → continuation of the previous transaction's description
            if desc_idx is not None:
                merged[-1][desc_idx] = (merged[-1][desc_idx] + " " + desc_text).strip()
        else:
            merged.append(padded)

    all_rows = [dict(zip(headers, r)) for r in merged]
    df = pd.DataFrame(all_rows)

    date_col  = find_col(headers, ["date", "txn", "value date", "posting"])
    desc_col  = find_col(headers, ["narration", "description", "details",
                                   "particulars", "remarks", "transaction"])
    debit_col = find_col(headers, ["debit", "withdrawal", "dr", "withdra"])
    cred_col  = find_col(headers, ["credit", "deposit", "cr", "deposi"])
    amt_col   = find_col(headers, ["amount", "amt"])

    if not date_col or not desc_col:
        return None, None

    result, err = build_result(df, date_col, desc_col, debit_col, cred_col, amt_col, bank, cats)
    if err or result is None or result.empty:
        return None, None
    return result, f"Imported {len(result)} transactions from PDF"


def _parse_pdf_text(file_path, bank, cats):
    lines = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines.extend(line.strip() for line in text.split("\n") if line.strip())
    if not lines:
        return None, "No text found in PDF."
    transactions = []
    for line in lines:
        m = _DATE_RE.search(line)
        if not m:
            continue
        date_str = m.group(0)
        rest = line[m.end():].strip()
        amounts_found = _AMT_RE.findall(rest)
        if not amounts_found:
            continue
        first_amt = _AMT_RE.search(rest)
        desc = rest[: first_amt.start()].strip()
        desc = re.sub(r'^[\s\-–|]+|[\s\-–|]+$', '', desc)
        if len(desc) < 3:
            continue
        floats = []
        for a in amounts_found:
            try:
                floats.append(float(a.replace(",", "")))
            except Exception:
                floats.append(0.0)
        transactions.append({"date_str": date_str, "description": desc, "amounts": floats})
    rows = []
    for t in transactions:
        try:
            date = pd.to_datetime(t["date_str"], dayfirst=True, errors="coerce")
            if pd.isna(date):
                continue
            amts = t["amounts"]
            debit, credit = (amts[0], amts[1]) if len(amts) >= 3 else (amts[0], 0.0)
            txn_type = "Credit" if credit > 0 and debit == 0 else "Debit"
            amount   = credit if txn_type == "Credit" else debit
            if amount <= 0:
                continue
            cat = categorize(t["description"], cats)
            rows.append({
                "Date": date, "Description": t["description"],
                "Debit": debit, "Credit": credit, "Amount": amount,
                "Type": txn_type, "Bank": bank,
                "Category": cat, "TxnType": get_txn_type(t["description"], cat),
                "ID": str(uuid.uuid4())[:8],
            })
        except Exception:
            continue
    if not rows:
        return None, "Found dated lines but could not parse amounts."
    return pd.DataFrame(rows), f"Imported {len(rows)} transactions from PDF"


def parse_pdf(file_path, bank, cats):
    try:
        result, msg = _parse_pdf_tables(file_path, bank, cats)
        if result is not None and not result.empty:
            return result, msg
        result, msg = _parse_pdf_text(file_path, bank, cats)
        if result is not None and not result.empty:
            return result, msg
        return None, (
            "Could not parse this PDF automatically. "
            "Try downloading the statement as CSV from your bank's netbanking portal."
        )
    except Exception as e:
        return None, f"PDF parsing error: {str(e)}"


# ─── Display helpers ──────────────────────────────────────────────────────────

_DISPLAY_COLS = ["Date", "Description", "Amount", "Type", "TxnType", "Category", "Bank"]

_TXNTYPE_ORDER  = ["UPI", "UPI Others", "Bank to Bank", "Cash", "Other"]
_TXNTYPE_CHOICES = ["All"] + _TXNTYPE_ORDER


def get_display_df(df, bank_f="All", cat_f="All", type_f="All", txntype_f="All"):
    if df is None or df.empty:
        return pd.DataFrame(columns=_DISPLAY_COLS)
    filtered = df.copy()
    if bank_f    != "All": filtered = filtered[filtered["Bank"]    == bank_f]
    if cat_f     != "All": filtered = filtered[filtered["Category"] == cat_f]
    if type_f    != "All": filtered = filtered[filtered["Type"]     == type_f]
    if txntype_f != "All": filtered = filtered[filtered["TxnType"]  == txntype_f]
    out = filtered[_DISPLAY_COLS].copy()
    out["Date"]        = out["Date"].dt.strftime("%d %b %Y")
    out["Amount"]      = out["Amount"].round(2)
    out["Description"] = out["Description"].str[:60]   # truncate for display
    return out.sort_values("Date", ascending=False).reset_index(drop=True)


def summary_stats(df):
    if df is None or df.empty:
        return "_No transactions loaded yet. Import a bank statement to get started._"
    total_debit  = df[df["Type"] == "Debit"]["Amount"].sum()
    total_credit = df[df["Type"] == "Credit"]["Amount"].sum()
    net          = total_credit - total_debit
    net_color    = "#16a34a" if net >= 0 else "#dc2626"
    net_sign     = "+" if net >= 0 else ""
    return (
        f"<div style='display:flex; gap:24px; flex-wrap:wrap; padding:12px 0;'>"
        f"<div style='background:#f1f5f9; border-radius:10px; padding:12px 18px; min-width:130px;'>"
        f"<div style='color:#64748b; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em;'>Transactions</div>"
        f"<div style='font-size:1.4rem; font-weight:700; color:#1e293b;'>{len(df):,}</div>"
        f"<div style='color:#64748b; font-size:0.78rem;'>{df['Bank'].nunique()} account(s)</div></div>"
        f"<div style='background:#fef2f2; border-radius:10px; padding:12px 18px; min-width:150px;'>"
        f"<div style='color:#64748b; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em;'>Total Spent</div>"
        f"<div style='font-size:1.4rem; font-weight:700; color:#dc2626;'>₹{total_debit:,.0f}</div>"
        f"<div style='color:#64748b; font-size:0.78rem;'>Debits</div></div>"
        f"<div style='background:#f0fdf4; border-radius:10px; padding:12px 18px; min-width:150px;'>"
        f"<div style='color:#64748b; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em;'>Total Income</div>"
        f"<div style='font-size:1.4rem; font-weight:700; color:#16a34a;'>₹{total_credit:,.0f}</div>"
        f"<div style='color:#64748b; font-size:0.78rem;'>Credits</div></div>"
        f"<div style='background:#f8fafc; border-radius:10px; padding:12px 18px; min-width:130px; border:2px solid {net_color}33;'>"
        f"<div style='color:#64748b; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em;'>Net</div>"
        f"<div style='font-size:1.4rem; font-weight:700; color:{net_color};'>{net_sign}₹{abs(net):,.0f}</div>"
        f"<div style='color:#64748b; font-size:0.78rem;'>Savings</div></div>"
        f"</div>"
    )


def get_txntype_summary(df):
    """Count + amount of debit transactions grouped by TxnType."""
    empty = pd.DataFrame(columns=["Transaction Type", "Count", "Amount (₹)"])
    if df is None or df.empty:
        return empty
    debits = df[df["Type"] == "Debit"]
    if debits.empty:
        return empty
    grp = debits.groupby("TxnType").agg(Count=("Amount", "count"), Amount=("Amount", "sum")).reset_index()
    grp.columns = ["Transaction Type", "Count", "Amount (₹)"]
    grp["Amount (₹)"] = grp["Amount (₹)"].round(2)
    # Sort by our preferred order
    order_map = {t: i for i, t in enumerate(_TXNTYPE_ORDER)}
    grp["_order"] = grp["Transaction Type"].map(lambda x: order_map.get(x, 99))
    grp = grp.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    # Totals row
    total = pd.DataFrame([{
        "Transaction Type": "TOTAL",
        "Count": grp["Count"].sum(),
        "Amount (₹)": round(grp["Amount (₹)"].sum(), 2),
    }])
    return pd.concat([grp, total], ignore_index=True)


def get_category_summary(df):
    """Count + amount of debit transactions grouped by Category."""
    empty = pd.DataFrame(columns=["Category", "Count", "Amount (₹)"])
    if df is None or df.empty:
        return empty
    debits = df[df["Type"] == "Debit"]
    if debits.empty:
        return empty
    grp = debits.groupby("Category").agg(Count=("Amount", "count"), Amount=("Amount", "sum")).reset_index()
    grp.columns = ["Category", "Count", "Amount (₹)"]
    grp["Amount (₹)"] = grp["Amount (₹)"].round(2)
    grp = grp.sort_values("Count", ascending=False).reset_index(drop=True)
    # Totals row
    total = pd.DataFrame([{
        "Category": "TOTAL",
        "Count": grp["Count"].sum(),
        "Amount (₹)": round(grp["Amount (₹)"].sum(), 2),
    }])
    return pd.concat([grp, total], ignore_index=True)


def get_bank_choices(df):
    if df is None or df.empty:
        return ["All"]
    return ["All"] + sorted(df["Bank"].dropna().unique().tolist())


def get_cat_choices():
    return ["All"] + list(load_categories().keys())


def get_categories_df():
    cats = load_categories()
    return pd.DataFrame(
        [(name, ", ".join(kws)) for name, kws in cats.items()],
        columns=["Category", "Keywords"],
    )


# ─── Analytics charts ─────────────────────────────────────────────────────────

EMPTY_FIG = go.Figure().add_annotation(
    text="Upload transactions to see charts",
    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
    font=dict(size=16, color="gray"),
)

_TXNTYPE_COLORS = {
    "UPI":          "#636EFA",
    "UPI Others":   "#EF553B",
    "Bank to Bank": "#00CC96",
    "Cash":         "#FFA15A",
    "Other":        "#AB63FA",
}


def make_charts(df, bank_f="All"):
    if df is None or df.empty:
        return EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, EMPTY_FIG
    data = df.copy()
    if bank_f != "All":
        data = data[data["Bank"] == bank_f]
    expenses = data[data["Type"] == "Debit"]

    # Pie – Category
    cat_sum = expenses.groupby("Category")["Amount"].sum().reset_index()
    cat_sum = cat_sum[cat_sum["Amount"] > 0].sort_values("Amount", ascending=False)
    pie_cat = px.pie(cat_sum, values="Amount", names="Category",
                     title="Spending by Category",
                     color_discrete_sequence=px.colors.qualitative.Set3, hole=0.3)
    pie_cat.update_layout(template="plotly_white", legend_title_text="Category")
    pie_cat.update_traces(textposition="inside", textinfo="percent+label")

    # Pie – TxnType
    txn_sum = expenses.groupby("TxnType")["Amount"].sum().reset_index()
    txn_sum = txn_sum[txn_sum["Amount"] > 0].sort_values("Amount", ascending=False)
    pie_txn = px.pie(txn_sum, values="Amount", names="TxnType",
                     title="Spending by Transaction Type",
                     color="TxnType", color_discrete_map=_TXNTYPE_COLORS, hole=0.3)
    pie_txn.update_layout(template="plotly_white", legend_title_text="Type")
    pie_txn.update_traces(textposition="inside", textinfo="percent+label")

    # Bar – monthly
    data["Month"] = data["Date"].dt.to_period("M").astype(str)
    monthly = data.groupby(["Month", "Type"])["Amount"].sum().reset_index()
    bar = px.bar(monthly, x="Month", y="Amount", color="Type",
                 title="Monthly Income vs Expenses",
                 color_discrete_map={"Debit": "#EF553B", "Credit": "#00CC96"},
                 barmode="group", labels={"Amount": "₹ Amount", "Month": "Month"})
    bar.update_layout(template="plotly_white")

    # Bar – bank
    bank_sum = expenses.groupby("Bank")["Amount"].sum().reset_index()
    bank_bar = px.bar(bank_sum, x="Bank", y="Amount",
                      title="Spending by Bank / Account",
                      color="Bank", labels={"Amount": "₹ Amount"}, text_auto=".2s")
    bank_bar.update_layout(template="plotly_white", showlegend=False)

    return pie_cat, pie_txn, bar, bank_bar


def make_budget_charts(df):
    """Return (grouped_bar, utilization_bar, summary_df) for budget vs actual."""
    _empty_tbl = pd.DataFrame(columns=["Category", "Budget (₹)", "Actual (₹)", "Variance (₹)", "Used (%)"])
    if df is None or df.empty:
        return EMPTY_FIG, EMPTY_FIG, _empty_tbl

    actual = df[df["Type"] == "Debit"].groupby("Category")["Amount"].sum()

    rows = []
    for cat, budget in DEFAULT_BUDGETS.items():
        if budget <= 0:
            continue
        spent = round(float(actual.get(cat, 0)), 2)
        variance = round(budget - spent, 2)
        pct = round(spent / budget * 100, 1)
        rows.append({
            "Category":     cat,
            "Budget (₹)":   budget,
            "Actual (₹)":   spent,
            "Variance (₹)": variance,
            "Used (%)":     pct,
        })

    if not rows:
        return EMPTY_FIG, EMPTY_FIG, _empty_tbl

    bdf = pd.DataFrame(rows).sort_values("Used (%)", ascending=False).reset_index(drop=True)

    # Chart 1: Grouped bar — Budget vs Actual
    actual_colors = [
        "#EF553B" if r["Actual (₹)"] > r["Budget (₹)"] else "#00CC96"
        for _, r in bdf.iterrows()
    ]
    fig_bar = go.Figure([
        go.Bar(name="Budget",       x=bdf["Category"], y=bdf["Budget (₹)"],
               marker_color="#A8D8EA", opacity=0.85),
        go.Bar(name="Actual Spend", x=bdf["Category"], y=bdf["Actual (₹)"],
               marker_color=actual_colors),
    ])
    fig_bar.update_layout(
        barmode="group",
        title="Budget vs Actual Spend by Category",
        template="plotly_white",
        xaxis_tickangle=-30,
        yaxis_title="₹ Amount",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )

    # Chart 2: Horizontal utilization bars
    bar_colors = [
        "#EF553B" if p > 100 else "#FFA15A" if p > 80 else "#00CC96"
        for p in bdf["Used (%)"]
    ]
    fig_util = go.Figure(go.Bar(
        x=bdf["Used (%)"],
        y=bdf["Category"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{p}%" for p in bdf["Used (%)"]],
        textposition="outside",
    ))
    x_max = max(bdf["Used (%)"].max() * 1.15, 115)
    fig_util.add_vline(x=100, line_dash="dash", line_color="#EF553B",
                       annotation_text="Budget limit", annotation_position="top")
    fig_util.update_layout(
        title="Budget Utilization (%)",
        template="plotly_white",
        xaxis=dict(title="% of Budget Used", range=[0, x_max]),
        yaxis=dict(autorange="reversed"),
        height=max(300, len(bdf) * 38),
    )

    # Summary table with totals row
    display_tbl = bdf[["Category", "Budget (₹)", "Actual (₹)", "Variance (₹)", "Used (%)"]].copy()
    display_tbl["Variance (₹)"] = display_tbl["Variance (₹)"].apply(
        lambda v: f"+{v:,.0f}" if v >= 0 else f"{v:,.0f}"
    )
    total_budget = bdf["Budget (₹)"].sum()
    total_actual = bdf["Actual (₹)"].sum()
    total_var    = total_budget - total_actual
    total_pct    = round(total_actual / total_budget * 100, 1) if total_budget else 0
    totals = pd.DataFrame([{
        "Category":     "TOTAL",
        "Budget (₹)":   total_budget,
        "Actual (₹)":   round(total_actual, 2),
        "Variance (₹)": f"+{total_var:,.0f}" if total_var >= 0 else f"{total_var:,.0f}",
        "Used (%)":     total_pct,
    }])
    return fig_bar, fig_util, pd.concat([display_tbl, totals], ignore_index=True)


# ─── Insight Engine ───────────────────────────────────────────────────────────

_RECOMMENDATIONS = {
    "Dining & Restaurants": "Consider cooking at home more often or setting a weekly dining-out limit.",
    "Groceries":            "Try planning a weekly meal list to avoid impulse buys and reduce waste.",
    "Transport & Fuel":     "Look into carpooling, public transit, or consolidating trips to cut fuel costs.",
    "Shopping":             "Review recent purchases — unsubscribe from sale alerts that trigger impulse spending.",
    "Entertainment & Sports": "Check for unused subscriptions (OTT, gym) that can be paused or cancelled.",
    "Utilities":            "Audit devices on standby; switching to LED and time-of-use tariffs can reduce bills.",
    "Healthcare & Medical": "Ensure health insurance covers recurring expenses to reduce out-of-pocket spend.",
    "Personal Care":        "Compare at-home alternatives for services like grooming or spa visits.",
    "Housing & Rent":       "Review maintenance charges and shared utility splits with landlord or society.",
    "Investments":          "Great commitment — ensure SIPs are aligned with your current risk appetite.",
    "Insurance":            "Compare premiums annually; bundling policies can reduce total cost.",
    "Education":            "Look for free/discounted courses on Coursera, NPTEL, or YouTube as supplements.",
    "Farm Land":            "Track seasonal expenses separately; consider a dedicated farm account.",
    "Others":               "High 'Others' spend usually means transactions need better categorization — review them.",
}


def generate_insights(budget_df: pd.DataFrame) -> list[str]:
    """
    Analyse budget_df (columns: Category, Budget (₹), Actual (₹), Used (%))
    and return a list of human-readable, emoji-prefixed insight strings.
    The list is sorted: Critical first, then Warnings, then Underutilized.
    Skips the synthetic TOTAL row and categories with no budget.
    """
    if budget_df is None or budget_df.empty:
        return ["_No budget data yet — import transactions to see insights._"]

    # Drop totals row and rows with zero budget
    df = budget_df[
        (budget_df["Category"] != "TOTAL") &
        (budget_df["Budget (₹)"].apply(lambda x: float(str(x).replace(",", "") or 0)) > 0)
    ].copy()

    if df.empty:
        return ["_No budgeted categories found._"]

    # Coerce numeric columns (they may be strings if formatted)
    def _num(col):
        return df[col].apply(lambda x: float(str(x).replace(",", "").replace("+", "") or 0))

    df["_budget"] = _num("Budget (₹)")
    df["_actual"] = _num("Actual (₹)")
    df["_pct"]    = _num("Used (%)")
    df["_var"]    = df["_actual"] - df["_budget"]   # positive = overspent

    criticals, warnings, underused, ok = [], [], [], []

    for _, row in df.iterrows():
        cat    = row["Category"]
        pct    = row["_pct"]
        var    = row["_var"]
        actual = row["_actual"]
        budget = row["_budget"]
        rec    = _RECOMMENDATIONS.get(cat, "Review this category's spending pattern.")

        if cat == "Others" and pct > 200:
            criticals.append(
                f"🚨 **Others** overspent by ₹{var:,.0f} ({pct:.0f}%) — "
                f"likely misclassification. {rec}"
            )
        elif pct > 120:
            criticals.append(
                f"🚨 **{cat}** overspent by ₹{var:,.0f} ({pct:.0f}% of budget). {rec}"
            )
        elif pct >= 90:
            remaining = budget - actual
            warnings.append(
                f"⚠️ **{cat}** is at {pct:.0f}% of budget — "
                f"only ₹{remaining:,.0f} remaining. {rec}"
            )
        elif pct < 50:
            saved = budget - actual
            underused.append(
                f"💡 **{cat}** is underutilized at {pct:.0f}% "
                f"(₹{saved:,.0f} unspent this period)."
            )
        else:
            ok.append(f"✅ **{cat}** is on track at {pct:.0f}% of budget.")

    # Sort criticals by overspend amount (highest first), take top 3 critical
    def _overspend(s):
        # extract ₹ amount from string for sorting
        m = re.search(r'₹([\d,]+)', s)
        return int(m.group(1).replace(",", "")) if m else 0

    criticals.sort(key=_overspend, reverse=True)
    warnings.sort(key=_overspend, reverse=True)

    return criticals[:3] + warnings + underused + ok


def generate_recommendations(insights: list[str]) -> list[str]:
    """
    Given the insight list, return a deduplicated set of action recommendations.
    Recommendations are already embedded in each insight string; this extracts
    and deduplicates the action part (everything after the em-dash or period).
    """
    recs = []
    for line in insights:
        # Extract sentence after the last '. ' or '— '
        for sep in [" — ", ". "]:
            if sep in line:
                tail = line.rsplit(sep, 1)[-1].strip()
                if tail and tail not in recs:
                    recs.append(tail)
                break
    return recs


def build_insights_markdown(df: pd.DataFrame) -> str:
    """Render insights + recommendations as a Gradio Markdown string."""
    insights = generate_insights(df)
    if not insights or insights == ["_No budget data yet — import transactions to see insights._"]:
        return "_Import transactions to see budget insights._"

    criticals = [i for i in insights if i.startswith("🚨")]
    warnings  = [i for i in insights if i.startswith("⚠️")]
    underused = [i for i in insights if i.startswith("💡")]
    ok_items  = [i for i in insights if i.startswith("✅")]

    sections = []
    if criticals:
        sections.append("#### 🚨 Critical Overspend")
        sections += [f"- {i}" for i in criticals]
    if warnings:
        sections.append("\n#### ⚠️ Approaching Limit")
        sections += [f"- {i}" for i in warnings]
    if underused:
        sections.append("\n#### 💡 Underutilized Budget")
        sections += [f"- {i}" for i in underused]
    if ok_items:
        sections.append("\n#### ✅ On Track")
        sections += [f"- {i}" for i in ok_items]

    recs = generate_recommendations(insights)
    if recs:
        sections += ["\n---", "#### 📋 Recommended Actions", ""]
        sections += [f"{idx+1}. {r}" for idx, r in enumerate(recs)]

    return "\n".join(sections)


# ─── Gradio App ───────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Family Expense Tracker",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    ),
    css="""
    .summary-box { background: #f8fafc; border-radius: 12px; padding: 16px 20px;
                   border-left: 4px solid #3b82f6; margin-bottom: 4px; }
    .insight-box { background: #f0fdf4; border-radius: 12px; padding: 16px 20px;
                   border-left: 4px solid #22c55e; }
    .section-header { font-size: 1.05rem; font-weight: 600; color: #1e293b;
                      margin: 8px 0 4px 0; }
    .tab-nav button { font-weight: 600 !important; }
    footer { display: none !important; }
    """
) as app:
    state = gr.State(load_transactions())

    gr.Markdown(
        """<div style='text-align:center; padding: 18px 0 8px 0;'>
        <span style='font-size:2rem; font-weight:700; color:#1e293b;'>🏠 Family Expense Tracker</span><br/>
        <span style='color:#64748b; font-size:0.95rem;'>Track · Categorize · Analyse · Stay on Budget</span>
        </div>"""
    )

    with gr.Tabs():

        # ── Tab 1: Import Transactions ────────────────────────────────────────
        with gr.TabItem("📥 Import Transactions"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Bank Statement")
                    file_input = gr.File(
                        label="CSV or PDF",
                        file_types=[".csv", ".pdf"],
                    )
                    bank_input = gr.Textbox(
                        label="Bank / Account Name",
                        placeholder="Auto-detected from file — edit if needed",
                    )
                    import_btn = gr.Button("⬆️ Import", variant="primary", size="lg")
                    import_msg = gr.Markdown("")

                with gr.Column(scale=2):
                    preview_header_md = gr.Markdown("### Preview — Newly Imported Rows")
                    preview_table = gr.Dataframe(
                        headers=_DISPLAY_COLS, interactive=False, wrap=True,
                    )
                    with gr.Accordion("🔍 Debug raw extraction", open=False):
                        debug_btn = gr.Button(
                            "Show Raw Extracted Data", variant="secondary", size="sm"
                        )
                        debug_out = gr.Textbox(
                            label="Raw rows (first 30 shown)",
                            lines=15, interactive=False,
                        )

        # ── Tab 2: Transactions ───────────────────────────────────────────────
        with gr.TabItem("📋 Transactions"):

            # Filters
            with gr.Row(equal_height=True):
                cat_filter = gr.Dropdown(
                    choices=get_cat_choices(), value="All",
                    label="Category", scale=2,
                )
                txntype_filter = gr.Dropdown(
                    choices=_TXNTYPE_CHOICES, value="All",
                    label="Transaction Type", scale=1,
                )
                bank_filter = gr.Dropdown(
                    choices=["All"], value="All",
                    label="Bank", scale=1,
                )
                type_filter = gr.Dropdown(
                    choices=["All", "Debit", "Credit"], value="All",
                    label="Debit / Credit", scale=1,
                )
                refresh_btn = gr.Button("🔄 Filter", variant="secondary", scale=1, size="sm")

            gr.Markdown(
                "<span style='color:#64748b; font-size:0.85rem;'>"
                "Click a row to select it, then pick a category and hit Save.</span>"
            )
            txn_table = gr.Dataframe(
                headers=_DISPLAY_COLS, interactive=False, wrap=True,
                column_count=(len(_DISPLAY_COLS), "fixed"),
            )

            selected_row_state = gr.State(None)

            selected_desc_md = gr.Markdown(
                "<span style='color:#64748b;'>_No transaction selected — click a row above._</span>"
            )

            with gr.Row(equal_height=True):
                row_cat_dropdown = gr.Dropdown(
                    choices=list(load_categories().keys()),
                    label="Change Category To",
                    scale=3,
                    interactive=True,
                )
                save_btn = gr.Button("💾 Save", variant="primary", scale=1, size="sm")
                ai_btn   = gr.Button("🤖 AI Categorize", variant="secondary", scale=1, size="sm")
            action_msg = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("### Spending Summary")
            with gr.Row():
                txn_cat_summary_tbl = gr.Dataframe(
                    headers=["Category", "Count", "Amount (₹)"],
                    interactive=False, label="By Category", wrap=False,
                )
                txn_txntype_summary_tbl = gr.Dataframe(
                    headers=["Transaction Type", "Count", "Amount (₹)"],
                    interactive=False, label="By Transaction Type", wrap=False,
                )

            gr.Markdown("---")
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear All Data", variant="stop", size="sm")
            clear_msg = gr.Markdown("")

        # ── Tab 3: Analytics ──────────────────────────────────────────────────
        with gr.TabItem("📊 Analytics"):

            # ── KPI summary bar ───────────────────────────────────────────────
            summary_md = gr.Markdown("Import transactions to see analytics.")

            # ── Controls ──────────────────────────────────────────────────────
            with gr.Row(equal_height=True):
                analytics_bank = gr.Dropdown(
                    choices=["All"], value="All",
                    label="Filter by Bank", scale=2,
                )
                analytics_refresh = gr.Button(
                    "🔄 Refresh", variant="secondary", scale=1, size="sm"
                )

            gr.Markdown("---")

            # ── Spending overview pies ────────────────────────────────────────
            gr.Markdown("### Spending Overview")
            with gr.Row():
                pie_cat_chart   = gr.Plot(label="By Category")
                pie_txn_chart   = gr.Plot(label="By Transaction Type")

            # ── Budget vs Actual ──────────────────────────────────────────────
            gr.Markdown("---")
            gr.Markdown("### Budget vs Actual")
            gr.Markdown(
                "<span style='color:#64748b; font-size:0.85rem;'>"
                "🟢 Under budget &nbsp;|&nbsp; 🟠 >80% used &nbsp;|&nbsp; 🔴 Over budget"
                "</span>"
            )
            with gr.Row():
                budget_bar_chart  = gr.Plot(label="Budget vs Actual Spend")
                budget_util_chart = gr.Plot(label="Budget Utilization (%)")

            budget_table = gr.Dataframe(
                headers=["Category", "Budget (₹)", "Actual (₹)", "Variance (₹)", "Used (%)"],
                interactive=False, label="Budget Summary", wrap=False,
            )

            # ── Insights (below visuals) ──────────────────────────────────────
            gr.Markdown("---")
            gr.Markdown("### Budget Insights & Recommendations")
            insights_md = gr.Markdown("_Import transactions to see budget insights._")

            # Hidden components (keep wiring intact)
            bar_month_chart = gr.Plot(visible=False)
            bank_bar_chart  = gr.Plot(visible=False)
            cat_summary_tbl = gr.Dataframe(
                headers=["Category", "Count", "Amount (₹)"],
                interactive=False, visible=False,
            )
            txntype_summary_tbl = gr.Dataframe(
                headers=["Transaction Type", "Count", "Amount (₹)"],
                interactive=False, visible=False,
            )

    # ─── Event handlers ───────────────────────────────────────────────────────

    def on_debug_pdf(file):
        if file is None:
            return "Upload a file first."
        ext = os.path.splitext(file.name)[1].lower()
        try:
            if ext == ".csv":
                for enc in ("utf-8-sig", "latin-1"):
                    try:
                        with open(file.name, encoding=enc) as f:
                            raw = f.readlines()
                        break
                    except UnicodeDecodeError:
                        continue
                hi = _find_header_row(raw)
                lines = [f"Header found at line {hi}: {raw[hi].strip()}",
                         f"Columns: {raw[hi].strip()}", "---"]
                lines += [f"row {i:02d}: {l.rstrip()}" for i, l in enumerate(raw[hi+1:hi+31])]
                return "\n".join(lines)
            else:
                parts = []
                with pdfplumber.open(file.name) as pdf:
                    for i, page in enumerate(pdf.pages[:3]):
                        tables = page.extract_tables() or []
                        parts.append(f"\n=== Page {i+1} — {len(tables)} table(s) ===")
                        for j, tbl in enumerate(tables[:2]):
                            parts.append(f"--- Table {j+1} ({len(tbl)} rows) ---")
                            for k, row in enumerate(tbl[:25]):
                                cells = [str(c).replace("\n","⏎").strip() if c else "·" for c in row]
                                parts.append(f"  row {k:02d}: {' | '.join(cells)}")
                        if not tables:
                            parts.append((page.extract_text() or "(no text)")[:600])
                return "\n".join(parts)[:5000]
        except Exception as e:
            return f"Error: {e}"

    debug_btn.click(on_debug_pdf, inputs=[file_input], outputs=[debug_out])

    # ── Analytics outputs helper ──────────────────────────────────────────────
    def _analytics_outputs(df, bank_f="All"):
        charts = make_charts(df, bank_f=bank_f)
        return (
            summary_stats(df),
            charts[0], charts[1], charts[2], charts[3],
            get_category_summary(df),
            get_txntype_summary(df),
        )

    def _budget_outputs(df):
        b = make_budget_charts(df)
        # b[2] is the budget summary dataframe — feed it to the insight engine
        insights = build_insights_markdown(b[2])
        return b[0], b[1], b[2], insights  # fig_bar, fig_util, summary_df, insights_md

    # ── Import ────────────────────────────────────────────────────────────────
    _PREVIEW_HEADER_DEFAULT = "### Preview — Newly Imported Rows"

    def on_import(file, bank, df_state):
        if file is None:
            return (df_state, "Please select a file first.",
                    _PREVIEW_HEADER_DEFAULT,
                    pd.DataFrame(columns=_DISPLAY_COLS),
                    gr.Dropdown(choices=get_bank_choices(df_state)),
                    gr.Dropdown(choices=get_bank_choices(df_state)),
                    get_display_df(df_state),
                    *_analytics_outputs(df_state),
                    get_category_summary(df_state),
                    get_txntype_summary(df_state),
                    *_budget_outputs(df_state))

        bank = (bank or "Unknown Bank").strip()
        cats = load_categories()
        ext  = os.path.splitext(file.name)[1].lower()

        if ext == ".csv":
            new_df, msg = parse_csv(file.name, bank, cats)
        elif ext == ".pdf":
            new_df, msg = parse_pdf(file.name, bank, cats)
        else:
            return (df_state, f"Unsupported file type: {ext}",
                    _PREVIEW_HEADER_DEFAULT,
                    pd.DataFrame(columns=_DISPLAY_COLS),
                    gr.Dropdown(choices=get_bank_choices(df_state)),
                    gr.Dropdown(choices=get_bank_choices(df_state)),
                    get_display_df(df_state),
                    *_analytics_outputs(df_state),
                    get_category_summary(df_state),
                    get_txntype_summary(df_state),
                    *_budget_outputs(df_state))

        if new_df is None:
            return (df_state, f"❌ {msg}",
                    _PREVIEW_HEADER_DEFAULT,
                    pd.DataFrame(columns=_DISPLAY_COLS),
                    gr.Dropdown(choices=get_bank_choices(df_state)),
                    gr.Dropdown(choices=get_bank_choices(df_state)),
                    get_display_df(df_state),
                    *_analytics_outputs(df_state),
                    get_category_summary(df_state),
                    get_txntype_summary(df_state),
                    *_budget_outputs(df_state))

        if df_state is not None and not df_state.empty:
            combined = pd.concat([df_state, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["Date", "Description", "Amount"], keep="first"
            )
        else:
            combined = new_df

        added = len(new_df)
        preview_header = f"### Preview — {added} Transaction{'s' if added != 1 else ''} Added"
        save_transactions(combined)
        bank_dd = gr.Dropdown(choices=get_bank_choices(combined), value="All")
        return (
            combined,
            f"✅ **Imported:** {msg} from **{bank}**",
            preview_header,
            get_display_df(new_df),
            bank_dd, bank_dd,
            get_display_df(combined),
            *_analytics_outputs(combined),
            get_category_summary(combined),
            get_txntype_summary(combined),
            *_budget_outputs(combined),
        )

    file_input.change(
        fn=lambda f: detect_bank(f.name) if f else "",
        inputs=[file_input],
        outputs=[bank_input],
    )

    import_btn.click(
        on_import,
        inputs=[file_input, bank_input, state],
        outputs=[state, import_msg, preview_header_md, preview_table,
                 bank_filter, analytics_bank,
                 txn_table,
                 summary_md,
                 pie_cat_chart, pie_txn_chart, bar_month_chart, bank_bar_chart,
                 cat_summary_tbl, txntype_summary_tbl,
                 txn_cat_summary_tbl, txn_txntype_summary_tbl,
                 budget_bar_chart, budget_util_chart, budget_table, insights_md],
    )

    # ── Filter / refresh transaction table ────────────────────────────────────
    def on_refresh(df, cat_f, txntype_f, bank_f, type_f):
        return get_display_df(df, bank_f=bank_f, cat_f=cat_f,
                              type_f=type_f, txntype_f=txntype_f)

    refresh_btn.click(
        on_refresh,
        inputs=[state, cat_filter, txntype_filter, bank_filter, type_filter],
        outputs=[txn_table],
    )

    # ── Row selection → populate category dropdown ────────────────────────────
    def on_row_select(evt: gr.SelectData, tbl):
        if tbl is None or tbl.empty:
            return None, "_No transaction selected._", gr.Dropdown()
        row_idx = evt.index[0]
        row = tbl.iloc[row_idx]
        desc    = str(row.get("Description", ""))
        amount  = row.get("Amount", 0)
        cur_cat = str(row.get("Category", "Others"))
        info = f"_Selected: **{desc}** &nbsp;|&nbsp; Amount: **₹{amount}** &nbsp;|&nbsp; Current category: **{cur_cat}**_"
        return (desc, amount), info, gr.Dropdown(value=cur_cat)

    txn_table.select(
        on_row_select,
        inputs=[txn_table],
        outputs=[selected_row_state, selected_desc_md, row_cat_dropdown],
    )

    # ── Manual category save ──────────────────────────────────────────────────
    def on_save_changes(row_key, new_cat, df_state):
        if df_state is None or df_state.empty:
            return df_state, "Nothing to save.", get_display_df(df_state), get_category_summary(df_state), get_txntype_summary(df_state)
        if row_key is None:
            return df_state, "⚠️ No row selected. Click a row in the table first.", get_display_df(df_state), get_category_summary(df_state), get_txntype_summary(df_state)
        if not new_cat:
            return df_state, "⚠️ Please select a category.", get_display_df(df_state), get_category_summary(df_state), get_txntype_summary(df_state)

        desc_prefix, amount = row_key
        full = df_state.copy()
        try:
            amount_f = round(float(amount), 2)
        except (ValueError, TypeError):
            return df_state, "Invalid row data.", get_display_df(df_state), get_category_summary(df_state), get_txntype_summary(df_state)

        mask = (
            (full["Description"].str[:60] == str(desc_prefix)[:60]) &
            (full["Amount"].round(2) == amount_f)
        )
        changed = 0
        for idx in full.index[mask]:
            if full.at[idx, "Category"] != new_cat:
                full.at[idx, "Category"] = new_cat
                full.at[idx, "TxnType"]  = get_txn_type(
                    full.at[idx, "Description"], new_cat
                )
                changed += 1

        if changed:
            # Learn: store merchant keyword → category so future imports auto-categorize
            update_merchant_cache(str(desc_prefix), new_cat)

        save_transactions(full)
        msg = f"✅ Updated **{changed}** transaction(s) to **{new_cat}**." if changed else "No changes made (category was already set)."
        return full, msg, get_display_df(full), get_category_summary(full), get_txntype_summary(full)

    save_btn.click(
        on_save_changes,
        inputs=[selected_row_state, row_cat_dropdown, state],
        outputs=[state, action_msg, txn_table, txn_cat_summary_tbl, txn_txntype_summary_tbl],
    )

    # ── AI categorize ─────────────────────────────────────────────────────────
    def on_ai_categorize(df_state):
        if df_state is None or df_state.empty:
            return df_state, "No data to categorize.", get_display_df(df_state), get_category_summary(df_state), get_txntype_summary(df_state)
        updated, msg = ai_categorize_others(df_state)
        return updated, msg, get_display_df(updated), get_category_summary(updated), get_txntype_summary(updated)

    ai_btn.click(
        on_ai_categorize,
        inputs=[state],
        outputs=[state, action_msg, txn_table, txn_cat_summary_tbl, txn_txntype_summary_tbl],
    )

    # ── Clear all ─────────────────────────────────────────────────────────────
    def on_clear():
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        empty = pd.DataFrame(
            columns=["ID","Date","Description","Debit","Credit",
                     "Type","Amount","Category","TxnType","Bank"]
        )
        return (empty, "✅ All data cleared.",
                pd.DataFrame(columns=_DISPLAY_COLS),
                gr.Dropdown(choices=["All"], value="All"),
                gr.Dropdown(choices=["All"], value="All"),
                *_analytics_outputs(empty),
                get_category_summary(empty),
                get_txntype_summary(empty),
                *_budget_outputs(empty))

    clear_btn.click(
        on_clear,
        outputs=[state, clear_msg, txn_table,
                 bank_filter, analytics_bank,
                 summary_md,
                 pie_cat_chart, pie_txn_chart, bar_month_chart, bank_bar_chart,
                 cat_summary_tbl, txntype_summary_tbl,
                 txn_cat_summary_tbl, txn_txntype_summary_tbl,
                 budget_bar_chart, budget_util_chart, budget_table, insights_md],
    )

    # ── Analytics refresh (bank filter) ──────────────────────────────────────
    def on_analytics_refresh(df, bank_f):
        return (*_analytics_outputs(df, bank_f=bank_f), *_budget_outputs(df))

    analytics_refresh.click(
        on_analytics_refresh,
        inputs=[state, analytics_bank],
        outputs=[summary_md,
                 pie_cat_chart, pie_txn_chart, bar_month_chart, bank_bar_chart,
                 cat_summary_tbl, txntype_summary_tbl,
                 budget_bar_chart, budget_util_chart, budget_table, insights_md],
    )

    # ── Initial load ──────────────────────────────────────────────────────────
    def on_load(df):
        bank_dd = gr.Dropdown(choices=get_bank_choices(df), value="All")
        return (
            bank_dd, bank_dd,
            get_display_df(df),
            *_analytics_outputs(df),
            get_category_summary(df),
            get_txntype_summary(df),
            *_budget_outputs(df),
        )

    app.load(
        on_load,
        inputs=[state],
        outputs=[bank_filter, analytics_bank,
                 txn_table,
                 summary_md,
                 pie_cat_chart, pie_txn_chart, bar_month_chart, bank_bar_chart,
                 cat_summary_tbl, txntype_summary_tbl,
                 txn_cat_summary_tbl, txn_txntype_summary_tbl,
                 budget_bar_chart, budget_util_chart, budget_table, insights_md],
    )


if __name__ == "__main__":
    app.launch()
