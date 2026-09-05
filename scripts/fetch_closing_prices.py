from __future__ import annotations

import io
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

NSE_HOME = "https://www.nseindia.com/"
BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/products/content/"
    "sec_bhavdata_full_{date}.csv"
)
IST = ZoneInfo("Asia/Kolkata")
REQUEST_TIMEOUT = (15, 45)
MAX_404_RETRIES = 5
RETRY_WAIT_SECONDS = 180

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/csv;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_HOME,
    "Connection": "keep-alive",
}

HISTORICAL_COLUMNS = [
    "SYMBOL",
    "SERIES",
    "OPEN_PRICE",
    "HIGH_PRICE",
    "LOW_PRICE",
    "CLOSE_PRICE",
    "PREV_CLOSE",
    "TTL_TRD_QNTY",
    "TURNOVER_LACS",
    "DATE",
]

PRICE_COLUMNS = [
    "OPEN_PRICE",
    "HIGH_PRICE",
    "LOW_PRICE",
    "CLOSE_PRICE",
    "PREV_CLOSE",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def create_nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def handshake(session: requests.Session) -> None:
    logging.info("Performing NSE cookie handshake...")
    response = session.get(NSE_HOME, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    logging.info("NSE handshake successful (HTTP %s).", response.status_code)


def download_bhavcopy(session: requests.Session, nse_date: str) -> bytes | None:
    url = BHAVCOPY_URL.format(date=nse_date)
    retries_used = 0

    while True:
        logging.info("Downloading %s", url)
        response = session.get(url, timeout=REQUEST_TIMEOUT)

        # A stale/blocked cookie occasionally manifests as 403. Refresh once and retry.
        if response.status_code == 403:
            logging.warning("Received HTTP 403; refreshing NSE session cookies and retrying once.")
            handshake(session)
            response = session.get(url, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            if not response.content.strip():
                raise RuntimeError("NSE returned HTTP 200 but the bhavcopy body was empty.")
            return response.content

        if response.status_code == 404:
            if retries_used >= MAX_404_RETRIES:
                logging.warning(
                    "Bhavcopy is still unavailable after %d retries. "
                    "Treating this as a market holiday/upload delay and exiting successfully.",
                    MAX_404_RETRIES,
                )
                return None

            retries_used += 1
            logging.warning(
                "Bhavcopy not found (HTTP 404). Retry %d/%d in %d seconds...",
                retries_used,
                MAX_404_RETRIES,
                RETRY_WAIT_SECONDS,
            )
            time.sleep(RETRY_WAIT_SECONDS)
            continue

        response.raise_for_status()


def clean_bhavcopy(raw_csv: bytes, trade_date_iso: str, root: Path) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw_csv), dtype=str, skipinitialspace=True)
    df.columns = df.columns.astype(str).str.strip()

    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]):
            df[column] = df[column].str.strip()

    if "DATE1" in df.columns and "DATE" not in df.columns:
        df = df.rename(columns={"DATE1": "DATE"})

    required = set(HISTORICAL_COLUMNS) - {"DATE"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Bhavcopy is missing required columns: {', '.join(missing)}")

    df["SYMBOL"] = df["SYMBOL"].str.upper()
    df["SERIES"] = df["SERIES"].str.upper()
    df = df.loc[df["SERIES"].eq("EQ")].copy()

    universe_path = root / "universe" / "nifty_universe.csv"
    if universe_path.exists():
        universe_df = pd.read_csv(universe_path, dtype=str, skipinitialspace=True)
        universe_df.columns = universe_df.columns.astype(str).str.strip()
        for column in universe_df.columns:
            if pd.api.types.is_object_dtype(universe_df[column]):
                universe_df[column] = universe_df[column].str.strip()

        symbol_column = "SYMBOL" if "SYMBOL" in universe_df.columns else universe_df.columns[0]
        symbols = set(universe_df[symbol_column].dropna().str.upper())
        if not symbols:
            raise ValueError(f"Universe file exists but contains no symbols: {universe_path}")

        before = len(df)
        df = df.loc[df["SYMBOL"].isin(symbols)].copy()
        logging.info("Universe filter kept %d of %d EQ rows.", len(df), before)
    else:
        logging.info("No universe/nifty_universe.csv found; exporting all EQ stocks.")

    if df.empty:
        raise ValueError("No EQ rows remain after filtering; refusing to publish an empty payload.")

    numeric_columns = [
        "OPEN_PRICE",
        "HIGH_PRICE",
        "LOW_PRICE",
        "CLOSE_PRICE",
        "PREV_CLOSE",
        "TTL_TRD_QNTY",
        "TURNOVER_LACS",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df[PRICE_COLUMNS].isna().any().any():
        raise ValueError("One or more required price fields are null; refusing to publish invalid JSON.")

    # Normalize the date instead of depending on NSE's display format in DATE1.
    df["DATE"] = trade_date_iso
    df = df[HISTORICAL_COLUMNS].sort_values("SYMBOL", kind="stable").reset_index(drop=True)
    return df


def write_outputs(df: pd.DataFrame, trade_date: datetime, root: Path) -> tuple[Path, Path]:
    data_dir = root / "data"
    public_dir = root / "public"
    data_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    archive_path = data_dir / f"nifty_closing_{trade_date:%Y_%m_%d}.csv"
    latest_path = public_dir / "latest.json"

    df.to_csv(archive_path, index=False, lineterminator="\n")

    payload = [
        {
            "symbol": row.SYMBOL,
            "close": float(row.CLOSE_PRICE),
            "open": float(row.OPEN_PRICE),
            "high": float(row.HIGH_PRICE),
            "low": float(row.LOW_PRICE),
            "prev_close": float(row.PREV_CLOSE),
            "date": row.DATE,
        }
        for row in df.itertuples(index=False)
    ]

    with latest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    return archive_path, latest_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    now_ist = datetime.now(IST)
    nse_date = now_ist.strftime("%d%m%Y")
    trade_date_iso = now_ist.strftime("%Y-%m-%d")
    root = project_root()

    logging.info("Target NSE trade date: %s", trade_date_iso)

    session = create_nse_session()
    try:
        handshake(session)
        raw_csv = download_bhavcopy(session, nse_date)
        if raw_csv is None:
            return 0

        df = clean_bhavcopy(raw_csv, trade_date_iso, root)
        archive_path, latest_path = write_outputs(df, now_ist, root)
        logging.info("Wrote %d stocks to %s", len(df), archive_path)
        logging.info("Updated frontend payload at %s", latest_path)
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logging.exception("NSE EOD pipeline failed.")
        sys.exit(1)
