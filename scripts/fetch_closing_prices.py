from __future__ import annotations

import io
import json
import logging
import os
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
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "text/csv;q=0.8,*/*;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
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
    """
    Attempt the normal NSE homepage cookie handshake.

    GitHub Actions uses cloud/datacenter IP addresses. NSE sometimes
    returns HTTP 403 to those IPs. A failed homepage handshake should
    therefore not prevent us from trying the separate archive host.
    """

    logging.info("Attempting NSE homepage cookie handshake...")

    try:
        response = session.get(
            NSE_HOME,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 200:
            logging.info(
                "NSE homepage handshake successful. Cookies stored."
            )
            return

        logging.warning(
            "NSE homepage handshake returned HTTP %s. "
            "This commonly occurs on cloud runners. "
            "Continuing with the NSE archive host.",
            response.status_code,
        )

    except requests.RequestException as exc:
        logging.warning(
            "NSE homepage handshake could not be completed: %s. "
            "Continuing with the NSE archive host.",
            exc,
        )


def download_bhavcopy(
    session: requests.Session,
    nse_date: str,
) -> bytes | None:

    url = BHAVCOPY_URL.format(date=nse_date)

    retries_used = 0

    while True:

        logging.info(
            "Requesting NSE bhavcopy: %s",
            url,
        )

        try:
            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT,
            )

        except requests.RequestException as exc:
            raise RuntimeError(
                f"Network error while contacting NSE archive: {exc}"
            ) from exc

        if response.status_code == 200:

            if not response.content.strip():
                raise RuntimeError(
                    "NSE returned HTTP 200, but the bhavcopy was empty."
                )

            logging.info(
                "Bhavcopy downloaded successfully (%d bytes).",
                len(response.content),
            )

            return response.content

        if response.status_code == 404:

            if retries_used >= MAX_404_RETRIES:

                logging.warning(
                    "Bhavcopy still unavailable after %d retries. "
                    "This is likely a market holiday or NSE upload delay. "
                    "Exiting successfully without changing existing data.",
                    MAX_404_RETRIES,
                )

                return None

            retries_used += 1

            logging.warning(
                "Bhavcopy not found (HTTP 404). "
                "Retry %d/%d in %d seconds.",
                retries_used,
                MAX_404_RETRIES,
                RETRY_WAIT_SECONDS,
            )

            time.sleep(RETRY_WAIT_SECONDS)

            continue

        if response.status_code == 403:

            raise RuntimeError(
                "The NSE archive server itself returned HTTP 403. "
                "The homepage 403 is allowed, but the archive download "
                "must be accessible for this pipeline to work."
            )

        response.raise_for_status()


def clean_bhavcopy(
    raw_csv: bytes,
    trade_date_iso: str,
    root: Path,
) -> pd.DataFrame:

    df = pd.read_csv(
        io.BytesIO(raw_csv),
        dtype=str,
        skipinitialspace=True,
    )

    # Remove whitespace from column names.
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # Remove whitespace from text cells.
    for column in df.columns:

        if pd.api.types.is_object_dtype(df[column]):

            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
            )

    # NSE normally calls the date column DATE1.
    if "DATE1" in df.columns and "DATE" not in df.columns:
        df = df.rename(
            columns={"DATE1": "DATE"}
        )

    required_columns = set(
        HISTORICAL_COLUMNS
    ) - {"DATE"}

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Bhavcopy is missing required columns: "
            + ", ".join(missing_columns)
        )

    df["SYMBOL"] = (
        df["SYMBOL"]
        .str.strip()
        .str.upper()
    )

    df["SERIES"] = (
        df["SERIES"]
        .str.strip()
        .str.upper()
    )

    # Keep only normal equity shares.
    df = df.loc[
        df["SERIES"].eq("EQ")
    ].copy()

    logging.info(
        "Found %d regular EQ stocks.",
        len(df),
    )

    universe_path = (
        root
        / "universe"
        / "nifty_universe.csv"
    )

    if universe_path.exists():

        logging.info(
            "Universe file detected: %s",
            universe_path,
        )

        universe_df = pd.read_csv(
            universe_path,
            dtype=str,
            skipinitialspace=True,
        )

        universe_df.columns = (
            universe_df.columns
            .astype(str)
            .str.strip()
        )

        symbol_column = (
            "SYMBOL"
            if "SYMBOL" in universe_df.columns
            else universe_df.columns[0]
        )

        symbols = set(
            universe_df[symbol_column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )

        if not symbols:

            raise ValueError(
                "nifty_universe.csv exists but contains no symbols."
            )

        before_filter = len(df)

        df = df.loc[
            df["SYMBOL"].isin(symbols)
        ].copy()

        logging.info(
            "Universe filtering retained %d of %d EQ stocks.",
            len(df),
            before_filter,
        )

    else:

        logging.info(
            "No universe/nifty_universe.csv found. "
            "Exporting all EQ stocks."
        )

    if df.empty:

        raise ValueError(
            "No EQ stocks remain after filtering."
        )

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

        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    if df[PRICE_COLUMNS].isna().any().any():

        raise ValueError(
            "One or more required price values are missing."
        )

    # Force a consistent YYYY-MM-DD date.
    df["DATE"] = trade_date_iso

    df = (
        df[HISTORICAL_COLUMNS]
        .sort_values(
            "SYMBOL",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return df


def write_outputs(
    df: pd.DataFrame,
    trade_date: datetime,
    root: Path,
) -> tuple[Path, Path]:

    data_directory = root / "data"

    public_directory = root / "public"

    data_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    public_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_path = (
        data_directory
        / f"nifty_closing_{trade_date:%Y_%m_%d}.csv"
    )

    latest_path = (
        public_directory
        / "latest.json"
    )

    df.to_csv(
        archive_path,
        index=False,
        lineterminator="\n",
    )

    payload = []

    for row in df.itertuples(index=False):

        payload.append(
            {
                "symbol": row.SYMBOL,
                "close": float(row.CLOSE_PRICE),
                "open": float(row.OPEN_PRICE),
                "high": float(row.HIGH_PRICE),
                "low": float(row.LOW_PRICE),
                "prev_close": float(row.PREV_CLOSE),
                "date": row.DATE,
            }
        )

    with latest_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

        file.write("\n")

    return archive_path, latest_path


def determine_trade_date() -> datetime:

    override = os.getenv(
        "NSE_TRADE_DATE",
        "",
    ).strip()

    if override:

        try:
            parsed = datetime.strptime(
                override,
                "%Y-%m-%d",
            )

        except ValueError as exc:

            raise ValueError(
                "NSE_TRADE_DATE must use YYYY-MM-DD format. "
                "Example: 2026-09-04"
            ) from exc

        trade_date = datetime(
            parsed.year,
            parsed.month,
            parsed.day,
            17,
            30,
            tzinfo=IST,
        )

        logging.info(
            "Manual trade-date override selected: %s",
            trade_date.strftime("%Y-%m-%d"),
        )

        return trade_date

    return datetime.now(IST)


def main() -> int:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    trade_date = determine_trade_date()

    # Scheduled workflow never runs on weekends, but this prevents
    # accidental manual weekend runs from waiting for 15 minutes.
    if (
        not os.getenv("NSE_TRADE_DATE", "").strip()
        and trade_date.weekday() >= 5
    ):

        logging.info(
            "It is currently a weekend in India (%s). "
            "NSE cash market is closed. "
            "Exiting successfully.",
            trade_date.strftime("%Y-%m-%d"),
        )

        return 0

    nse_date = trade_date.strftime(
        "%d%m%Y"
    )

    trade_date_iso = trade_date.strftime(
        "%Y-%m-%d"
    )

    root = project_root()

    logging.info(
        "Target NSE trade date: %s",
        trade_date_iso,
    )

    session = create_nse_session()

    try:

        # We still perform the requested homepage handshake.
        # A homepage 403 is no longer fatal on cloud runners.
        handshake(session)

        raw_csv = download_bhavcopy(
            session,
            nse_date,
        )

        if raw_csv is None:
            return 0

        dataframe = clean_bhavcopy(
            raw_csv,
            trade_date_iso,
            root,
        )

        archive_path, latest_path = write_outputs(
            dataframe,
            trade_date,
            root,
        )

        logging.info(
            "Successfully wrote %d stocks.",
            len(dataframe),
        )

        logging.info(
            "Historical CSV: %s",
            archive_path,
        )

        logging.info(
            "Latest frontend JSON: %s",
            latest_path,
        )

        return 0

    finally:

        session.close()


if __name__ == "__main__":

    try:

        sys.exit(
            main()
        )

    except Exception:

        logging.exception(
            "NSE EOD pipeline failed."
        )

        sys.exit(1)
