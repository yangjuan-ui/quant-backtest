"""13F quarterly holdings (supplementary reference layer) via SEC EDGAR.

Honest scope, read carefully:
  * 13F-HR filings are QUARTERLY with up to a 45-day reporting lag. This
    module is a *reference layer*, not a daily signal - it is never mixed
    into the daily universe at daily frequency.
  * Holdings are stored with issuer name + CUSIP + value + shares. There is
    no reliable free CUSIP->ticker mapping, so this module does NOT invent
    tickers; instead it reports overlap between 13F issuers and the daily
    traded universe by normalized name (best-effort, informational only).
  * Free, no API key. SEC asks for a descriptive User-Agent: set SEC_CONTACT
    to your own email in config below (or via the SEC_CONTACT env var).

Data flow per fund CIK:
  data.sec.gov/submissions/CIK{cik}.json -> latest 13F-HR accession ->
  Archives/edgar index.json -> information-table XML -> parse <infoTable>.
"""
from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET

import requests

log = logging.getLogger(__name__)

SEC_CONTACT = os.environ.get("SEC_CONTACT", "you@example.com")
SEC_HEADERS = {
    "User-Agent": f"quant-backtest/1.0 (contact: {SEC_CONTACT})",
    "Accept": "application/json",
    "Host": "data.sec.gov",
}
SEC_DELAY = 0.6  # be polite to EDGAR

# Funds to track: {display name: CIK}. Add any 13F filer you follow
# (find CIKs via https://www.sec.gov/edgar/searchedgar/companysearch).
FUND_CIKS = {
    "Berkshire Hathaway": "0001067983",
}


def _sec_get(url: str, timeout: int = 30) -> requests.Response:
    headers = dict(SEC_HEADERS)
    if "data.sec.gov" not in url:
        headers.pop("Host", None)
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    time.sleep(SEC_DELAY)
    return r


def get_latest_13f_hr(cik: str) -> dict | None:
    """Latest 13F-HR filing metadata for a CIK, or None."""
    cik10 = cik.zfill(10)
    data = _sec_get(f"https://data.sec.gov/submissions/CIK{cik10}.json").json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    for i, form in enumerate(forms):
        if form == "13F-HR":
            return {
                "accession": recent["accessionNumber"][i].replace("-", ""),
                "filing_date": recent["filingDate"][i],
                "report_period": recent["reportDate"][i],
            }
    return None


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_info_table_xml(cik: str, accession: str) -> bytes:
    cik_num = str(int(cik))  # Archives URLs use CIK without leading zeros
    idx = _sec_get(
        f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{accession}/index.json"
    ).json()
    items = idx.get("directory", {}).get("item", [])
    xml_names = [it["name"] for it in items
                 if str(it.get("name", "")).lower().endswith(".xml")]
    # prefer names hinting at the information table
    xml_names.sort(key=lambda n: (0 if "info" in n.lower() else 1, n))
    for name in xml_names:
        url = (f"https://www.sec.gov/Archives/edgar/data/"
               f"{cik_num}/{accession}/{name}")
        content = _sec_get(url).content
        if b"infoTable" in content:
            log.info("13F info table: %s", name)
            return content
    raise RuntimeError("information-table XML not found in filing index")


def parse_info_table(xml_bytes: bytes) -> list[dict]:
    """Parse <infoTable> entries -> issuer/cusip/value/shares."""
    root = ET.fromstring(xml_bytes)
    holdings = []
    for elem in root.iter():
        if _localname(elem.tag) != "infoTable":
            continue
        fields: dict[str, str] = {}
        for child in elem:
            fields[_localname(child.tag)] = (child.text or "").strip()
            for grand in child:  # e.g. shrsOrPrnAmt/sshPrnamt
                fields[_localname(grand.tag)] = (grand.text or "").strip()
        try:
            value_usd = float(fields.get("value", "0") or 0) * 1000.0
        except ValueError:
            value_usd = 0.0
        try:
            shares = float(fields.get("sshPrnamt", "0").replace(",", "") or 0)
        except ValueError:
            shares = 0.0
        holdings.append({
            "issuer": fields.get("nameOfIssuer", ""),
            "cusip": fields.get("cusip", ""),
            "value_usd": value_usd,
            "shares": shares,
        })
    return holdings


def fetch_13f_holdings(fund_ciks: dict[str, str] | None = None) -> list[dict]:
    """Latest 13F-HR holdings for each configured fund.

    Returns a flat list of dicts with fund/cik/filing_date/report_period/
    issuer/cusip/value_usd/shares. Failures per fund are logged, not fatal.
    """
    fund_ciks = fund_ciks or FUND_CIKS
    all_holdings: list[dict] = []
    for fund, cik in fund_ciks.items():
        try:
            meta = get_latest_13f_hr(cik)
            if not meta:
                log.warning("%s (%s): no 13F-HR filing found", fund, cik)
                continue
            xml = _find_info_table_xml(cik, meta["accession"])
            rows = parse_info_table(xml)
            for r in rows:
                r.update({"fund": fund, "cik": cik,
                          "filing_date": meta["filing_date"],
                          "report_period": meta["report_period"]})
            log.info("%s: %d holdings (filed %s, period %s)", fund, len(rows),
                     meta["filing_date"], meta["report_period"])
            all_holdings.extend(rows)
        except Exception as e:
            log.warning("%s (%s) 13F fetch failed: %s", fund, cik, e)
    # biggest positions first
    all_holdings.sort(key=lambda h: h["value_usd"], reverse=True)
    return all_holdings


def _norm_name(s: str) -> str:
    return "".join(ch for ch in s.upper() if ch.isalnum())


def overlap_with_universe(holdings: list[dict],
                          universe_names: dict[str, str]) -> list[dict]:
    """Best-effort overlap: 13F issuer names vs {normalized_name: symbol}.

    Informational only - name matching is fuzzy by nature and is reported
    as a reference, never as a trading signal.
    """
    index = {_norm_name(v): k for k, v in universe_names.items()}
    hits = []
    for h in holdings:
        key = _norm_name(h["issuer"])
        if key in index:
            hits.append({**h, "matched_symbol": index[key]})
    return hits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    hs = fetch_13f_holdings()
    print(f"13F holdings: {len(hs)} rows")
    for h in hs[:10]:
        print(f"  {h['issuer'][:40]:40s} ${h['value_usd']:>14,.0f}")
