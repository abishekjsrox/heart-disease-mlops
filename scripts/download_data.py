"""Download the UCI Heart Disease (Cleveland) data and write the cleaned CSV.

Usage:  python scripts/download_data.py [--force]
"""
import argparse
import logging

from heartrisk import data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="re-download even if present")
    args = p.parse_args()
    data.download(force=args.force)
    df = data.make_clean_dataset()
    print(f"clean rows={len(df)} missing={int(df.isna().sum().sum())}")
