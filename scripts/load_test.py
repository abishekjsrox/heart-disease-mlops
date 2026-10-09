"""Send a stream of realistic requests to the API so the dashboards have data.

Usage:  python scripts/load_test.py --url http://localhost:8000 --n 300
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request

from heartrisk import data


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--delay", type=float, default=0.1)
    args = p.parse_args()
    rows = data.load_clean().drop(columns=["target"])
    floats = {"trestbps", "chol", "thalach", "oldpeak"}
    records = [{k: (None if v != v else (float(v) if k in floats else int(v)))
                for k, v in r.items()} for r in rows.to_dict("records")]
    codes = {}
    for i in range(args.n):
        rec = dict(random.choice(records))
        if random.random() < 0.05:          # a few deliberately invalid requests
            rec["age"] = 150
        code = post(f"{args.url}/predict", rec)
        codes[code] = codes.get(code, 0) + 1
        time.sleep(args.delay)
    print("status counts:", codes)
