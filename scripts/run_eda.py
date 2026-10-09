"""Generate EDA figures and summary into reports/.  Usage: python scripts/run_eda.py"""
import json

from heartrisk import data, eda

if __name__ == "__main__":
    info = eda.run(data.load_clean())
    print(json.dumps(info, indent=2))
