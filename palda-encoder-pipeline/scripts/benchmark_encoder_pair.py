#!/usr/bin/env python3
"""Compare warm runtime encoder latency on identical product-grouped Validation input."""
import argparse
import asyncio
import gc
import json
import statistics
import sys
import time
from pathlib import Path

from palda_encoder.io import iter_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-src", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime_src))
    import torch
    from palda_runtime.models import ServiceEncoderModel

    groups = {}
    rows = list(iter_jsonl(args.validation))
    for row in rows:
        key = (row["product_id"], row["category_key"])
        request = groups.setdefault(key, {
            "product": {"productId": key[0], "categoryPath": key[1]},
            "reviewSample": {"items": []}, "qnaSample": {"items": []},
        })
        request["reviewSample" if row["record_type"] == "review" else "qnaSample"]["items"].append({"id": row["sample_id"], "text": row["text"]})
    requests = [groups[k] for k in sorted(groups)]
    report = {"rows": len(rows), "products": len(groups), "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "batchSize": 64, "maxLength": 256,
              "scope": "warm encoder runtime only; fixed product-grouped Validation; not E2E/SLO", "timing": {}}
    for name, path in (("base", args.base), ("candidate", args.candidate)):
        model = ServiceEncoderModel(path, path / "taxonomy.json", device="cuda:0")
        async def run():
            count = 0
            for request in requests:
                count += len(await model.classify(request))
            assert count == len(rows)
        asyncio.run(run())
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            asyncio.run(run())
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        median = statistics.median(times)
        report["timing"][name] = {"seconds": times, "p50Seconds": median, "itemsPerSecond": len(rows) / median}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    ratio = report["timing"]["candidate"]["itemsPerSecond"] / report["timing"]["base"]["itemsPerSecond"]
    report.update(throughputRatio=ratio, passes20PercentRegressionGuard=ratio >= .8)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
