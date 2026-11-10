"""Evaluate saved RECIST predictions using original NPZ labels and spacing."""

import argparse
import json
import statistics
from pathlib import Path

import nibabel as nib
import numpy as np

from src.utils.recist_metrics import score_case


def load_prediction(directory, case):
    nifti = directory / f"{case}.nii.gz"
    if nifti.is_file():
        # Inference stores ZYX arrays with identity affines. Do not infer spacing
        # or reorient the array from that affine; the original NPZ is authoritative.
        return np.asanyarray(nib.load(nifti).dataobj)
    npz = directory / f"{case}.npz"
    if npz.is_file():
        with np.load(npz, allow_pickle=False) as data:
            return data["segs"]
    raise FileNotFoundError(f"Missing prediction for {case}: expected {nifti} or {npz}")


def evaluate(inputs, predictions, cases=None):
    paths = [inputs / f"{case}.npz" for case in cases] if cases else sorted(inputs.glob("*.npz"))
    if not paths:
        raise ValueError(f"No reference NPZ files found in {inputs}")
    if len(set(paths)) != len(paths):
        raise ValueError("Each case must be evaluated once")
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            gt, spacing = data["gts"], data["spacing"]
        result = score_case(gt, load_prediction(predictions, path.stem), spacing)
        records.append({"case": path.stem, **result})

    summary = {}
    for metric in ("dsc", "nsd"):
        values = [record["means"][metric] for record in records]
        summary[metric] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        }
    return {
        "protocol": {
            "score_units": "fraction",
            "aggregation": "equal lesion weights within cases; equal case weights across the dataset",
            "nsd_tolerance_mm": 2.0,
            "nsd_minimum_lesion_dsc": 0.2,
            "spacing": "original NPZ XYZ spacing reordered to ZYX array order",
            "prediction_format": "ZYX arrays from this repository; NIfTI affines are not used for spacing",
            "missing_lesions": "zero scores",
            "prediction_only_ids": "ignored",
            "empty_reference": "rejected",
        },
        "n_cases": len(records),
        "n_lesions": sum(record["n_lesions"] for record in records),
        "summary": summary,
        "cases": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Original NPZ files containing gts and spacing")
    parser.add_argument("--predictions", type=Path, required=True, help="Saved .nii.gz or .npz predictions")
    parser.add_argument("--output", type=Path, required=True, help="JSON result file")
    parser.add_argument("--case", action="append", help="Optional case ID; repeat to select multiple cases")
    args = parser.parse_args()
    result = evaluate(args.inputs, args.predictions, args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Evaluated {result['n_cases']} cases and {result['n_lesions']} lesions")
    for metric, values in result["summary"].items():
        print(f"{metric.upper()}: {100 * values['mean']:.1f}%")
    print(f"Saved per-lesion, per-case and aggregate results to {args.output}")


if __name__ == "__main__":
    main()
