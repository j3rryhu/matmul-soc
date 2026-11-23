"""Prepare matched TinyViT validation inputs or report probability errors."""

import argparse
import json
from pathlib import Path

import numpy as np

from softmax_fixed import Config, quantize_scores, softmax_codes
from softmax_reference import softmax_reference


ROOT = Path(__file__).resolve().parent


def load_validation(captures):
    manifest = json.loads((captures / "report.json").read_text())
    cases = []
    for image in manifest["images"]:
        if image["split"] == "validation":
            with np.load(captures / image["artifact"]) as capture:
                cases.append(
                    tuple(capture[key] for key in ("scores", "pytorch_probabilities", "values"))
                )
    if not cases:
        raise ValueError("No validation images in the capture manifest")
    return manifest, tuple(np.stack(tensors) for tensors in zip(*cases))


def errors(decoded, reference):
    reference = reference.astype(np.float64, copy=False)
    delta = decoded - reference
    return {
        "relative_rmse_pct": float(100 * np.linalg.norm(delta) / np.linalg.norm(reference)),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mean_absolute_error": float(np.mean(np.abs(delta))),
        "max_absolute_error": float(np.max(np.abs(delta))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "report"))
    parser.add_argument("--captures", type=Path, default=ROOT / "runs/reference_21m")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/benchmark")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest, (scores, fp32_probabilities, values) = load_validation(args.captures)
    config = Config(json.loads((ROOT / "example/int8_config.json").read_text())["score_scale"])
    quantized, clipped = quantize_scores(scores, config)
    length = quantized.shape[-1]
    rows = quantized.reshape(-1, length)
    if args.action == "prepare":
        np.savetxt(args.out / "scores.txt", rows, fmt="%d",
                   header=f"{len(rows)} {length}", comments="")
        return

    numbers = np.fromfile(args.out / "cpu_vectors.txt", dtype=np.int32, sep=" ")
    if numbers.size != 2 + len(rows) * (1 + 2 * length):
        raise ValueError("Unexpected C output vector count")
    if tuple(numbers[:2]) != (len(rows), rows.size):
        raise ValueError("C output header does not match the validation set")
    vectors = numbers[2:].reshape(len(rows), 1 + 2 * length)
    if not np.all(vectors[:, 0] == length) or not np.array_equal(vectors[:, 1::2], rows):
        raise ValueError("C output scores do not match the validation set")
    cpu_codes = vectors[:, 2::2].reshape(quantized.shape)
    lut = np.array([int(word, 16) for word in (ROOT / "example/exp_lut.hex").read_text().split()])
    expected_codes, _ = softmax_codes(quantized, config, lut)
    mismatches = int(np.count_nonzero(expected_codes != cpu_codes))
    if mismatches:
        raise ValueError(f"C output mismatched the Python INT8 oracle on {mismatches} codes")

    decoded = cpu_codes.astype(np.float64) / 127
    fp_same_input = softmax_reference(quantized.astype(np.float64) * config.score_scale)
    report = {
        "model": manifest["model"],
        "layer": manifest["layer"],
        "validation_images": scores.shape[0],
        "rows": len(rows),
        "keys_per_row": length,
        "probability_codes": rows.size,
        "clipped_input_scores": clipped,
        "clipped_input_scores_pct": 100 * clipped / rows.size,
        "c_vs_python_code_mismatches": mismatches,
        "probabilities_vs_original_fp32_input": errors(decoded, fp32_probabilities),
        "probabilities_vs_float_softmax_on_same_int8_input": errors(decoded, fp_same_input),
        "probability_row_sum_mean_abs_error": float(np.mean(np.abs(decoded.sum(axis=-1) - 1))),
        "pv_vs_original_fp32_probabilities": errors(
            decoded @ values, fp32_probabilities.astype(np.float64) @ values
        ),
    }
    output = json.dumps(report, indent=2)
    (args.out / "report.json").write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
