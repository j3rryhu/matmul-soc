"""Select on calibration captures only, then evaluate held-out rows and model logits."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
from softmax_fixed import Config, make_lut, quantize_scores, softmax_fixed


def error(actual, expected):
    d = actual.astype(np.float64) - expected.astype(np.float64)
    return {"mae": float(np.abs(d).mean()), "rmse": float(np.sqrt(np.mean(d*d))),
            "max_absolute_error": float(np.abs(d).max())}


def evaluate(records, folder, config, export=None):
    rows = []
    for record in records:
        with np.load(folder / record["artifact"]) as data:
            p, codes, stats = softmax_fixed(data["scores"], config)
            context = p @ data["values"].astype(np.float64)
            rows.append({"image": record["image"], **stats,
                         "probability": error(p, data["pytorch_probabilities"]),
                         "context": error(context, data["pytorch_context"]),
                         "max_row_sum_error": float(np.abs(p.sum(-1)-1).max())})
            if export is not None:
                q, _ = quantize_scores(data["scores"], config)
                np.savez_compressed(export / record["artifact"], score_codes=q.astype(np.int16),
                                    probability_codes=codes.astype(np.uint16))
    # All captured tensors in this experiment have the same size.
    return {"mean_image_context_rmse": float(np.mean([r["context"]["rmse"] for r in rows])),
            "mean_image_probability_rmse": float(np.mean([r["probability"]["rmse"] for r in rows])),
            "max_probability_error": max(r["probability"]["max_absolute_error"] for r in rows),
            "max_context_error": max(r["context"]["max_absolute_error"] for r in rows),
            "max_row_sum_error": max(r["max_row_sum_error"] for r in rows),
            "input_clipped_count": sum(r["input_clipped_count"] for r in rows), "images": rows}


def evaluate_model(args, capture, config):
    import torch
    from PIL import Image
    from torchvision import transforms
    from capture_reference import sha256
    source = args.model_root / "tiny-vit/models/tiny_vit.py"
    if sha256(source) != capture["model_source_sha256"] or sha256(Path(capture["checkpoint"])) != capture["checkpoint_sha256"]:
        raise ValueError("Model source or checkpoint differs from capture")
    sys.path.insert(0, str(args.model_root / "tiny-vit"))
    from models.tiny_vit import tiny_vit_21m_224
    torch.set_num_threads(2)
    model = tiny_vit_21m_224(pretrained=False).float()
    state = torch.load(capture["checkpoint"], map_location="cpu", weights_only=True)["model"]
    model.load_state_dict({k:v for k,v in state.items() if not k.endswith("attention_bias_idxs")}, strict=True)
    model.eval()
    attention = model.get_submodule(capture["layer"])
    transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    rows, baseline_logits, modified_logits = [], [], []
    cache = {}

    def qkv_hook(module, inputs, output):
        cache["qkv"] = output

    def replace_attention(module, inputs, output):
        b, n, _ = cache["qkv"].shape
        q, k, v = cache["qkv"].view(b,n,module.num_heads,-1).split([module.key_dim,module.key_dim,module.d],dim=3)
        q, k, v = (x.permute(0,2,1,3) for x in (q,k,v))
        scores = (q @ k.transpose(-2,-1))*module.scale + module.ab
        p, _, _ = softmax_fixed(scores.numpy(), config)
        probabilities = torch.from_numpy(p.astype(np.float32))
        return module.proj((probabilities @ v).transpose(1,2).reshape(b,n,module.dh))

    with torch.inference_mode():
        for record in capture["images"]:
            path = args.model_root / "Testing_Image" / record["image"]
            if sha256(path) != record["sha256"]:
                raise ValueError("Image differs from capture")
            with Image.open(path) as im:
                batch = transform(im.convert("RGB")).unsqueeze(0)
            baseline = model(batch).numpy().copy()
            with np.load(args.reference / record["artifact"]) as data:
                np.testing.assert_array_equal(baseline, data["baseline_logits"])
            handles = [attention.qkv.register_forward_hook(qkv_hook), attention.register_forward_hook(replace_attention)]
            try:
                modified = model(batch).numpy().copy()
            finally:
                for handle in handles:
                    handle.remove()
            np.testing.assert_array_equal(model(batch).numpy(), baseline)
            row = {"image": record["image"], "split": record["split"], "logit_error": error(modified,baseline),
                   "baseline_class": int(baseline.argmax()), "modified_class": int(modified.argmax()),
                   "top1_agrees": bool(baseline.argmax()==modified.argmax()),
                   "top5_set_agrees": bool(set(baseline[0].argsort()[-5:])==set(modified[0].argsort()[-5:]))}
            rows.append(row)
            baseline_logits.append(baseline[0]); modified_logits.append(modified[0])
            print("Model", record["image"], "top1 agrees:", row["top1_agrees"], flush=True)
    np.savez_compressed(args.out / "logits.npz", baseline=np.stack(baseline_logits), modified=np.stack(modified_logits))
    return {"restoration_bit_exact": True, "images": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--reference", type=Path, default=root / "runs/reference_21m")
    parser.add_argument("--out", type=Path, default=root / "runs/fixed_21m")
    parser.add_argument("--model-root", type=Path, default=root.parents[1] / "TinyVit-on-FPGA")
    args = parser.parse_args()
    capture = json.loads((args.reference / "report.json").read_text())
    calibration = [r for r in capture["images"] if r["split"]=="calibration"]
    validation = [r for r in capture["images"] if r["split"]=="validation"]
    args.out.mkdir(parents=True, exist_ok=True)
    grid = []
    for fractional in (4,6,8):
        for cutoff in (8,12,16):
            config = Config(fractional_bits=fractional,cutoff=cutoff)
            metrics = evaluate(calibration,args.reference,config)
            grid.append({"config":asdict(config), "calibration":metrics})
            print("Calibration", fractional, cutoff, metrics["mean_image_context_rmse"],flush=True)
    # The selection uses calibration only; held-out images are loaded afterwards.
    selected = min(grid,key=lambda x:(x["calibration"]["mean_image_context_rmse"],
                                     x["config"]["cutoff"]*(1<<x["config"]["fractional_bits"])))
    config = Config(**selected["config"])
    report = {"selection_policy":"Minimum mean per-image P@V RMSE on calibration only; ROM size breaks exact ties",
              "config":config.metadata(), "calibration_grid":grid,
              "calibration":evaluate(calibration,args.reference,config,args.out),
              "validation":evaluate(validation,args.reference,config,args.out),
              "limitations":["One block, 10 images; agreement is not ground-truth accuracy",
                             "Integer division is a baseline, not a reciprocal implementation",
                             "Float scores are quantized at this boundary; integer QK matmul is not simulated",
                             "No hardware validation or acceptance thresholds"]}
    lut = make_lut(config)
    (args.out / "exp_lut.hex").write_text("".join(f"{int(x):04x}\n" for x in lut))
    (args.out / "config.json").write_text(json.dumps(config.metadata(),indent=2)+"\n")
    report["model"] = evaluate_model(args,capture,config)
    (args.out / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("PASS: fixed-point evaluation and model restoration",flush=True)


if __name__ == "__main__":
    main()
