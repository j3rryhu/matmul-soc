"""Capture one local TinyViT attention block without modifying its forward method."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from softmax_reference import softmax_reference


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errors(actual, expected):
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return {"mae": float(np.abs(delta).mean()),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "max_absolute_error": float(np.abs(delta).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path(__file__).resolve().parents[2] / "TinyVit-on-FPGA")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--layer", default="layers.1.blocks.0.attn")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs/reference_21m")
    args = parser.parse_args()
    checkpoint = args.checkpoint or args.model_root / "model_checkpoints/tiny_vit_21m_22kto1k_distill.pth"
    image_dir = args.images or args.model_root / "Testing_Image"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if len(images) < 2:
        raise ValueError("At least two images are required for separate calibration and validation sets")
    hashes = [sha256(p) for p in images]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate image contents would leak across the split")
    sys.path.insert(0, str(args.model_root / "tiny-vit"))
    from models.tiny_vit import tiny_vit_21m_224
    import timm

    torch.set_num_threads(2)
    model = tiny_vit_21m_224(pretrained=False).float()
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")["model"]
    model.load_state_dict({k: v for k, v in state.items() if not k.endswith("attention_bias_idxs")}, strict=True)
    # Populate the model's cached evaluation attention biases AFTER loading weights.
    model.eval()
    attention = model.get_submodule(args.layer)
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    args.out.mkdir(parents=True, exist_ok=True)
    split = min(len(images) - 1, max(1, int(len(images) * .7)))
    records, logits, score_sets = [], [], {"calibration": [], "validation": []}
    for index, image in enumerate(images):
        with Image.open(image) as im:
            batch = transform(im.convert("RGB")).unsqueeze(0)
        captured = {}

        def qkv_hook(module, inputs, output):
            if "qkv" in captured:
                raise RuntimeError("Expected one target attention invocation per image")
            captured["qkv"] = output.detach().clone()

        def output_hook(module, inputs, output):
            captured["output"] = output.detach().clone()

        with torch.inference_mode():
            baseline = model(batch).clone()
            handles = [attention.qkv.register_forward_hook(qkv_hook), attention.register_forward_hook(output_hook)]
            try:
                observed = model(batch)
            finally:
                for handle in handles:
                    handle.remove()
            torch.testing.assert_close(observed, baseline, rtol=0, atol=0)
            b, n, _ = captured["qkv"].shape
            q, k, v = captured["qkv"].view(b, n, attention.num_heads, -1).split(
                [attention.key_dim, attention.key_dim, attention.d], dim=3)
            q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
            scores = (q @ k.transpose(-2, -1)) * attention.scale + attention.ab
            probabilities = scores.softmax(dim=-1)
            context = probabilities @ v
            reconstructed = attention.proj(context.transpose(1, 2).reshape(b, n, attention.dh))
            torch.testing.assert_close(reconstructed, captured["output"], rtol=0, atol=0)
        scores_np, probs_np, v_np, context_np = (t.numpy().copy() for t in (scores, probabilities, v, context))
        reference = softmax_reference(scores_np)
        reference_context = reference @ v_np.astype(np.float64)
        np.testing.assert_allclose(reference, probs_np, rtol=2e-5, atol=2e-7)
        np.testing.assert_allclose(reference_context, context_np, rtol=2e-5, atol=2e-5)
        subset = "calibration" if index < split else "validation"
        score_sets[subset].append(scores_np.reshape(-1))
        artifact = f"{index:03d}_{image.stem}_reference.npz"
        np.savez_compressed(args.out / artifact, scores=scores_np, pytorch_probabilities=probs_np,
                            values=v_np, pytorch_context=context_np, reference_probabilities=reference,
                            baseline_logits=baseline.numpy())
        record = {"image": image.name, "sha256": hashes[index], "split": subset,
                  "artifact": artifact, "scores_shape": list(scores.shape), "values_shape": list(v.shape),
                  "row_length": n, "score_min": float(scores_np.min()), "score_max": float(scores_np.max()),
                  "shifted_score_min": float((scores_np - scores_np.max(axis=-1, keepdims=True)).min()),
                  "probability_error": errors(reference, probs_np), "context_error": errors(reference_context, context_np),
                  "pytorch_max_row_sum_error": float(np.abs(probs_np.astype(np.float64).sum(axis=-1) - 1).max()),
                  "reference_max_row_sum_error": float(np.abs(reference.sum(axis=-1) - 1).max()),
                  "baseline_class": int(baseline.argmax(dim=-1).item()),
                  "hooked_logits_bit_exact": True, "reconstructed_attention_bit_exact": True}
        records.append(record)
        logits.append(baseline.numpy()[0])
        print(f"Captured {image.name}: {list(scores.shape)}, max probability error={record['probability_error']['max_absolute_error']:.3g}", flush=True)
    ranges = {}
    for subset, arrays in score_sets.items():
        values = np.concatenate(arrays)
        ranges[subset] = {"min": float(values.min()), "max": float(values.max()),
                          "percentiles": dict(zip(["0.1", "1", "50", "99", "99.9"], map(float, np.percentile(values, [.1, 1, 50, 99, 99.9]))))}
    report = {"model": "tiny_vit_21m_224", "layer": args.layer, "checkpoint": str(checkpoint.resolve()),
              "checkpoint_sha256": sha256(checkpoint), "model_source_sha256": sha256(args.model_root / "tiny-vit/models/tiny_vit.py"),
              "torch_version": torch.__version__, "numpy_version": np.__version__, "timm_version": timm.__version__,
              "device": "cpu", "model_dtype": "float32", "reference_dtype": "float64", "transform": str(transform),
              "capture_method": "Reconstruct internals from actual qkv hook; verify projected attention and final logits bit-exact",
              "ranges": ranges, "images": records,
              "limitations": ["Small local image set; not an accuracy benchmark", "One attention block only",
                              "No fixed-point approximation or FPGA integration", "No ground-truth labels evaluated"]}
    np.savez_compressed(args.out / "logits.npz", baseline=np.stack(logits), image_names=np.array([p.name for p in images]))
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS: floating-point reference, attention reconstruction, and unchanged logits", flush=True)


if __name__ == "__main__":
    main()
