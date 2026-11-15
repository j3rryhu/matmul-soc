"""Validate all local TinyViT LayerNorms; test a standalone fixed replacement.

Reads existing local model assets only. Never downloads or edits model code.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from layernorm_reference import layernorm_reference
from layernorm_fixed import Config, layernorm_fixed, quantize_int8


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errors(actual, expected):
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return {'mae': float(np.abs(delta).mean()), 'rmse': float(np.sqrt(np.mean(delta**2))),
            'max_absolute_error': float(np.abs(delta).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, default=Path(__file__).resolve().parents[2] / 'TinyVit-on-FPGA')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--images', type=Path)
    parser.add_argument('--layer', default='layers.1.blocks.0.attn.norm')
    parser.add_argument('--config', type=Path, help='JSON with the four calibrated INT8 scales')
    parser.add_argument('--input-scale', type=float)
    parser.add_argument('--output-scale', type=float)
    parser.add_argument('--gamma-scale', type=float)
    parser.add_argument('--beta-scale', type=float)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'runs/int8_21m')
    args = parser.parse_args()
    scale_values = asdict(Config())
    if args.config is not None:
        loaded = json.loads(args.config.read_text(encoding='utf-8'))
        if not isinstance(loaded, dict) or set(loaded) != set(scale_values):
            raise ValueError('Config must contain exactly input/output/gamma/beta_scale')
        scale_values.update(loaded)
    for key in scale_values:
        if getattr(args, key) is not None:
            scale_values[key] = getattr(args, key)
    config = Config(**scale_values)
    checkpoint = args.checkpoint or args.model_root / 'model_checkpoints/tiny_vit_21m_22kto1k_distill.pth'
    image_dir = args.images or args.model_root / 'Testing_Image'
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    if not images:
        raise ValueError('No local test images found')
    hashes = [sha256(p) for p in images]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate image contents')
    sys.path.insert(0, str(args.model_root / 'tiny-vit'))
    from models.tiny_vit import tiny_vit_21m_224
    import timm

    torch.set_num_threads(2)
    model = tiny_vit_21m_224(pretrained=False).float()
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')['model']
    model.load_state_dict({k: v for k, v in state.items() if not k.endswith('attention_bias_idxs')}, strict=True)
    model.eval()  # Build evaluation bias caches after loading the checkpoint.
    norms = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.LayerNorm)}
    if args.layer not in norms:
        raise ValueError(f'Target must be a LayerNorm; available: {list(norms)}')
    if any(len(m.normalized_shape) != 1 for m in norms.values()):
        raise ValueError('Only channel-vector normalization is supported')
    target = norms[args.layer]
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    args.out.mkdir(parents=True, exist_ok=True)
    records, baseline_logits, replacement_logits = [], [], []
    for index, image in enumerate(images):
        with Image.open(image) as im:
            batch = transform(im.convert('RGB')).unsqueeze(0)
        captured = {}

        def hook_for(name):
            def hook(module, inputs, output):
                if name in captured:
                    raise RuntimeError(f'Unexpected repeated LayerNorm invocation: {name}')
                captured[name] = (inputs[0].detach().numpy().copy(), output.detach().numpy().copy())
            return hook

        with torch.inference_mode():
            baseline = model(batch).clone()
            handles = []
            try:
                for name, module in norms.items():
                    handles.append(module.register_forward_hook(hook_for(name)))
                observed = model(batch)
            finally:
                for handle in handles:
                    handle.remove()
            torch.testing.assert_close(observed, baseline, rtol=0, atol=0)
            if captured.keys() != norms.keys():
                raise AssertionError('Not every LayerNorm executed')
            layer_records = []
            for name, (x, y) in captured.items():
                module = norms[name]
                gamma, beta = module.weight.detach().numpy(), module.bias.detach().numpy()
                reference = layernorm_reference(x, gamma, beta, module.eps)
                np.testing.assert_allclose(reference, y, atol=5e-5, rtol=3e-5)
                layer_records.append({'layer': name, 'shape': list(x.shape), 'eps': module.eps,
                    'vectors': int(x.size // x.shape[-1]), 'input_min': float(x.min()),
                    'input_max': float(x.max()), 'reference_error': errors(reference, y)})
            x, y = captured[args.layer]
            gamma, beta = target.weight.detach().numpy(), target.bias.detach().numpy()
            fixed, codes, stats = layernorm_fixed(x, gamma, beta, config, target.eps)
            reference = layernorm_reference(x, gamma, beta, target.eps)
            q, _ = quantize_int8(x, config.input_scale)
            gq, _ = quantize_int8(gamma, config.gamma_scale)
            bq, _ = quantize_int8(beta, config.beta_scale)
            quantized_reference = layernorm_reference(q.astype(float)*config.input_scale,
                gq.astype(float)*config.gamma_scale, bq.astype(float)*config.beta_scale, target.eps)
            ideal_codes, ideal_clipped = quantize_int8(quantized_reference, config.output_scale)
            code_difference = codes.astype(int)-ideal_codes.astype(int)
            fixed_tensor = torch.from_numpy(fixed).to(dtype=torch.float32)

            def replacement(module, inputs, output):
                if tuple(output.shape) != fixed.shape:
                    raise AssertionError('Replacement shape mismatch')
                return fixed_tensor

            handle = target.register_forward_hook(replacement)
            try:
                replaced = model(batch).clone()
            finally:
                handle.remove()
            torch.testing.assert_close(model(batch), baseline, rtol=0, atol=0)
        artifact = f'{index:03d}_{image.stem}.npz'
        np.savez_compressed(args.out / artifact, inputs=x, gamma=gamma, beta=beta,
            pytorch_output=y, reference_output=reference, input_codes=q, gamma_codes=gq,
            beta_codes=bq, output_codes=codes, ideal_output_codes=ideal_codes,
            quantized_input_reference=quantized_reference, fixed_output=fixed,
            baseline_logits=baseline.numpy(), replacement_logits=replaced.numpy())
        record = {'image': image.name, 'sha256': hashes[index], 'artifact': artifact,
            'layers': layer_records, 'hooked_logits_bit_exact': True, 'restored_logits_bit_exact': True,
            'fixed_error': errors(fixed, reference), 'kernel_error_on_quantized_inputs': errors(fixed, quantized_reference),
            'input_affine_quantization_error': errors(quantized_reference, reference),
            'code_error_vs_ideal_int8': {'max_absolute_codes': int(np.abs(code_difference).max()),
                'different_count': int(np.count_nonzero(code_difference)), 'scalar_count': int(codes.size),
                'ideal_output_saturated_count': ideal_clipped},
            'stats': stats, 'logit_error': errors(replaced.numpy(), baseline.numpy()),
            'baseline_class': int(baseline.argmax(-1).item()), 'replacement_class': int(replaced.argmax(-1).item())}
        records.append(record)
        baseline_logits.append(baseline.numpy()[0])
        replacement_logits.append(replaced.numpy()[0])
        print(f'{image.name}: {len(layer_records)} reference checks PASS; fixed RMSE={record["fixed_error"]["rmse"]:.6g}', flush=True)
    report = {'model': 'tiny_vit_21m_224', 'fixed_target': args.layer,
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': sha256(checkpoint),
        'model_source_sha256': sha256(args.model_root / 'tiny-vit/models/tiny_vit.py'),
        'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'timm': timm.__version__},
        'transform': str(transform), 'config': config.metadata(target.eps), 'images': records,
        'config_file': str(args.config.resolve()) if args.config is not None else None,
        'limitations': ['INT8 storage/scale convention matches matmul/GELU; scales are explicit and target-specific',
            'Unlabeled local images; top-1 agreement is not classification accuracy',
            'Fixed replacement at one LayerNorm only; all other operations remain float32',
            'Inputs quantized at the upstream INT8 boundary; not actual matmul/GELU hardware captures',
            'INT8 gamma/beta parameters, with explicitly configured scales',
            'Exact integer sqrt/division baseline with wide intermediates, no RTL or FPGA timing validation',
            'No integration with matmul, GELU or softmax hardware']}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    np.savez_compressed(args.out / 'logits.npz', baseline=np.stack(baseline_logits),
        replacement=np.stack(replacement_logits), image_names=np.array([p.name for p in images]))
    print('PASS: all LayerNorm references, unchanged capture logits and restored model', flush=True)


if __name__ == '__main__':
    main()
