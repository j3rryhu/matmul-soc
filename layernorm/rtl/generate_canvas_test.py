"""Build padded Avalon images and goldens for tb_layernorm_canvas."""
import json
from pathlib import Path
import sys

import numpy as np

LAYER_NORM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAYER_NORM))
from layernorm_fixed import Config, layernorm_codes


def write_words(path, data):
    words = np.asarray(data, dtype=np.uint8).reshape(-1, 16)
    path.write_text("".join(bytes(word[::-1]).hex() + "\n" for word in words))


def main():
    work = LAYER_NORM / "rtl" / "canvas_vectors"
    work.mkdir(exist_ok=True)
    channels, contraction, tokens = 17, 16, 17
    k, n, m = np.arange(channels), np.arange(contraction), np.arange(tokens)
    weights = 4 * (k[:, None] % contraction == n)
    activations = (29 * n[:, None] + 13 * m) % 128 - 64
    bias = 4 * (k % 7 - 3)
    gamma = 24 + (17 * k) % 93
    beta = (19 * k) % 61 - 30

    cfg = Config(**json.loads((LAYER_NORM / "example/int8_config.json").read_text()))
    lut_file = LAYER_NORM.parent / "activation/example/gelu_int8.hex"
    lut = np.array([int(v, 16) for v in lut_file.read_text().split()], dtype=np.uint8)
    codes = np.arange(-128, 128)
    ratio = (1 / 32) / cfg.input_scale
    np.testing.assert_array_equal(
        np.rint(codes * ratio),
        np.sign(codes) * ((np.abs(codes) * 14249 + 32768) >> 16),
    )

    def pad(data):
        return np.pad(data, ((0, 0), (0, 32 - tokens)))

    write_words(work / "canvas_weights.hex", weights)
    write_words(work / "canvas_input.hex", pad(activations))
    bias_words = np.pad(bias.astype("<i4"), (0, 3))
    write_words(work / "canvas_bias.hex", bias_words.astype("<i4").view(np.uint8))
    coeff = np.zeros((24, 2), dtype=np.uint8)
    coeff[:channels] = np.column_stack((gamma, beta))
    write_words(work / "canvas_coeff.hex", coeff)

    raw = np.clip((weights @ activations + bias[:, None]) >> 2, -128, 127)
    gelu = lut[raw & 255].view(np.int8)
    scaled = np.rint(gelu * ratio).astype(np.int8)
    for name, before, norm_input in (("raw", raw, raw), ("gelu", gelu, scaled)):
        norm, _ = layernorm_codes(norm_input.T, gamma, beta, cfg)
        write_words(work / f"canvas_{name}.hex", pad(before))
        write_words(work / f"canvas_{name}_norm.hex", pad(norm.T))


if __name__ == "__main__":
    main()
