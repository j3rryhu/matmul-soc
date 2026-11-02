"""Export selected softmax contract plus deterministic synthetic and real row vectors."""
import json
from pathlib import Path
import numpy as np
from softmax_fixed import Config, make_lut, softmax_codes


def main():
    root = Path(__file__).resolve().parent
    metadata = json.loads((root / "runs/fixed_21m/config.json").read_text())
    config = Config(**{k:metadata[k] for k in Config.__dataclass_fields__})
    if config != Config(fractional_bits=8, cutoff=12):
        raise ValueError("RTL implements only the selected 16-bit, /256, cutoff-12 contract")
    out = root / "example"
    out.mkdir(exist_ok=True)
    rows = []
    for n in (1,2,49,196):
        rows.extend([np.zeros(n,dtype=np.int64), np.full(n,-32768,dtype=np.int64),
                     np.full(n,32767,dtype=np.int64),
                     np.array([32767]+[-32768]*(n-1),dtype=np.int64)])
    rows.append(np.array([0,-1,-2,-3071,-3072,-3073,-6553,-32768],dtype=np.int64))
    rng = np.random.default_rng(20261102)
    for _ in range(30):
        n=int(rng.integers(1,197))
        rows.append(rng.integers(-32768,32768,size=n,dtype=np.int64))
    capture=json.loads((root / "runs/reference_21m/report.json").read_text())
    for record in capture["images"]:
        with np.load(root / "runs/fixed_21m" / record["artifact"]) as data:
            real=data["score_codes"].reshape(-1,49)
            rows.extend(real[rng.choice(len(real),10,replace=False)].astype(np.int64))
    physical_rom = np.pad(make_lut(config), (0, 4096-len(make_lut(config))))
    (out / "exp_lut.hex").write_text("".join(f"{int(x):04x}\n" for x in physical_rom))
    with (out / "rows.txt").open("w") as f:
        for row in rows:
            expected,_=softmax_codes(row[None,:],config)
            f.write(str(len(row))+"\n")
            for score,probability in zip(row,expected[0]):
                f.write(f"{int(score)} {int(probability)}\n")
    manifest={"config":config.metadata(),"rows":len(rows),"elements":sum(map(len,rows)),
              "synthetic_rows":47,"real_rows":100,"seed":20261102,
              "physical_rom_entries":4096,"useful_rom_entries":3073,
              "source":"10 rows sampled per captured image, including calibration and validation; no tuning"}
    (out / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
