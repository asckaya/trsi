# facts

> Owns `trainer_rsi.facts`, the installed documents under
> `src/trainer_rsi/facts/hardware/`, and the rates a bound divides by.
> Read by `analyze/`, `plan/`.

## 1. Why documents

A roofline is a claim about a machine. A claim whose provenance is a constant in
a source file cannot be checked, corrected, or held to its conditions, and the
reader has no way to tell a vendor figure from a guess.

Every figure is therefore installed as data, and every figure names where it came
from.

## 2. Document schema

One `.toml` per device, schema `trainer_rsi.hardware/v1`, kind `device`.

```toml
[spec]
schema = "trainer_rsi.hardware/v1"
kind = "device"
id = "nvidia.h800_sxm"
label = "NVIDIA H800 SXM, 80 GB HBM3"

[compatibility]
architecture = "nvidia.sm90"

[facts.throughput.bf16]
value = 989500000000000
unit = "flop/s"
origin = "derived"
source = "https://…/NVIDIA H800.pdf"
conditions = "dense; the datasheet states 1979 TFLOPS with the footnote that dense figures are half."
```

- `unit` MUST be one of `byte`, `byte/s`, `flop/s`, `op/s`, `count`, `s`, `Hz`.
- `origin` MUST be one of `vendor`, `derived`, `reference`, `measured`.
- `source` MUST name where the figure came from. For `derived`, the arithmetic
  belongs in `conditions`; for `reference`, so does who measured it.
- `conditions` MUST state what makes the figure true — form factor, precision,
  sparsity, bus width, or the caveat that a number is inferred rather than
  published.
- A non-fact leaf under `facts` MUST be refused: a bare number has no provenance,
  and accepting one would let an unsourced figure into a report.

## 3. Absent is not unavailable

- A dtype the product has no mode for MUST be recorded as
  `status = "unavailable"` with a `conditions` line saying so.
- A figure nobody publishes MAY be omitted entirely.
- `HardwareDoc.amount` MUST raise on `unavailable`; `HardwareDoc.maybe` MUST
  return `None` only for `unavailable`, and MUST raise for absent.

The distinction reaches the report: an unavailable rate makes an op `unrated`,
which is a floor with no promise behind it, while an absent one lowers coverage.

## 4. Registry and digest

- The registry MUST read `Path(__file__).parent / "hardware"`, so an editable
  checkout and a wheel see the same files.
- `HardwareDoc.digest` MUST be SHA-256 of the file as installed, and every report
  MUST carry the digest of the document it used.
- An unknown `device_id` MUST be refused, listing what is installed.

## 5. Projection

`facts.envelope.limits` projects a document into the rates the analysis divides
by: peak per dtype, HBM bandwidth and capacity, and the interconnect.

- A dtype with no stated rate MUST be left out of the projection, not filled in.
- A link rate is stored bidirectional and MUST be halved for the per-direction
  rate a ring collective is limited by.
- A group wider than `PCIE_DOMAIN` has no installed link, and `Limits.link` MUST
  return `None` rather than price it at a rate it does not use.
