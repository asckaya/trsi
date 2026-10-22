from enum import StrEnum

TORCH_ALIASES: dict[str, str] = {
    "float64": "f64",
    "double": "f64",
    "float32": "f32",
    "float": "f32",
    "float16": "f16",
    "half": "f16",
    "bfloat16": "bf16",
    "float8_e4m3fn": "fp8e4m3",
    "float8_e4m3fnuz": "fp8e4m3",
    "float8_e5m2": "fp8e5m2",
    "float4_e2m1fn_x2": "f4e2m1",
    "int64": "i64",
    "long": "i64",
    "int32": "i32",
    "int": "i32",
    "int16": "i16",
    "short": "i16",
    "int8": "i8",
    "uint8": "u8",
    "bool": "bool",
}

DTYPE_BITS: dict[str, int] = {
    "f64": 64,
    "f32": 32,
    "tf32": 32,
    "f16": 16,
    "bf16": 16,
    "fp8e4m3": 8,
    "fp8e5m2": 8,
    "f4e2m1": 4,
    "i64": 64,
    "i32": 32,
    "i16": 16,
    "i8": 8,
    "u8": 8,
    "bool": 8,
}


class DType(StrEnum):
    """A storage or arithmetic type this project prices.

    A dtype not named here is refused rather than priced at a guessed width:
    every figure downstream multiplies by that width, so a wrong width is a
    report that reads as a right one. Sub-byte widths are packed.
    """

    F64 = "f64"
    F32 = "f32"
    TF32 = "tf32"
    F16 = "f16"
    BF16 = "bf16"
    FP8E4M3 = "fp8e4m3"
    FP8E5M2 = "fp8e5m2"
    F4E2M1 = "f4e2m1"
    I64 = "i64"
    I32 = "i32"
    I16 = "i16"
    I8 = "i8"
    U8 = "u8"
    BOOL = "bool"

    @property
    def bits(self) -> int:
        """Storage width in bits, packed sub-byte widths included."""
        return DTYPE_BITS[self.value]

    def nbytes(self, numel: int) -> int:
        """Bytes holding *numel* elements, rounded up to whole bytes."""
        return (numel * self.bits + 7) // 8

    @classmethod
    def of_torch(cls, name: str) -> DType:
        """Resolve a torch dtype spelling to its canonical form.

        Tensor core types below 16 bits are named by exponent and mantissa
        widths rather than by size, which is why this is a table and not a parse.

        Args:
            name: A `torch.float32` name, a `c10::BFloat16` name, or the bare
                suffix, in either spelling torch's profiler writes.

        Returns:
            The canonical dtype.

        Raises:
            KeyError: The name is not a dtype this project prices.
        """
        suffix = name.rsplit(".", maxsplit=1)[-1].rsplit("::", maxsplit=1)[-1].lower()
        canonical = TORCH_ALIASES.get(suffix)
        if canonical is None:
            raise KeyError(f"unknown dtype {name!r}; known suffixes: {sorted(TORCH_ALIASES)}")
        return cls(canonical)
