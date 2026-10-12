import argparse
import os
import tempfile
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "arabert_sentiment"
DEFAULT_OUTPUT_PATH = MODEL_PATH / "model.onnx"
INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids")


class _LogitsOnlyModel(nn.Module):
    def __init__(self, model: nn.Module, input_names: list[str]) -> None:
        super().__init__()
        self.model = model
        self.input_names = input_names

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        model_inputs = dict(zip(self.input_names, inputs, strict=True))
        return self.model(**model_inputs).logits


def export_onnx(output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    try:
        import onnx
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ONNX export requires the optional dependency. "
            "Run `uv sync --extra onnx` first."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.cpu().eval()

    tokens = tokenizer(
        "رائع المنتج",
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    input_names = [name for name in INPUT_NAMES if name in tokens]
    model_inputs = tuple(tokens[name] for name in input_names)
    dynamic_axes = {
        name: {0: "batch_size", 1: "sequence_length"} for name in input_names
    }
    dynamic_axes["logits"] = {0: "batch_size"}
    export_model = _LogitsOnlyModel(model, input_names)

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        suffix=".onnx",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        torch.onnx.export(
            export_model,
            model_inputs,
            str(temporary_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
        )
        onnx.checker.check_model(str(temporary_path))
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the sentiment model to ONNX.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output ONNX file (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()
    output_path = export_onnx(args.output)
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
