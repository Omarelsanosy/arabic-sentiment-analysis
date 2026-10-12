
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[0]
MODEL_PATH = PROJECT_ROOT / "models" / "arabert_sentiment"
ONNX_PATH = MODEL_PATH / "model.onnx"

TEXTS = [
    "المنتج رائع جدا",
    "الخدمة سيئة للغاية",
    "التجربة عادية",
]

MAX_LENGTH = 128
WARMUP_RUNS = 5
BENCHMARK_RUNS = 30


def prepare_inputs(tokenizer, texts):
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    return {
        name: tensor
        for name, tensor in tokens.items()
    }


def benchmark_pytorch(model, inputs):
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(**inputs)

        timings = []

        for _ in range(BENCHMARK_RUNS):
            start = time.perf_counter()
            outputs = model(**inputs)
            timings.append((time.perf_counter() - start) * 1000)

    return outputs.logits.cpu().numpy(), timings


def benchmark_onnx(session, inputs):
    onnx_inputs = {
        name: tensor.cpu().numpy()
        for name, tensor in inputs.items()
    }

    for _ in range(WARMUP_RUNS):
        session.run(["logits"], onnx_inputs)

    timings = []
    outputs = None

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()
        outputs = session.run(["logits"], onnx_inputs)[0]
        timings.append((time.perf_counter() - start) * 1000)

    return outputs, timings


def print_metrics(name, timings):
    print(f"\n{name}")
    print(f"Mean latency: {np.mean(timings):.3f} ms")
    print(f"P50 latency:  {np.percentile(timings, 50):.3f} ms")
    print(f"P95 latency:  {np.percentile(timings, 95):.3f} ms")


def main():
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {ONNX_PATH}\n"
            "Run the ONNX export script first."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    pytorch_model = (
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            attn_implementation="eager",
        )
    )

    pytorch_model.cpu().eval()

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    inputs = prepare_inputs(tokenizer, TEXTS)

    pytorch_logits, pytorch_timings = benchmark_pytorch(
        pytorch_model, inputs
    )

    onnx_logits, onnx_timings = benchmark_onnx(
        session, inputs
    )

    max_diff = np.max(
        np.abs(pytorch_logits - onnx_logits)
    )

    print("\n=== Correctness Check ===")
    print(f"PyTorch logits shape: {pytorch_logits.shape}")
    print(f"ONNX logits shape:    {onnx_logits.shape}")
    print(f"Maximum absolute difference: {max_diff:.8f}")

    np.testing.assert_allclose(
        pytorch_logits,
        onnx_logits,
        rtol=1e-3,
        atol=1e-4,
    )

    print("PASS: PyTorch and ONNX logits are close.")

    print_metrics("PyTorch CPU", pytorch_timings)
    print_metrics("ONNX Runtime CPU", onnx_timings)

    speedup = np.mean(pytorch_timings) / np.mean(onnx_timings)

    print(f"\nONNX/PyTorch speed ratio: {speedup:.2f}x")


if __name__ == "__main__":
    main()

