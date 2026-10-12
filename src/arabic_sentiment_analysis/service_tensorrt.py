import threading
from pathlib import Path
from typing import Any

import bentoml
import numpy as np
from transformers import AutoConfig, AutoTokenizer

from arabic_sentiment_analysis.preprocessing import clean_arabic_text


class TensorRTEngine:
    def __init__(self, engine_path: Path) -> None:
        try:
            import tensorrt as trt
            from cuda.bindings import driver as cuda
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "The TensorRT Bento requires the `tensorrt-cu12` and "
                "`cuda-python` packages."
            ) from error

        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self._cuda = cuda
        self._lock = threading.Lock()
        self._context = None
        self._stream = None
        self._runtime = None
        self._engine = None
        self._execution_context = None

        self._check_cuda(cuda.cuInit(0), "initialize CUDA")
        device, = self._check_cuda(cuda.cuDeviceGet(0), "select CUDA device")
        self._context, = self._check_cuda(
            cuda.cuCtxCreate(None, 0, device),
            "create CUDA context",
        )
        try:
            self._stream, = self._check_cuda(
                cuda.cuStreamCreate(0),
                "create CUDA stream",
            )
            self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self._engine = self._runtime.deserialize_cuda_engine(
                engine_path.read_bytes()
            )
            if self._engine is None:
                raise RuntimeError(
                    "TensorRT could not deserialize the engine. Build the engine "
                    "with the same TensorRT version and operating system used "
                    "to run this Bento."
                )
            self._execution_context = self._engine.create_execution_context()
            if self._execution_context is None:
                raise RuntimeError("TensorRT could not create an execution context")
        except Exception:
            cuda.cuCtxDestroy(self._context)
            self._context = None
            raise

    def _check_cuda(self, result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
        status, *values = result
        if status != self._cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA failed to {operation}: {status}")
        return tuple(values)

    def infer(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        if self._execution_context is None or self._stream is None:
            raise RuntimeError("TensorRT engine is not initialized")

        with self._lock:
            self._check_cuda(
                self._cuda.cuCtxSetCurrent(self._context),
                "activate CUDA context",
            )
            allocations = []
            try:
                for name, value in inputs.items():
                    pointer, = self._check_cuda(
                        self._cuda.cuMemAlloc(value.nbytes),
                        f"allocate input {name}",
                    )
                    allocations.append(pointer)
                    self._check_cuda(
                        self._cuda.cuMemcpyHtoD(pointer, value, value.nbytes),
                        f"copy input {name}",
                    )
                    if not self._execution_context.set_input_shape(
                        name, value.shape
                    ):
                        raise RuntimeError(
                            f"TensorRT rejected input shape for {name}: {value.shape}"
                        )
                    if not self._execution_context.set_tensor_address(
                        name, int(pointer)
                    ):
                        raise RuntimeError(f"TensorRT rejected input buffer {name}")

                output_shape = tuple(
                    self._execution_context.get_tensor_shape("logits")
                )
                if any(dimension < 1 for dimension in output_shape):
                    raise RuntimeError(
                        f"TensorRT produced an invalid output shape: {output_shape}"
                    )
                logits = np.empty(output_shape, dtype=np.float32)
                output_pointer, = self._check_cuda(
                    self._cuda.cuMemAlloc(logits.nbytes),
                    "allocate logits",
                )
                allocations.append(output_pointer)
                if not self._execution_context.set_tensor_address(
                    "logits", int(output_pointer)
                ):
                    raise RuntimeError("TensorRT rejected the logits output buffer")
                if not self._execution_context.execute_async_v3(
                    int(self._stream)
                ):
                    raise RuntimeError("TensorRT inference execution failed")
                self._check_cuda(
                    self._cuda.cuStreamSynchronize(self._stream),
                    "wait for inference",
                )
                self._check_cuda(
                    self._cuda.cuMemcpyDtoH(
                        logits,
                        output_pointer,
                        logits.nbytes,
                    ),
                    "copy logits to host",
                )
                return logits
            finally:
                for pointer in allocations:
                    self._check_cuda(
                        self._cuda.cuMemFree(pointer),
                        "free inference buffer",
                    )


@bentoml.service
class TensorRTSentimentService:
    max_batch_size = 16

    def __init__(self) -> None:
        model_path = (
            Path(__file__).resolve().parents[2] / "models" / "arabert_sentiment"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.engine = TensorRTEngine(model_path / "model.fp16.engine")

    @bentoml.api
    def predict(self, text: list[str]) -> list[dict[str, str | float]]:
        cleaned_text = [clean_arabic_text(item) for item in text]
        if not cleaned_text:
            return []

        logits_chunks = []
        for start in range(0, len(cleaned_text), self.max_batch_size):
            tokens = self.tokenizer(
                cleaned_text[start : start + self.max_batch_size],
                return_tensors="np",
                truncation=True,
                padding=True,
                max_length=128,
            )
            inputs = {
                name: np.ascontiguousarray(value, dtype=np.int64)
                for name, value in tokens.items()
            }
            logits_chunks.append(self.engine.infer(inputs))
        logits = np.concatenate(logits_chunks, axis=0)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        predicted_ids = probabilities.argmax(axis=1)
        confidences = probabilities.max(axis=1)
        return [
            {
                "sentiment": self.config.id2label[int(label_id)],
                "confidence": float(confidence),
            }
            for label_id, confidence in zip(predicted_ids, confidences, strict=True)
        ]
