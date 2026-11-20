"""Estimate the A3 image-text retrieval compute budget without downloading weights.

Every model is built with random weights, so the script needs no downloads and no
dataset. It reports parameters, forward FLOPs, checkpoint sizes and, on a CUDA
device, the peak memory and pairs per second of one full training step with a
symmetric InfoNCE loss. The JSON it writes backs the compute section of the A3
proposal.
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torchvision
from loguru import logger
from torch import nn
from torch.nn import functional
from torch.utils.flop_counter import FlopCounterMode

DEFAULT_OUTPUT = Path("results/a3/compute/estimate.json")

IMAGE_SIDE = 224
IMAGE_DIM = 2048
TEXT_DIM = 1024
PROJECTION_DIM = 1024
EMBED_DIM = 300
VOCAB_SIZE = 7692 + 4  # word vocabulary plus the four special tokens
CAPTION_LENGTH = 36
TEMPERATURE = 0.07

# Shape of OpenAI CLIP ViT-B/32, built from scratch with random weights.
CLIP_IMAGE_DIM = 768
CLIP_EMBED_DIM = 512
CLIP_VOCAB_SIZE = 49408  # CLIP's byte-pair vocabulary
CLIP_TEXT_LENGTH = 77
CLIP_DEPTH = 12
CLIP_HEADS = 8
CLIP_FEEDFORWARD_DIM = 2048

CACHE_IMAGES = 8000
BYTES_PER_PARAMETER = 4  # float32 checkpoint
DEFAULT_BATCH_SIZE = 128
BENCHMARK_STEPS = 10
WARMUP_STEPS = 3


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="device to benchmark on (default: cuda when available)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="pairs per training step for the GPU benchmark",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated model names to measure (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="where to write the estimate JSON",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    """Turn --device into a concrete device, refusing CUDA when it is unavailable."""
    if name == "cuda" and not torch.cuda.is_available():
        logger.error("--device cuda requested but torch.cuda.is_available() is False")
        raise SystemExit(1)
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def build_image_encoder() -> nn.Module:
    """ResNet-50 without its classifier: 3x224x224 images to 2048 features."""
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Identity()
    return model


class TextEncoderGRU(nn.Module):
    """One-layer GRU over token ids; the last hidden state is the text embedding."""

    vocab_size = VOCAB_SIZE
    caption_length = CAPTION_LENGTH

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.gru = nn.GRU(EMBED_DIM, TEXT_DIM, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the final hidden state for a batch of token id sequences."""
        _, hidden = self.gru(self.embedding(tokens))
        return hidden[-1]


class ClipTextEncoder(nn.Module):
    """CLIP text tower: token and positional embeddings, pre-LN transformer, EOT head."""

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(CLIP_VOCAB_SIZE, CLIP_EMBED_DIM)
        self.position_embedding = nn.Parameter(torch.empty(CLIP_TEXT_LENGTH, CLIP_EMBED_DIM))
        nn.init.normal_(self.position_embedding, std=0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=CLIP_EMBED_DIM,
            nhead=CLIP_HEADS,
            dim_feedforward=CLIP_FEEDFORWARD_DIM,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=CLIP_DEPTH, enable_nested_tensor=False
        )
        self.final_layer_norm = nn.LayerNorm(CLIP_EMBED_DIM)
        self.projection = nn.Linear(CLIP_EMBED_DIM, CLIP_EMBED_DIM, bias=False)
        # Boolean mask with True above the diagonal, so each token sees only the past.
        causal = torch.triu(
            torch.ones(CLIP_TEXT_LENGTH, CLIP_TEXT_LENGTH, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed a padded caption and return the projected feature at the EOT token."""
        hidden = self.token_embedding(tokens) + self.position_embedding
        hidden = self.transformer(hidden, mask=self.causal_mask)
        return self.projection(self.final_layer_norm(hidden)[:, -1])


class ClipVitB32Proxy(nn.Module):
    """Trainable CLIP ViT-B/32 stand-in: torchvision ViT image tower plus a CLIP text tower."""

    vocab_size = CLIP_VOCAB_SIZE
    caption_length = CLIP_TEXT_LENGTH

    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = torchvision.models.vit_b_32(weights=None)
        self.image_encoder.heads = nn.Identity()
        self.image_projection = nn.Linear(CLIP_IMAGE_DIM, CLIP_EMBED_DIM, bias=False)
        self.text_encoder = ClipTextEncoder()

    def forward(
        self, images: torch.Tensor, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an image batch and a token batch into the shared embedding space."""
        image_features = self.image_encoder(images)
        image_embeddings = functional.normalize(self.image_projection(image_features), dim=-1)
        text_embeddings = functional.normalize(self.text_encoder(tokens), dim=-1)
        return image_embeddings, text_embeddings


class VisualSemanticEmbedding(nn.Module):
    """Dual encoder that returns L2-normalised image and text embeddings."""

    vocab_size = VOCAB_SIZE
    caption_length = CAPTION_LENGTH

    def __init__(self, finetune_layer4: bool) -> None:
        super().__init__()
        self.finetune_layer4 = finetune_layer4
        self.image_encoder = build_image_encoder()
        self.image_projection = nn.Linear(IMAGE_DIM, PROJECTION_DIM)
        self.text_encoder = TextEncoderGRU()
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False
        if finetune_layer4:
            for parameter in self.image_encoder.layer4.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True) -> "VisualSemanticEmbedding":
        """Keep a frozen image encoder in eval mode, as the frozen baseline requires."""
        super().train(mode)
        if not self.finetune_layer4:
            self.image_encoder.eval()
        return self

    def forward(
        self, images: torch.Tensor, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an image batch and a token batch into the shared embedding space."""
        image_features = self.image_encoder(images)
        image_embeddings = functional.normalize(self.image_projection(image_features), dim=-1)
        text_embeddings = functional.normalize(self.text_encoder(tokens), dim=-1)
        return image_embeddings, text_embeddings


def example_inputs(kind: str, model: nn.Module) -> tuple[torch.Tensor, ...]:
    """Build the forward-FLOPs inputs matching a model's declared input kind."""
    image = torch.randn(1, 3, IMAGE_SIDE, IMAGE_SIDE)
    if kind == "image":
        return (image,)
    tokens = torch.randint(0, model.vocab_size, (1, model.caption_length))
    if kind == "tokens":
        return (tokens,)
    return image, tokens


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    """Return the total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def forward_flops(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> int:
    """Count the forward FLOPs of one sample pair with torch's FLOP counter."""
    model.eval()
    # Fused multi-head attention kernels hide their matmuls from the counter, so count
    # through the unfused path. This changes how attention runs, not its arithmetic.
    fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(*inputs)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
    return counter.get_total_flops()


def training_step(
    model: nn.Module,
    images: torch.Tensor,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Run one forward, symmetric InfoNCE loss, backward pass and optimiser step."""
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        image_embeddings, text_embeddings = model(images, tokens)
        logits = (image_embeddings @ text_embeddings.T).float() / TEMPERATURE
    loss = 0.5 * (
        functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.T, labels)
    )
    loss.backward()
    optimizer.step()


def benchmark(model: nn.Module, batch_size: int) -> dict[str, float]:
    """Measure peak GPU memory and pairs per second over the timed training steps."""
    device = torch.device("cuda")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4
    )
    images = torch.randn(batch_size, 3, IMAGE_SIDE, IMAGE_SIDE, device=device)
    tokens = torch.randint(0, model.vocab_size, (batch_size, model.caption_length), device=device)
    labels = torch.arange(batch_size, device=device)

    for _ in range(WARMUP_STEPS):
        training_step(model, images, tokens, labels, optimizer)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(BENCHMARK_STEPS):
        training_step(model, images, tokens, labels, optimizer)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return {
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1e6,
        "pairs_per_second": batch_size * BENCHMARK_STEPS / elapsed,
    }


def checkpoint_sizes(total: int, trainable: int) -> dict[str, float]:
    """Estimate float32 checkpoint size in MB for the trained and the full model."""
    return {
        "checkpoint_trainable_mb": trainable * BYTES_PER_PARAMETER / 1e6,
        "checkpoint_full_mb": total * BYTES_PER_PARAMETER / 1e6,
    }


def feature_cache_sizes() -> dict[str, Any]:
    """Size of the cached ResNet-50 features for the whole image set."""
    features = CACHE_IMAGES * IMAGE_DIM
    return {
        "images": CACHE_IMAGES,
        "dim": IMAGE_DIM,
        "fp32_mb": features * 4 / 1e6,
        "fp16_mb": features * 2 / 1e6,
    }


def format_table(results: dict[str, dict[str, Any]]) -> str:
    """Render one fixed-width row per model for stdout."""
    header = (
        f"{'model':<26} {'total params':>14} {'trainable':>14} "
        f"{'GFLOPs':>8} {'peak GPU MB':>12} {'pairs/s':>10}"
    )
    lines = [header, "-" * len(header)]
    for name, record in results.items():
        peak = record["peak_gpu_memory_mb"]
        throughput = record["pairs_per_second"]
        peak_text = "n/a" if peak is None else f"{peak:.1f}"
        throughput_text = "n/a" if throughput is None else f"{throughput:.1f}"
        lines.append(
            f"{name:<26} {record['total_params']:>14,} {record['trainable_params']:>14,} "
            f"{record['forward_gflops']:>8.3f} {peak_text:>12} {throughput_text:>10}"
        )
    return "\n".join(lines)


def main() -> None:
    """Build every model with random weights, measure it, and write the estimate."""
    args = parse_args()
    device = resolve_device(args.device)

    # Factories build each model only when it is measured, so earlier models do not
    # occupy GPU memory while a later one is benchmarked. The input kind names what
    # forward_flops feeds the model, so token length and vocabulary follow the model.
    catalogue: tuple[tuple[str, Callable[[], nn.Module], str, bool], ...] = (
        ("image_encoder_resnet50", build_image_encoder, "image", False),
        ("text_encoder_gru", TextEncoderGRU, "tokens", False),
        ("vse_frozen", lambda: VisualSemanticEmbedding(finetune_layer4=False), "pair", True),
        ("vse_finetune", lambda: VisualSemanticEmbedding(finetune_layer4=True), "pair", True),
        ("clip_vitb32_proxy", ClipVitB32Proxy, "pair", True),
    )
    if args.models:
        selected = [name.strip() for name in args.models.split(",") if name.strip()]
        known = {name for name, _, _, _ in catalogue}
        unknown = [name for name in selected if name not in known]
        if unknown:
            logger.error("Unknown --models entry: {}", ", ".join(unknown))
            logger.error("Available models: {}", ", ".join(sorted(known)))
            raise SystemExit(1)
        catalogue = tuple(entry for entry in catalogue if entry[0] in selected)

    results: dict[str, dict[str, Any]] = {}
    for name, factory, input_kind, retrieval_model in catalogue:
        logger.info("Measuring {}", name)
        model = factory()
        total, trainable = parameter_counts(model)
        flops = forward_flops(model, example_inputs(input_kind, model))
        record: dict[str, Any] = {
            "total_params": total,
            "trainable_params": trainable,
            "forward_flops": flops,
            "forward_gflops": flops / 1e9,
            **checkpoint_sizes(total, trainable),
            "peak_gpu_memory_mb": None,
            "pairs_per_second": None,
        }
        if not retrieval_model:
            record["benchmark_note"] = "component encoder: has no retrieval training step"
        elif device.type == "cuda":
            measured = benchmark(model, args.batch_size)
            record.update(measured)
            logger.info(
                "  {:.1f} GB peak, {:.1f} pairs/s",
                measured["peak_gpu_memory_mb"] / 1e3,
                measured["pairs_per_second"],
            )
        results[name] = record
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    estimate = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "batch_size": args.batch_size,
        "benchmark_steps": BENCHMARK_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "models": results,
        "resnet50_feature_cache": feature_cache_sizes(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(estimate, indent=2), encoding="utf-8")

    print(format_table(results))
    print(
        f"\nResNet-50 cache for {CACHE_IMAGES:,} images: "
        f"{estimate['resnet50_feature_cache']['fp32_mb']:.1f} MB fp32, "
        f"{estimate['resnet50_feature_cache']['fp16_mb']:.1f} MB fp16"
    )
    if device.type != "cuda":
        print("GPU memory and throughput are null on CPU.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
