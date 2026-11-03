"""Exercise distributed accumulation, combined progress, and checkpoint resume."""

import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


class _Progress:
    def __init__(self, *, total, disable=False, **kwargs):
        self.total = total
        self.disable = disable
        self.n = 0
        self.updates = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, amount):
        self.n += amount
        self.updates.append(self.n)

    def set_postfix(self, **kwargs):
        pass


def _distributed_worker(rank, directory):
    from src.training import epoch, segmentation
    from src.utils.seed import make_generator, seed_everything

    directory = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(directory / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    progress = []
    validation_samples = []

    class ValidationDataset(TensorDataset):
        samples = 0

        def __getitem__(self, index):
            self.samples += 1
            return super().__getitem__(index)

    def progress_factory(**kwargs):
        bar = _Progress(**kwargs)
        progress.append(bar)
        return bar

    patch = pytest.MonkeyPatch()
    patch.setattr(epoch, "tqdm", progress_factory)

    def run(name, epochs, resume=False):
        seed_everything(7)
        images = torch.arange(12 * 3 * 4 * 4, dtype=torch.float32).reshape(12, 3, 4, 4) / 100
        targets = torch.arange(12 * 4 * 4).reshape(12, 4, 4) % 3
        dataset = TensorDataset(images, targets)
        sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, seed=7)
        train = DataLoader(dataset, batch_size=2, sampler=sampler, generator=make_generator(7))
        validation_dataset = ValidationDataset(images[:5], targets[:5])
        validation = DataLoader(validation_dataset, batch_size=2)
        model = DistributedDataParallel(nn.Sequential(nn.Conv2d(3, 3, 1), nn.Dropout(0.25)))
        seed_everything(20 + rank)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = segmentation.warmup_cosine(optimizer, 1, 4)
        result = segmentation.fit_segmentation(
            model,
            train,
            validation,
            optimizer,
            scheduler,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            epochs=epochs,
            accumulation_steps=2,
            seed=7,
            checkpoint_dir=directory / name / "checkpoints",
            run_dir=directory / name / "run",
            checkpoint_config={"name": name},
            num_classes=3,
            resume=resume,
        )
        validation_samples.append(validation_dataset.samples)
        return model.module.state_dict(), [record.train_loss for record in result.history]

    try:
        straight, straight_loss = run("straight", 2)
        run("resumed", 1)
        resumed, resumed_loss = run("resumed", 2, resume=True)
        torch.save(
            {
                "validation_samples": validation_samples,
                "straight": straight,
                "resumed": resumed,
                "straight_loss": straight_loss,
                "resumed_loss": resumed_loss,
                "progress": [
                    {
                        "total": bar.total,
                        "n": bar.n,
                        "updates": bar.updates,
                        "disabled": bar.disable,
                    }
                    for bar in progress
                ],
            },
            directory / f"rank{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires the Gloo backend")
def test_two_process_training_combines_progress_and_resumes(tmp_path):
    context = mp.spawn(_distributed_worker, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic() + 75
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise AssertionError("distributed training did not finish within 75 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)

    ranks = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
    for name, expected in ranks[0]["straight"].items():
        torch.testing.assert_close(expected, ranks[1]["straight"][name], rtol=0, atol=0)
        for result in ranks:
            torch.testing.assert_close(expected, result["resumed"][name], rtol=0, atol=0)
    assert ranks[0]["straight_loss"] == ranks[1]["straight_loss"]
    assert ranks[0]["straight_loss"] == pytest.approx(ranks[0]["resumed_loss"])
    assert len(ranks[0]["progress"]) == 4
    for bar in ranks[0]["progress"]:
        assert not bar["disabled"]
        assert bar["total"] == bar["n"] == 12
        assert bar["updates"] == [8, 12]
    assert all(bar["disabled"] and not bar["updates"] for bar in ranks[1]["progress"])

    assert ranks[0]["validation_samples"] == [10, 5, 5]
    assert ranks[1]["validation_samples"] == [0, 0, 0]

    payload = torch.load(tmp_path / "straight/checkpoints/last.pt", weights_only=True)
    assert payload["world_size"] == len(payload["rng_states"]) == 2
    assert not torch.equal(payload["rng_states"][0]["torch"], payload["rng_states"][1]["torch"])
    restored = nn.Sequential(nn.Conv2d(3, 3, 1), nn.Dropout(0.25))
    from src.training.segmentation import load_training_state

    load_training_state(tmp_path / "straight/checkpoints/last.pt", restored, restore_rng=False)
    with pytest.raises(ValueError, match="same world size"):
        load_training_state(tmp_path / "straight/checkpoints/last.pt", restored)
    assert payload["metadata"]["global_step"] == 4
    assert sorted(path.name for path in (tmp_path / "straight/checkpoints").iterdir()) == [
        "best.pt",
        "last.pt",
    ]
    history = json.loads((tmp_path / "straight/run/history.json").read_text())
    assert [record["epoch"] for record in history] == [1, 2]
