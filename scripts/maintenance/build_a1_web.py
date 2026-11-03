# Long inline HTML templates retain their readable paragraph structure.
# ruff: noqa: E501
"""Build the A1 website from 18 explicitly selected saved runs.
Run: uv run python -m scripts.maintenance.build_a1_web
"""

import csv
import hashlib
import html
import io
import json
import re
import shutil
from pathlib import Path

from scripts.evaluation.compare import aggregate, to_csv, to_full_csv
from src.plots.classification import plot_comparative_learning_curves

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
FIG = WEB / "images/comparison/a1"
OUT = WEB / "data/a1"
REPO = "https://github.com/vietanh1802/CO3133-Deep-Learning-261-Placeholder"
SOURCE = REPO + "/tree/a2/m1"
MODELS = {
    "linear": "Linear",
    "mlp": "MLP",
    "cnn": "CNN",
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Transformer",
}
DATASETS = {"fashion_mnist": "Fashion-MNIST", "mnist": "MNIST", "cifar10": "CIFAR-10"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(headings, rows, caption=""):
    cap = f"<caption>{caption}</caption>" if caption else ""
    head = "".join(f'<th scope="col">{v}</th>' for v in headings)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
    return f'<div class="table-scroll"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(path, alt, caption):
    return f'<figure><a href="{path}.png"><img src="{path}.png" alt="{alt}" loading="lazy"></a><figcaption>{caption} <a href="{path}.pdf">PDF</a></figcaption></figure>'


def section(key, title, content):
    return f'<section id="{key}"><h2>{title}</h2>\n{content}\n</section>'


def code(text):
    return f"<pre><code>{html.escape(text)}</code></pre>"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, bundles = [], {}
    for dataset in DATASETS:
        bundles[dataset] = {}
        for model in MODELS:
            name = f"{dataset}_{'cnn_spatial' if model == 'cnn' else model}_seed42"
            directory = ROOT / "results/a1" / name
            result = read(directory / "test_result.json")
            history = read(directory / "training_summary.json")
            errors = read(directory / "test_error_analysis.json")
            assert result["name"] == name and result["split"] == "test"
            assert result["seed"] == 42 and result["config"]["epochs"] == 50
            assert result["sample_count"] == 10000
            assert len(history["train_loss"]) == len(history["validation_loss"]) == 50
            assert abs(history["best_validation_loss"] - min(history["validation_loss"])) < 1e-8
            bundles[dataset][model] = {"result": result, "history": history, "errors": errors}
            records.append(result)
            for source, target in [
                ("test_confusion_matrix", "confusion"),
                ("test_confident_errors", "errors"),
                ("test_confident_correct", "correct"),
            ]:
                if target == "correct" and model != "cnn":
                    continue
                for ext in ["png", "pdf"]:
                    shutil.copyfile(
                        directory / f"{source}.{ext}", FIG / f"{dataset}_{model}_{target}.{ext}"
                    )
        histories = {MODELS[m]: b["history"] for m, b in bundles[dataset].items()}
        maximum = max(
            v for h in histories.values() for k in ["train_loss", "validation_loss"] for v in h[k]
        )
        plot_comparative_learning_curves(
            histories, FIG / f"{dataset}_learning.png", loss_max=maximum
        )
    runs = [{"dataset": r["dataset"], "model": r["model"], "name": r["name"]} for r in records]
    (FIG / "runs.json").write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    (OUT / "experiments.json").write_text(json.dumps(bundles, indent=2) + "\n", encoding="utf-8")
    (OUT / "AI_USAGE.md").write_bytes((ROOT / "AI_USAGE.md").read_bytes())
    (WEB / "data/AI_USAGE.md").write_bytes((ROOT / "AI_USAGE.md").read_bytes())
    eda = read(ROOT / "results/a1/eda/fashion_mnist/summary.json")
    (OUT / "eda_summary.json").write_text(json.dumps(eda, indent=2) + "\n", encoding="utf-8")
    fields = [
        "dataset",
        "model",
        "run",
        "seed",
        "epochs",
        "accuracy",
        "macro_f1",
        "parameter_count",
        "best_epoch",
        "best_validation_loss",
        "training_seconds",
        "inference_seconds",
        "evaluation_device",
        "python",
        "torch",
        "git_commit",
        "checkpoint",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for r in records:
        writer.writerow(
            {
                "dataset": r["dataset"],
                "model": r["model"],
                "run": r["name"],
                "seed": r["seed"],
                "epochs": 50,
                **r["metrics"],
                "parameter_count": r["parameter_count"],
                "best_epoch": r["checkpoint"]["epoch"],
                "best_validation_loss": r["checkpoint"]["best_validation_loss"],
                "training_seconds": r["training_seconds"],
                "inference_seconds": r["inference_seconds"],
                "evaluation_device": r["environment"]["device_name"],
                "python": r["environment"]["python"],
                "torch": r["environment"]["torch"],
                "git_commit": r["git_commit"],
                "checkpoint": r["checkpoint"]["path"],
            }
        )
    (OUT / "results.csv").write_text(stream.getvalue(), encoding="utf-8")
    (FIG / "comparison_test.csv").write_text(to_csv(aggregate(records)), encoding="utf-8")
    (FIG / "comparison_full_test.csv").write_text(to_full_csv(records), encoding="utf-8")
    downloads = WEB / "downloads/a1"
    downloads.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in ["Placeholder_A1_Report.pdf", "Placeholder_A1_Slide.pptx"]:
        source, target = ROOT / ".tmp" / name, downloads / name
        if source.exists():
            shutil.copyfile(source, target)
        if not target.exists():
            raise FileNotFoundError(f"Place {name} in .tmp/ before the first build")
        manifest.append(
            {
                "file": name,
                "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    (downloads / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    sections = []
    sections.append(
        section(
            "problem",
            "Problem statement",
            """<p>Can image structure improve classification beyond weighted sums of pixels? We compare Linear, MLP, CNN, GRU, LSTM, and Transformer classifiers on the same Fashion-MNIST split. Each image receives one of 10 clothing labels. Low resolution and similar garment outlines make Shirt, Coat, and Pullover difficult to distinguish.</p><p>Fashion-MNIST is the main study. MNIST is used for development and debugging; CIFAR-10 is an optional extension. All reported runs use seed 42 and a 50-epoch budget.</p>""",
        )
    )
    content = """<p><a href="https://github.com/zalandoresearch/fashion-mnist">Fashion-MNIST (Zalando Research)</a> contains 70,000 labeled product images: 60,000 official training images and 10,000 test images. Inputs are 28 × 28 grayscale pixels; labels cover 10 clothing categories. We use the authors' MIT-licensed distribution in compressed IDX format.</p><p>Every class has 6,000 official training images and 1,000 test images, giving a maximum-to-minimum class ratio of 1.00. Class weighting is not needed to compensate for unequal class counts. Our random training/validation split is not stratified, so its per-class counts vary slightly.</p><div class="gallery gallery-pair">"""
    for name, alt, caption in [
        (
            "class_distribution",
            "Fashion-MNIST class counts",
            "All 10 classes have 6,000 official training images.",
        ),
        (
            "representative_samples",
            "Real Fashion-MNIST examples from each class",
            "Training examples show within-class variation and similar garment outlines.",
        ),
        (
            "class_mean_images",
            "Fashion-MNIST class-average images",
            "Averaging reveals recurring silhouettes and suppresses individual details.",
        ),
        (
            "class_similarity",
            "Similarity of centered class-average images",
            "Centered prototypes have cosine similarity 0.86 for Pullover–Coat. Similarity describes the data; it does not establish the cause of an error.",
        ),
    ]:
        content += figure("images/eda/fashion_mnist/" + name, alt, caption)
    content += """</div><p>About 53.0% of training pixels fall in the lowest intensity bin (below approximately 8/255). With intensity above 32 treated as foreground, mean coverage ranges from 24.5% for Sandal to 59.8% for Pullover. Similar silhouettes motivate examining upper-body garment confusions.</p><p><a href="data/a1/eda_summary.json">Download EDA statistics</a>. Other dataset sources: <a href="https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html">MNIST</a> and <a href="https://cave.cs.toronto.edu/kriz/cifar.html">CIFAR-10</a>.</p>"""
    sections.append(section("dataset", "Dataset and EDA", content))
    content = table(
        ["Dataset", "Role", "Input (C × H × W)", "Train", "Validation", "Test"],
        [
            ["Fashion-MNIST", "Main comparison", "1 × 28 × 28", "54,000", "6,000", "10,000"],
            ["MNIST", "Development", "1 × 28 × 28", "54,000", "6,000", "10,000"],
            ["CIFAR-10", "Extension", "3 × 32 × 32", "45,000", "5,000", "10,000"],
        ],
    )
    content += """<p>A seed-42 random permutation reserves 10% of the official training set for validation. All models share the same indices within each dataset. The official test split is used after selecting the checkpoint with the lowest validation cross-entropy. Training and validation indices do not overlap; cross-split duplicate or product-identity checks have not been completed.</p><p>Training uses Adam (learning rate 0.001, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸), batch size 128, float32, no weight decay, no scheduler, and no early stopping. Every model trains for 50 epochs. This is a fixed common recipe rather than a separate hyperparameter search for each architecture.</p><p>Grayscale training applies random rotation within ±10° and translation up to 10%, then normalization with mean/std 0.5. CIFAR-10 uses a random 32 × 32 crop with 4-pixel padding and horizontal flipping, then channel means (0.4914, 0.4822, 0.4465) and standard deviations (0.2470, 0.2435, 0.2616). Validation and test use the same normalization without augmentation.</p><p>A Fashion-MNIST batch contains float32 images [B, 1, 28, 28] and integer labels [B], with B = 128 except for the final partial batch. Normalized grayscale values lie in [−1, 1].</p>"""
    sections.append(section("setup", "Experimental setup", content))
    content = """<div class="pipeline" role="img" aria-label="Pipeline from raw images to preprocessing, data loading, model, loss, optimization, checkpoint selection, and test evaluation"><span>Raw images</span><span>Preprocessing</span><span>DataLoader</span><span>Model → logits</span><span>Cross-entropy → Adam</span><span>Validation → checkpoint</span><span>Test prediction → metrics</span></div>"""
    content += table(
        ["Model", "Representation and architecture", "Main assumption or limit"],
        [
            ["Linear", "Flatten pixels → 10 logits", "Affine baseline; no learned local filters."],
            [
                "MLP",
                "Flatten → 256 → 128 → 10; ReLU and dropout 0.2",
                "Nonlinear pixel combinations; no spatial weight sharing.",
            ],
            [
                "CNN",
                "Two 3 × 3 convolution blocks (32/64 channels), BatchNorm, ReLU, 2 × 2 max pooling; flatten → 128 → 10",
                "Shared local filters; flattening retains position. CNN denotes the spatial variant throughout this page.",
            ],
            [
                "GRU",
                "Image rows → one recurrent layer, hidden size 128 → final state → 10",
                "Shared updates across ordered rows; final-state bottleneck.",
            ],
            [
                "LSTM",
                "Same row sequence and hidden size; gated hidden/cell states → final hidden state → 10",
                "Memory across rows; sequential processing.",
            ],
            [
                "Transformer",
                "Row projection to 128 + sinusoidal positions → 2 encoder layers, 4 heads → mean pooling → 10",
                "Attention connects rows directly; less built-in local spatial structure.",
            ],
        ],
    )
    content += """<p>Grayscale row models use 28 steps with 28 values per step. CIFAR-10 gives 32 steps with 96 values per step. Transformer feed-forward width is 2,048 with dropout 0.1. The group implements model assembly and recurrent cell updates; affine, convolution, normalization, pooling, and Transformer encoder layers use PyTorch modules.</p><p>All models minimize cross-entropy directly from logits without a preceding softmax. Accuracy is the fraction of correct labels; macro-F1 averages F1 across 10 classes; recall measures the correctly predicted fraction of each true class. Parameter counts include encoder and classification head.</p><p><strong>Hypothesis:</strong> nonlinear and structure-aware representations improve on Linear under the common recipe. The changed factor is the model configuration; split, seed, preprocessing, optimizer, epoch budget, and checkpoint rule stay fixed. We assess accuracy, macro-F1, class errors, parameter count, and recorded compute cost. Architecture, size, and regularization change together, so this does not isolate one component's causal effect.</p>"""
    sections.append(section("methodology", "Methodology", content))
    content = "<p>Selected checkpoints are evaluated on the official 10,000-image test sets. These are single-seed results. CNN uses the spatial variant on all three datasets.</p>"
    for dataset, title in DATASETS.items():
        rows = []
        for m, b in bundles[dataset].items():
            r = b["result"]
            rows.append(
                [
                    MODELS[m],
                    f"{100 * r['metrics']['accuracy']:.2f}%",
                    f"{r['metrics']['macro_f1']:.4f}",
                    f"{r['parameter_count']:,}",
                    r["checkpoint"]["epoch"],
                ]
            )
        t = table(
            ["Model", "Test accuracy", "Macro-F1", "Parameters", "Best epoch"],
            rows,
            title + " test results",
        )
        role = "development" if dataset == "mnist" else "extension"
        content += (
            f"<h3>{title} — main comparison</h3>{t}"
            if dataset == "fashion_mnist"
            else f"<details><summary>{title} — {role} results</summary>{t}</details>"
        )
    content += '<p><a href="data/a1/results.csv" download>All 18 results (CSV)</a> · <a href="data/a1/experiments.json">Run records, histories, and error summaries (JSON)</a></p>'
    sections.append(section("results", "Results", content))
    content = "<p>Saved durations for the main Fashion-MNIST runs are listed below. Training is the recorded total training duration. Inference is accumulated model forward-pass time over 10,000 test images at batch size 128, excluding data loading, device copies, softmax, and metric calculation.</p>"
    rows = []
    for m, b in bundles["fashion_mnist"].items():
        r = b["result"]
        rows.append(
            [
                MODELS[m],
                f"{r['training_seconds']:.2f}",
                f"{r['inference_seconds']:.4f}",
                html.escape(r["environment"]["device_name"].replace("NVIDIA GeForce ", "")),
                html.escape(r["environment"]["python"] + " / " + r["environment"]["torch"]),
            ]
        )
    content += table(
        [
            "Model",
            "Training (s)",
            "Test forward passes (s)",
            "Recorded evaluation GPU",
            "Python / PyTorch",
        ],
        rows,
    )
    content += """<p class="note">Timings use different hardware and lack a shared warm-up and repeated-trial protocol. They document compute cost but do not support a speed ranking. The GPU column comes from evaluation metadata, not a complete training-hardware record.</p><p>The LSTM training record explicitly identifies CUDA on RTX 4060 Laptop GPU, Python 3.12.14, PyTorch 2.14.0+cu126, torchvision 0.29.0+cu126, and CUDA build 12.6. Other JSON records identify their evaluation environments; a complete per-run training-environment audit remains a limitation.</p>"""
    sections.append(section("compute", "Recorded compute cost", content))
    content = "<p>Curves use exactly the runs in the result tables. Dots mark the lowest validation loss. Training uses augmentation and configured dropout; validation does not. A training–validation gap alone does not diagnose underfitting.</p>"
    for dataset, title in DATASETS.items():
        h = bundles[dataset]["cnn"]["history"]
        part = (
            '<div class="gallery">'
            + figure(
                f"images/comparison/a1/{dataset}_learning",
                f"Six-model losses on {title}",
                f"{title}: 50 epochs, seed 42, identical split within the dataset.",
            )
            + "</div>"
        )
        part += f"<p>CNN reaches its lowest validation loss of {h['best_validation_loss']:.4f} at epoch {h['best_epoch']}. Its final training/validation losses are {h['train_loss'][-1]:.4f}/{h['validation_loss'][-1]:.4f}.</p>"
        if dataset == "cifar10":
            part += "<p>Transformer validation loss reaches its minimum at epoch 30 and then rises as training loss falls, consistent with overfitting in that run. CNN has the lowest minimum validation loss in the updated six-model comparison.</p>"
        content += (
            f"<h3>{title}</h3>{part}"
            if dataset == "fashion_mnist"
            else f"<details><summary>{title} learning curves</summary>{part}</details>"
        )
    sections.append(section("learning", "Learning curves", content))
    sections.append(
        section(
            "comparison",
            "Comparison and discussion",
            """<p>On Fashion-MNIST, MLP improves on Linear by 10.13 percentage points, supporting the value of nonlinear pixel combinations. CNN adds another 5.41 points and leads Transformer by 2.16 points while using 421,738 rather than 1,191,050 parameters. Local filters and spatial feature maps are a plausible advantage for product silhouettes, although size and regularization also differ.</p><p>GRU reaches 89.88% with 61,962 parameters: 2.71 points below CNN with approximately 6.8 times fewer parameters. It is a smaller alternative when weight storage matters. Recurrent models summarize rows through a final state; Transformer relates rows directly, but its greater capacity does not yield the highest accuracy under this recipe.</p><p>The extension follows the same broad pattern: CNN reaches 79.02% on CIFAR-10, ahead of Transformer at 68.71% by 10.31 points. CNN macro-F1 is 0.7880. MNIST development results are much closer: CNN 99.41% and Transformer 99.31%. These results describe the tested configurations, not the best achievable performance of each family.</p><p>Accuracy and parameter count support an accuracy–storage comparison. Timings cannot establish an accuracy–speed trade-off because hardware and measurement conditions differ.</p>""",
        )
    )
    content = """<p>Confusion-matrix rows are true classes and columns are predictions. Cell labels give counts; color gives the fraction of each true class. Recall tables use all 10,000 test images.</p><p>Fashion-MNIST CNN Shirt recall is 79.5%, versus 62.1% for MLP. Its strongest classes are Bag (99.2%) and Trouser (99.0%). There are 81 Shirt → T-shirt/top errors and 61 Coat → Pullover errors. Similar silhouettes suggest inspecting finer garment details; targeted augmentation or representation changes need controlled experiments to verify any improvement.</p>"""
    for dataset, title in DATASETS.items():
        group = bundles[dataset]
        classes = list(group["cnn"]["errors"]["per_class_accuracy"])
        part = table(
            ["True class"] + [MODELS[m] + " recall" for m in MODELS],
            [
                [html.escape(c)]
                + [f"{100 * group[m]['errors']['per_class_accuracy'][c]:.1f}%" for m in MODELS]
                for c in classes
            ],
        )
        if dataset == "cifar10":
            part += "<p>CNN identifies 91.8% of automobiles but 56.3% of cats. Its largest confusions are Cat → Dog (147) and Dog → Cat (136). Inspecting animal appearance and backgrounds may help explain these errors; their causes have not been tested.</p>"
        part += '<h4>Confusion matrices</h4><div class="gallery gallery-pair">'
        for m in MODELS:
            part += figure(
                f"images/comparison/a1/{dataset}_{m}_confusion",
                f"{MODELS[m]} confusion matrix on {title}",
                f"{MODELS[m]}. {title} test split.",
            )
        part += '</div><h4>Correct and incorrect predictions</h4><p>Each panel shows 5 highest-confidence correct predictions or errors. Models use different examples; these selections do not estimate error frequency. A confident mistake alone does not establish poor calibration.</p><div class="gallery">'
        part += figure(
            f"images/comparison/a1/{dataset}_cnn_correct",
            f"CNN correct predictions on {title}",
            "CNN: highest-confidence correct predictions.",
        )
        part += '</div><details><summary>Highest-confidence errors for all 6 models</summary><div class="gallery gallery-pair">'
        for m in MODELS:
            part += figure(
                f"images/comparison/a1/{dataset}_{m}_errors",
                f"{MODELS[m]} errors on {title}",
                f"{MODELS[m]}: highest-confidence errors.",
            )
        part += "</div></details>"
        content += (
            f"<h3>{title}</h3>{part}"
            if dataset == "fashion_mnist"
            else f"<details><summary>{title} error analysis</summary>{part}</details>"
        )
    sections.append(section("error-analysis", "Error analysis", content))
    sections.append(
        section(
            "conclusion",
            "Conclusion and limitations",
            """<p>CNN is the strongest tested configuration for Fashion-MNIST accuracy and macro-F1, while GRU offers a smaller model with a moderate accuracy cost. The results support using image structure under this recipe. They do not establish a universal architecture ranking or speed advantage.</p><p>Only one seed is used per model. Architecture, size, and regularization vary together; no component ablation or systematic hyperparameter search isolates their contributions. Product duplicates across splits have not been audited. Small product images do not represent full scenes or unseen categories. Future work should repeat seeds, control representation or model size, inspect garment errors, and benchmark inference under matched conditions.</p>""",
        )
    )
    content = "<p>Run commands from the repository root. Datasets download automatically. Use matching dataset/config pairs. Training writes the selected checkpoint before evaluation reads it.</p>"
    content += code(
        "uv sync --locked --dev\nuv run python -m scripts.eda.run_fashion_mnist\n\n# Repeat for: linear mlp cnn gru lstm transformer\nuv run python -m scripts.training.train_image_classification --data fashion_mnist --config configs/image_classification/fashion_mnist/cnn.yaml\nuv run python -m scripts.evaluation.evaluate --data fashion_mnist --config configs/image_classification/fashion_mnist/cnn.yaml"
    )
    content += """<p>The CNN checkpoint is <code>checkpoints/fashion_mnist_cnn_spatial_seed42/best.pt</code>; other Fashion-MNIST run names use <code>fashion_mnist_&lt;model&gt;_seed42</code>. MNIST and CIFAR-10 use matching configuration directories. Reconstruction requires training; checkpoint weights are not bundled with this site.</p><p><a href="data/a1/experiments.json">Experiment records</a> provide configurations, seed, checkpoint path/epoch, saved revision, histories, and error summaries for all 18 selected runs. Some recorded revisions have a <code>-dirty</code> suffix: uncommitted changes existed at evaluation time, so those hashes alone do not identify the full historical source state. Local prediction tensors must be regenerated from checkpoints.</p>"""
    content += f'<p><a href="{SOURCE}">A1 integration code</a> · <a href="{REPO}/blob/a2/m1/README.md">README</a>. To rebuild this page and the selected-run figures after evaluation:</p>'
    content += code("uv run python -m scripts.maintenance.build_a1_web")
    sections.append(section("limitations", "Reproduction and run records", content))
    content = f'''<ul class="links"><li><strong>Final report:</strong> <a href="downloads/a1/Placeholder_A1_Report.pdf">Read PDF</a> · <a href="downloads/a1/Placeholder_A1_Report.pdf" download>Download PDF</a></li><li><strong>Presentation slides:</strong> <a href="downloads/a1/Placeholder_A1_Slide.pptx" download>Download PowerPoint</a></li><li><strong>YouTube presentation:</strong> <a href="https://www.youtube.com/watch?v=nQdd3HrIEeE">Watch the group presentation</a></li><li><strong>Source code:</strong> <a href="{SOURCE}">A1 integration branch</a></li><li><strong>Checkpoints:</strong> <a href="#limitations">Training and reconstruction instructions</a></li><li><strong>AI usage:</strong> <a href="#ai-usage">Assignment 1 disclosure</a> · <a href="data/a1/AI_USAGE.md" download>Detailed log</a></li><li><strong>Results:</strong> <a href="data/a1/results.csv" download>CSV</a> · <a href="data/a1/experiments.json">JSON evidence</a></li></ul><div class="video-frame"><iframe src="https://www.youtube-nocookie.com/embed/nQdd3HrIEeE" title="Group Placeholder — Assignment 1 presentation" loading="lazy" referrerpolicy="strict-origin-when-cross-origin" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe></div>'''
    sections.append(section("links", "Report, slides, and presentation", content))
    ai = (
        (ROOT / "AI_USAGE.md")
        .read_text(encoding="utf-8")
        .split("## Assignment 1", 1)[1]
        .split("## Assignment 2", 1)[0]
    )
    entries = re.split(r"(?m)^### Entry ", ai)[1:]
    content = "<p>The group used Codex/GPT and Claude tools for scaffolding, coding and integration support, model review, EDA, test generation, report preparation, slides, video narration, and this website. AI generated some code and text as well as reviewing existing work. Members remain responsible for understanding and verifying their submitted contributions.</p>"
    for entry in entries:
        number = entry.splitlines()[0]
        rows = []
        user = "Group use"
        for line in entry.splitlines()[1:]:
            match = re.match(r"- \*\*(.+?):\*\*\s*(.*)", line)
            if match and match[2].strip():
                rows.append(
                    [
                        html.escape(match[1]),
                        re.sub(
                            chr(96) + r"([^" + chr(96) + r"]+)" + chr(96),
                            r"<code>\1</code>",
                            html.escape(match[2]),
                        ),
                    ]
                )
                if match[1] == "Used by":
                    user = match[2]
        fences = chr(96) * 3
        prompts = re.findall(fences + r"\s*\n(.*?)\n" + fences, entry, flags=re.S)
        content += (
            f"<details><summary>Entry {html.escape(number)} — {html.escape(user)}</summary>"
            + table(["Field", "Disclosure"], rows)
        )
        for prompt in prompts:
            content += "<p>Representative prompt:</p>" + code(prompt)
        content += "</details>"
    content += '<p>The detailed log records which checks were performed and identifies the member responsible for final review.</p><p><a href="data/a1/AI_USAGE.md" download>Download the complete AI usage log</a>.</p>'
    sections.append(section("ai-usage", "AI Usage Disclosure — Assignment 1", content))
    sections.append(
        section(
            "members",
            "Group members",
            table(
                ["Member", "Student ID", "Main A1 contribution"],
                [
                    ["Vũ Đức Việt Anh", "2352074", "Website, training pipeline, LSTM"],
                    [
                        "Trần Lâm Anh",
                        "2352067",
                        "Data loading, preprocessing, EDA, Transformer, metrics",
                    ],
                    ["Lê Nguyên Khang", "2352470", "Error analysis, GRU, shared utilities, review"],
                    ["Trần Nguyên Giáp", "2352284", "Linear/CNN models, evaluation pipeline"],
                ],
            ),
        )
    )
    old = (WEB / "assignment1.html").read_text(encoding="utf-8")
    prefix = old[: old.index("</header><main") + len("</header>")]
    prefix = prefix.replace('<html lang="en">', '<html lang="en" class="a1-page">')
    nav_items = [
        ("problem", "Problem"),
        ("dataset", "Dataset and EDA"),
        ("setup", "Experimental setup"),
        ("methodology", "Methodology"),
        ("results", "Results"),
        ("compute", "Compute cost"),
        ("learning", "Learning curves"),
        ("comparison", "Discussion"),
        ("error-analysis", "Error analysis"),
        ("conclusion", "Conclusion"),
        ("limitations", "Reproduction"),
        ("links", "Report, slides, video"),
        ("ai-usage", "AI disclosure"),
        ("members", "Group members"),
    ]
    nav = "".join(f'<a href="#{key}">{label}</a>' for key, label in nav_items)
    body = f'''<main id="main" class="report-shell"><aside class="contents"><p class="eyebrow">Assignment 1</p><nav aria-label="Report contents">{nav}</nav><a class="contents-repo" href="{SOURCE}">View source code</a></aside><article><header class="article-header"><p class="eyebrow">CO3133 · Semester 261 · Group Placeholder</p><h1>Image classification</h1><p class="article-subtitle">From linear models to modern sequence models: a comparative study on Fashion-MNIST.</p><p class="course-details">Deep Learning and Its Applications<br>Ho Chi Minh City University of Technology – VNU-HCM<br>Faculty of Computer Science and Engineering · Instructor: Lê Thành Sách</p><div class="deliverable-actions"><a class="button" href="downloads/a1/Placeholder_A1_Report.pdf">Read report</a><a href="downloads/a1/Placeholder_A1_Slide.pptx" download>Download slides</a><a href="https://www.youtube.com/watch?v=nQdd3HrIEeE">Watch video ↗</a></div></header>'''
    body += "\n\n".join(sections)
    body += "</article></main><footer>CO3133 – Deep Learning and Its Applications · Semester 261 · Group Placeholder</footer></body></html>\n"
    (WEB / "assignment1.html").write_text(prefix + body, encoding="utf-8")
    print(f"Built A1 page: {len(records)} runs, {len(entries)} AI entries, 2 final downloads")


if __name__ == "__main__":
    main()
