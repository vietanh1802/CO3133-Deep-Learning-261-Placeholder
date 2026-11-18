# AI Usage Disclosure

## Assignment 1

<!-- One entry per tool use. Template below. -->

### Entry 1

- **Tool / model:** GPT-5.6 Sol (Light)
- **Used by:** Le Nguyen Khang (2352470)
- **Date / development stage:** 2026-09-15 - Scaffolding
- **Purpose:** Implement the structure of the project with blank files to ensure consistency.
- **Affected sections / files:** All source files.
- **Representative prompt:**

```
Scaffold the blank project. DO NOT implement the functions or classes in the source files.
Technical requirements: The blank project should have different composable modules for training, validation, configuration, utilities.
Use factory pattern to assign different models and datasets without modifying the core training loop and avoid code duplication (BAD CODE).
Write the blank functions and classes in the source files. DO NOT implement.
Behavioral requirements: Use uv, ruff, and ty check for blazing fast development. Pre-commit hooks for code quality.
```

- **How the output was edited and verified:** The whole structure of the codebase is generated while leaving the main code untouched. These are left blank to be implemented with care and knowledge by other students. GPT does not guide the implementation.

### Entry 2

- **Tool / model:** Codex
- **Used by:** Le Nguyen Khang (2352470)
- **Date / development stage:** 2026-09-21 - Assignment 1 draft integration
- **Purpose:** Connect and verify the Fashion-MNIST draft pipeline, including the MLP, classification head, training entry point, configurations, tests, experiment runs, and draft result text.
- **Affected sections / files:** Dataset exports; model and configuration modules; training and evaluation scripts; tests; Assignment 1 page; draft report; generated EDA and experiment artifacts.
- **Representative prompt:** "Become the AI graders and check for me what have we done in progress and what are the remaining tasks that do we have to do? Score us over the scale of 10."
- **How the output was edited and verified:** Ruff, ty, and the full test suite were run. The MLP was checked by overfitting one real batch. Linear and MLP were trained on the same seed-42 split, selected by validation loss, and evaluated once on the official test split. All reported values were copied from saved JSON result records.

### Entry 3

- **Tool / model:** Claude Sonnet 5 (Claude Code)
- **Used by:** Tran Lam Anh (2352067)
- **Date / development stage:** 2026-09-23 - Assignment 1 exploratory data analysis
- **Purpose:** Extend the EDA beyond the four items required in Section 13 of the course handbook, and remove the duplication between the two dataset entry points. Added train-vs-test split comparison, pixel-intensity distribution, per-class average images, a class-prototype similarity matrix, per-class foreground coverage, and a machine-readable `summary.json`.
- **Affected sections / files:** `src/data/eda.py`, `src/data/eda_plots.py`, `src/data/eda_report.py`, `scripts/eda/run_fashion_mnist.py`, `scripts/eda/run_mnist.py`, `tests/test_eda.py`, `README.md`, report Part 1 (Problem and data), and the regenerated Fashion-MNIST EDA figures.
- **Representative prompt:** "I think there should be more EDA to see more about data, what should be them? List out so that I can choose."
- **AI contribution:** The student set the goal, the scope and the target branch, and decided which of the proposed analyses to keep and what to report. The model proposed the candidate analyses, explained what each one shows, and wrote the implementation, the unit tests and the accompanying text.
- **How the output was edited and verified:** Ruff lint and format and the full test suite (57 tests) were run, including 17 new unit tests that check every statistic against a six-image toy dataset with hand-computed expected values. Both EDA entry points were executed end to end on the real datasets and every generated figure was inspected. One figure drafted for the report (the share of pixels in the darkest intensity bin) was wrong and was corrected against `summary.json` before committing. The similarity matrix was cross-checked against the confusion pairs already reported in the draft error analysis, and the MNIST run against the well-known 4/9 and 7/9 confusions.

### Entry 4

- **Tool / model:** Claude Opus 5.5 (Medium)
- **Used by:** Tran Nguyen Giap (2352284)
- **Date / development stage:** 2026-10-26 - Models 
- **Purpose:** Check my implementation of some models. 
- **Affected sections / files:** Models/ folder
- **Representative prompt:**

```
Here is my implementation of Linear, RNN, and CNN. Check if they are correct and mark what is wrong.
```

- **How the output was edited and verified:** The implementation is compared with standard library like timm, pytorch models. 

### Entry 5

- **Tool / model:** Codex (GPT-6).
- **Used by:** Tran Lam Anh (2352067).
- **Date / development stage:** 2026-10-27 - Assignment 1, Part 1 report preparation.
- **Purpose:** Complete Problem and Data Description in the existing group LaTeX report, covering the task, dataset, EDA, split protocol, and preprocessing.
- **Affected sections / files:** `reports/a1/Sections/3-Problem-and-Data.tex`, `Tables/data_splits.tex`, two report images, supporting references, and this use's report disclosure.
- **Representative prompt (translated):** "Write report Part 1 as planned, then let me review it. Push only after I approve."
- **AI contribution:** Drafted and edited the English text and table; prepared a compact grid of real Fashion-MNIST images and a heatmap of the saved class-similarity statistics. Added source citations and prepared a PDF review copy.
- **Verification:** The assistant checked counts, seed, split logic, normalization, and augmentation against the code and configurations; recomputed the saved EDA mean, standard deviation, and lowest-bin pixel share from the local dataset; checked the four download checksums; and inspected an actual validation batch. The student approved publication on 2026-10-27. No training or model-evaluation run was added for this writing task.

### Entry 6

- **Tool / model:** Codex
- **Used by:** Vu Duc Viet Anh (2352074)
- **Date / development stage:** Assignment 1 - LSTM pipeline and training-code review.
- **Purpose:** Use AI as a supporting code-review tool to check the existing LSTM pipeline and image-classification training code for possible bugs or inconsistencies. The scope was reviewing existing code and understanding potential issues.
- **Affected sections / files:** `src/models/lstm.py`, `scripts/training/train_image_classification.py`.
- **Representative prompt (paraphrased):**

```
Please check my existing LSTM pipeline and the Assignment 1 training code in scripts/training for possible errors and explain the issues you identify.
```

- **How the output was edited and verified:** Codex provided review feedback on the existing implementation. Its role was to support code checking, rather than supply a complete replacement pipeline. Specific code edits and verification runs are not recorded for this review.

### Entry 7

- **Tool / model:** Codex
- **Used by:** Vu Duc Viet Anh (2352074)
- **Date / development stage:** Assignment 1 - Model unit-test development.
- **Purpose:** Use AI to generate supporting unit-test code for checking whether the existing Assignment 1 model implementations behave as expected.
- **Affected sections / files:** `tests/components/test_models.py`.
- **Representative prompt (paraphrased):**

```
Generate unit tests in tests/test_models.py to check whether the Assignment 1 models behave as expected. Include assertions that can reveal implementation errors.
```

- **AI contribution:** Drafted test cases and assertions to support validation of the existing models. The AI-generated code for this use concerns the tests.
- **How the output was edited and verified:** The current test file includes checks for model-factory dispatch, output shapes, forward/backward behavior, finite loss and gradients, invalid inputs, checkpoint reloading, and comparisons of custom LSTM/GRU outputs with PyTorch under matched weights. No test-run result is recorded for this use.

## Assignment 2

### Entry 1

- **Tool / model:** Codex (GPT-6).
- **Used by:** Tran Lam Anh (2352067).
- **Date / development stage:** 2026-10-27 - DeepLabV3 implementation and testing.
- **Purpose:** Explain segmentation inputs/outputs and implement the pretrained SUIM model, fine-tuning configuration, separate backbone/head optimizer rates, and integration tests.
- **Affected sections / files:** `src/models/deeplabv3.py`, model exports, segmentation config and optimizer, training entry point, `configs/segmentation/suim/pretrained.yaml`, README, and DeepLabV3 tests.
- **Representative prompt (translated):** "Give me the knowledge I need, then implement and test. I will review before committing, pushing, and creating a PR."
- **AI contribution:** Explained tensor shapes, logits, transfer learning, optimizer parameter groups, and BatchNorm behavior; wrote the implementation and tests. BatchNorm running statistics are fixed by default while its affine parameters and the remaining network still receive gradients.
- **Verification:** `uv run --locked python -m scripts.check` passed lint, formatting, type checking, and 157 CPU tests. Tests cover real randomly initialized DeepLabV3 forward/backward, singleton batches, full-resolution outputs, and BatchNorm statistics. Download-free tests verify weight selection and optimizer groups. A synthetic SUIM integration test uses a small substitute backbone for training, interruption/resume, checkpoint selection, evidence export, and evaluation. A separate CPU smoke check loaded the actual `COCO_WITH_VOC_LABELS_V1` weights and verified forward/backward on one synthetic image. No SUIM training experiment or GPU benchmark was run; student review is pending.

### Entry 2

- **Tool / model:** Claude Opus 5.5 (Claude Code).
- **Used by:** Vu Duc Viet Anh (2352074)
- **Date / development stage:** 2026-10-04 to 2026-11-09 - A2.M1 dataset proposal, SUIM data audit, EDA and split.
- **Purpose:** Understand what the dataset proposal must contain; use AI as a source of reference code and as a code reviewer for the SUIM measurement functions, the EDA runner and the fixed train/validation split; later, check the data audit against the original SUIM release and the Dataset Ninja copy, and extend the EDA with audit examples, storage size and an example batch.
- **Affected sections / files:** `reports/a2_proposal.md`; image and mask pairing and measurement functions in `src/data/suim.py`; `scripts/eda/run_suim.py`; `src/eda/suim.py`; `results/a2/eda/suim/`; `results/a2/splits/suim/` (commits `e218c44`, `85ab66d` and `dc02a75`).
- **Representative prompts (the first is reconstructed because the 2026-10-04 session log was not kept; the others are translated from the session log):**

```
Read the handbook and explain what the A2.M1 dataset proposal must contain. Then give me reference code for pairing SUIM images with their masks, auditing the mask colours and making a group-aware train/validation split, and check my implementation.
Do I need example figures in the EDA for cases such as images duplicated between train and test, or images among the 85 faulty masks?
OK, do that part.
Why put (a) and (b) in one figure? I want one figure for (a) and one for (b), each with two examples.
Remove sha256, then check whether the EDA code still has unused variables.
```

- **AI contribution:** For the proposal, explained the requirements and prepared a Vietnamese reading copy that lists where each number comes from, which the member used to check the submitted English version. For the first audit, EDA and split code, provided reference code and reviewed the member's implementation; the member wrote, ran and committed that code. For the EDA extension, compared the two dataset copies, explained how the test duplicates were found, suggested the audit examples, and wrote the requested additions and clean-up in the EDA runner. The member decided which additions to make, changed the figure layout and removed the fields the member judged unnecessary.
- **How the output was edited and verified:** Each number in the proposal was traced to the EDA summary produced by the member's run. The split files were committed once and are read unchanged by every later training run. The member compared the regenerated EDA figures and `summary.json` with the previous outputs before committing. `tests/contracts/test_suim.py`, `tests/contracts/test_segmentation_data.py` and `tests/components/test_eda.py` check mask decoding, resizing, joint image and mask flips, use of the split files and the EDA summaries; they passed in the repository's continuous-integration run on 2026-11-11.
- **Member responsible for final verification:** Vu Duc Viet Anh (2352074).
- **Sources used for verification:** Course handbook (revision 14 Sep 2026), Assignment 2 dataset requirements; the original SUIM release and the Dataset Ninja copy (local); Islam et al., IROS 2020 (arXiv:2004.01241); `results/a2/eda/suim/summary.json`; the repository tests listed above.

### Entry 3

- **Tool / model:** Claude Opus 5.5 (Claude Code).
- **Used by:** Vu Duc Viet Anh (2352074)
- **Date / development stage:** 2026-11-08 to 2026-11-11 - U-Net baseline configuration check and one-time test evaluation of the selected checkpoints.
- **Purpose:** Check that the committed U-Net configuration matches the reported run, and score the selected U-Net and DeepLabV3 checkpoints once on the official test split.
- **Affected sections / files:** `configs/segmentation/suim/baseline.yaml` and `scripts/report/a2_data_tables.py` (commit `8f4a22b`); `scripts/evaluation/test_segmentation.py`, `scripts/training/train_segmentation.py`, `results/a2/suim_unet_seed42/test_summary.json` and the two DeepLabV3 test summaries (commit `cfc83f3`).
- **Representative prompts (translated):**

```
What differs between what I wrote in the proposal and what my teammates are running, for example the different GPUs and batch sizes you mentioned?
I have put the DeepLabV3 checkpoints in .tmp/checkpoint-a2. Check whether I can now run them on the test set.
OK, I agree with your approach. Start revising section 2.6 and run the test set.
```

- **AI contribution:** Found that the committed U-Net batch size differed from the reported run, restored it and added a check that compares the committed configurations with the run records. Wrote the one-time test script and ran it on the three selected checkpoints. The member chose which training run to report, approved the hypothesis and the test plan before the test split was used, and reviewed the test results for anomalies.
- **How the output was edited and verified:** The test script refuses a checkpoint whose epoch differs from the selected epoch in `training_summary.json` and refuses to overwrite an existing test summary, so each model is scored once. Each `test_summary.json` records the SHA-256 hash of the scored checkpoint. The full test suite passed in the repository's continuous-integration run on 2026-11-11.
- **Member responsible for final verification:** Vu Duc Viet Anh (2352074).
- **Sources used for verification:** `training_summary.json` and `history.json` of each run; the committed configurations; course handbook, Assignment 2 experimental requirements (selection on validation, single test evaluation).

### Entry 4

- **Tool / model:** Claude Opus 5.5 (Claude Code).
- **Used by:** Tran Nguyen Giap (2352284)
- **Date / development stage:** 2026-10-26 - A2.M2 U-Net baseline implementation and first training run.
- **Purpose:** Use AI as a source of reference code and as a code reviewer for the U-Net baseline, the segmentation training loop and the minimal SUIM loader written to train it.
- **Affected sections / files:** `src/models/unet.py`; segmentation training in `src/training/segmentation.py` and `scripts/training/train_segmentation.py`; `src/evaluation/compute_cost.py`; `configs/segmentation/suim/baseline.yaml`; `results/a2/suim_unet_seed42/` (commits `ae6d38e` and `a66bf14`; later restructured with the rest of the repository).
- **Representative prompt (reconstructed; the session log was not kept):**

```
Here is my U-Net baseline and the segmentation training loop for SUIM. Give me a reference implementation to compare against, then check my code and mark what is wrong.
```

- **AI contribution:** Provided reference code and checked the member's implementation; the member wrote, trained and committed the final model.
- **How the output was edited and verified:** `tests/components/test_segmentation_model.py` checks the U-Net output shape, rejects input sizes the network cannot handle and compares the parameter count with the proposal. `tests/pipelines/test_segmentation_resume.py` checks the warm-up and cosine schedule, checkpoint selection and exact resumption. The committed run records software versions, git commit, seed and the checkpoint selection rule.
- **Member responsible for final verification:** Tran Nguyen Giap (2352284).
- **Sources used for verification:** The repository tests listed above and the committed run records in `results/a2/suim_unet_seed42/`.

### Entry 5

- **Tool / model:** Codex.
- **Used by:** Le Nguyen Khang (2352470)
- **Date / development stage:** 2026-10-25 to 2026-10-27 - Segmentation metrics, evaluation and two-GPU training.
- **Purpose:** Use AI as a coding assistant for the confusion-matrix metrics (IoU, Dice and per-class IoU), the evaluation command that exports metrics and qualitative examples, and the two-GPU training support used for the first DeepLabV3 run.
- **Affected sections / files:** `src/evaluation/segmentation.py`; `scripts/evaluation/eval_segmentation.py`; segmentation metric tests; `scripts/training/train_segmentation.py` and the restructured `src/config/` and `src/data/` modules; `results/a2/suim_deeplabv3_ce_seed42_2gpu/` (commits `01ed515`, `a856c5b` and `65bc040`).
- **Representative prompt (reconstructed; the session log was not kept):**

```
Implement IoU and Dice for SUIM from a confusion matrix accumulated over the whole split, with per-class IoU, and an evaluation command that saves the metrics and qualitative examples. Add tests that compare the metrics with hand-computed values.
```

- **AI contribution:** Assisted with the metric functions, the evaluation command, the two-GPU training support and their tests; the member reviewed, tested and committed the final code.
- **How the output was edited and verified:** `tests/components/test_segmentation_metrics.py` compares the metrics with hand-computed values, rejects invalid labels and mismatched masks, and checks the evaluation command. `tests/pipelines/test_distributed_segmentation.py` trains with two processes and checks that progress is combined and resumed correctly. Both passed in the repository's continuous-integration run on 2026-11-11.
- **Member responsible for final verification:** Le Nguyen Khang (2352470).
- **Sources used for verification:** The repository tests listed above and the saved run records in `results/a2/suim_deeplabv3_ce_seed42_2gpu/`.

### Entry 6

- **Tool / model:** Codex.
- **Used by:** Le Nguyen Khang (2352470)
- **Date / development stage:** 2026-11-07 to 2026-11-10 - Controlled study, error-analysis figures, assignment page and report revision.
- **Purpose:** Use AI as a coding and writing assistant for the one-factor ablation study, the ablation and error-analysis figures, the class colour distribution figure, and revisions of the Assignment 2 page and report.
- **Affected sections / files:** `scripts/evaluation/ablate_segmentation.py`; `configs/segmentation/suim/ablation.yaml`; `configs/plotting.toml`; `scripts/plots/plot_segmentation_ablation.py`; `scripts/plots/plot_segmentation_errors.py`; `scripts/eda/plot_suim_colour.py`; `results/a2/ablation/`; `web/assignment2.html`; `reports/a2/Sections/` (commits `07cb410`, `cd849e4`, `6032646`, `397fe21`, `88adde1` and `4600fd1`).
- **Representative prompt (reconstructed; the session log was not kept):**

```
Build an ablation study that changes one factor at a time from the DeepLabV3 reference configuration, runs each configuration and exports paired comparisons with the reference. Then plot the results and the error analysis for the report and the website.
```

- **AI contribution:** Assisted with the ablation planner and runner, the plotting scripts, and the text of the assignment page and report; the member reviewed, tested and committed the final code and text.
- **How the output was edited and verified:** `tests/components/test_segmentation_ablation.py` checks that each configuration changes exactly one factor from the reference, that invalid settings are rejected, that a frozen backbone stays unchanged while the head updates, and that low-resolution scores are resized to the common grid before the argmax. `tests/pipelines/test_segmentation_ablation.py` plans, runs, resumes and exports the paired results, and `tests/components/test_segmentation_plots.py` checks that the ablation figures match the saved measurements. These tests passed in the repository's continuous-integration run on 2026-11-11.
- **Member responsible for final verification:** Le Nguyen Khang (2352470).
- **Sources used for verification:** The repository tests listed above; `results/a2/ablation/comparisons.csv` and `results/a2/ablation/manifest.json`.

## Assignment 3

### Entry 1

- **Tool / model:** Claude Code.
- **Used by:** Tran Nguyen Giap (2352284)
- **Date / development stage:** 2026-11-18 - M1 proposal.
- **Purpose:** Assistance with the compute analysis of the proposal.
- **Affected sections / files:** `scripts/a3/`; `results/a3/compute/`; `reports/a3_proposal.md`.
- **Representative prompt (reconstructed):**

```
Help estimate the compute cost of the planned models for the proposal.
```

- **AI contribution:** Coding and writing assistance; the member reviewed the final version.
- **How the output was edited and verified:** The results were checked by rerunning the script.
- **Member responsible for final verification:** Tran Nguyen Giap (2352284).
- **Sources used for verification:** `results/a3/compute/`.
