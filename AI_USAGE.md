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

### Entry 8

- **Tool / model:** Codex
- **Used by:** Vu Duc Viet Anh (2352074)
- **Date / development stage:** 2026-10-28 - Assignment 1 presentation and website preparation.
- **Purpose:** Support slide updates, draft and revise the English YouTube narration for four speakers, and complete the Assignment 1 website using the final report, slides, and saved experiment results.
- **Affected sections / files:** Assignment 1 PowerPoint and narration files; `web/index.html`; `web/assignment1.html`; `web/data/a1/`; `web/downloads/a1/`; website figures; `scripts/maintenance/build_a1_web.py`; `scripts/evaluation/compare.py`; `tests/components/test_comparison.py`.
- **Representative prompt (translated):** "Complete the Assignment 1 website using Placeholder_A1_Report.pdf and Placeholder_A1_Slide.pptx in .tmp, and the YouTube link already provided."
- **AI contribution:** Drafted and edited narration, assisted with slide result updates, generated website text and HTML, selected saved experiment records, and prepared downloadable evidence and figures. Completed the shared landing-page directory and changed comparison selection to exclude legacy CNN runs. This included content generation as well as review assistance.
- **How the output was edited and verified:** Narration was revised according to the student’s speaker allocation and number-format instructions. Website figures and tables are built from an explicit list of 18 saved seed-42, 50-epoch runs, with the spatial CNN variant selected for every dataset. Automated checks validate record counts, checkpoint-selection losses, result values, links, and copied download checksums. These checks do not establish that a student has completed final review.
- **Member responsible for final verification:** Vu Duc Viet Anh (2352074); final student review of these website and presentation changes is pending.
- **Sources used for verification:** Saved test_result.json, training_summary.json, and test_error_analysis.json records; Fashion-MNIST EDA summary; model and evaluation source code; the supplied final report and slides.

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

## Assignment 3
