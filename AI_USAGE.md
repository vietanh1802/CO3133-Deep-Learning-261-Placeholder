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

- **Tool / model:** Codex (GPT-6)
- **Used by:** Vu Duc Viet Anh (2352074)
- **Date / development stage:** 2026-10-25 - Assignment 1 LSTM review and integration
- **Purpose:** Review the existing custom LSTM on local branch `dev/a1-lstm` and integrate it into the current A1 training and evaluation pipeline.
- **Affected sections / files:** `src/models/lstm.py`, `configs/image_classification/fashion_mnist/lstm.yaml`, `tests/test_lstm.py`, `train.sh`, `README.md`, and this disclosure.
- **Representative prompt:** "Help me integrate the LSTM code into the current pipeline and tell me which files change."
- **AI contribution:** Inspected the existing implementation and gate equations, compared outputs and gradients with `torch.nn.LSTM`, copied the reviewed model from commit `d03efff` without changing its equations, and added the Fashion-MNIST configuration, integration tests, and run instructions.
- **How the output was edited and verified:** The isolated review passed 100 combined tests on CPU. After integration, all 72 project tests passed, including 15 new LSTM cases; those 15 were rerun successfully after adding explicit type narrowing in the tests. Full-project Ruff lint, format checks, and ty passed. The integration test exercised the real training and evaluation entry points using small synthetic loaders, including checkpoint selection/reload, metrics, JSON records, evidence tensors, and figures. No full Fashion-MNIST training or final test results are claimed.

## Assignment 2

## Assignment 3
