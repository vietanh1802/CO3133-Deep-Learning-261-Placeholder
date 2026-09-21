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

## Assignment 2

## Assignment 3
