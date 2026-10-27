# Assignment 1 report outline

Compile `main.tex`. Write in English and replace each blue prompt with report text.
The outline keeps the three main parts already used by the report.

| File in `Sections/` | What to write |
| --- | --- |
| `2-Abstract.tex` | Question, protocol, main finding, trade-off, and limitation. Write last. |
| `3-Problem-and-Data.tex` | Task, dataset, EDA, splits, and preprocessing. |
| `4-Methodology.tex` | Pipeline, architectures, metrics, training settings, and comparison protocol. |
| `5-Implementation-Results.tex` | Test comparison, learning behavior, errors, shared predictions, discussion, and conclusion. |
| `6-References.tex` | Sources cited in the report. |
| `7-Reproducibility.tex` | Run records, environment, commands, deliverables, and contributions. |
| `8-AI-Usage.tex` | Actual AI use and human review. |
| `9-Model-Evidence.tex` | Supporting figures included by the reproducibility appendix. |

Edit the cover information in `metadata.tex`. Keep shared formatting in
`preamble.tex` and `hcmut.sty`.

## Evidence to use

Store all report images directly in `Images/` and all table files in `Tables/`.
Model images use a model prefix, such as `cnn_learning_curves.png`, to avoid duplicate names.
Use `\reportfigure{Images/cnn_learning_curves.png}{Caption}{fig:cnn-learning}`
and `\input{Tables/comparison}` to include them.

Check table values and image run names against the source results in the repository's
`results/a1/` directory. The outline does not insert tables or figures automatically.
The obsolete `Original/` backups and `Evidence/` copies have been removed.

Use Fashion-MNIST for the main comparison. Identify MNIST as development data
and keep CIFAR-10 extension results separate. Report only completed experiments.
Do not infer reliable differences across runs from a single seed.

## Writing guidance

Keep methods in the methodology section and measured findings in the results section.
Explain shared settings once. Use consistent model descriptions and comparison axes.
Give every figure a purpose and a caption that states its dataset and selection rule.
Separate measured observations, possible explanations, and proposed experiments.
Remove unused optional sections and resolve all prompts before submission.
