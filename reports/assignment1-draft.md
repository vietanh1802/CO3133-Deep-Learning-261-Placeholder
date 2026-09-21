# Assignment 1 Draft

## Problem and data

The task is ten-class image classification on Fashion-MNIST. Each input is a 28-by-28
grayscale image and the output is one clothing-category label. MNIST is reserved for
development; all results below use Fashion-MNIST.

Fashion-MNIST provides 60,000 official training images and 10,000 official test images.
Every class has 6,000 training examples, so the imbalance ratio is 1.00. Using seed 42,
we reserve 6,000 examples from the official training set for validation and train on the
remaining 54,000. The test set remains untouched until checkpoint selection is complete.

Training inputs are converted to float tensors, randomly transformed by at most 10 degrees
and 10% translation, and normalized with mean 0.5 and standard deviation 0.5. Validation
and test inputs use the same conversion and normalization without random augmentation.

Generated EDA evidence:

- [Class distribution](../results/a1/eda/fashion_mnist/class_distribution.png)
- [Representative samples](../results/a1/eda/fashion_mnist/representative_samples.png)

## Methodology

The shared pipeline is: dataset → preprocessing → DataLoader → encoder → linear
classification head → cross-entropy loss → Adam → validation checkpoint selection → test
evaluation.

The Linear baseline flattens each image to 784 values and maps those values directly to ten
logits. The MLP also consumes flattened pixels, but uses hidden dimensions 256 and 128,
ReLU activations, and dropout 0.2 before the ten-class head. Neither applies softmax before
`CrossEntropyLoss`.

Both runs use learning rate 0.001, batch size 128, five epochs, seed 42, and the same
train/validation/test split. Training ran on Apple MPS. The selected checkpoint minimizes
validation loss. Test inference timings below were measured on CPU.

## Preliminary results

| Model | Runs | Accuracy | Macro-F1 | Parameters | Training time | Inference time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | 1 | 0.7612 | 0.7591 | 7,850 | 50.5 s | 0.005 s |
| MLP | 1 | 0.8198 | 0.8154 | 235,146 | 56.7 s | 0.019 s |

The MLP improves accuracy by 5.86 percentage points and macro-F1 by 0.0563. This supports
the limited draft conclusion that nonlinear combinations of flattened pixels outperform a
single linear decision boundary. The improvement costs approximately thirty times more
parameters and higher inference time.

Run evidence:

- [Linear learning curves](../results/a1/linear_seed42/learning_curves.png)
- [Linear confusion matrix](../results/a1/linear_seed42/test_confusion_matrix.png)
- [MLP learning curves](../results/a1/mlp_seed42/learning_curves.png)
- [MLP confusion matrix](../results/a1/mlp_seed42/test_confusion_matrix.png)
- [Linear confident errors](../results/a1/linear_seed42/test_confident_errors.png)
- [Linear uncertain errors](../results/a1/linear_seed42/test_uncertain_errors.png)
- [MLP confident errors](../results/a1/mlp_seed42/test_confident_errors.png)
- [MLP uncertain errors](../results/a1/mlp_seed42/test_uncertain_errors.png)
- [Machine-readable comparison](../results/a1/comparison_test.csv)

## Error analysis

The dominant failure mode for both preliminary models is confusion among visually similar
upper-body garments. Shirt is the weakest class: its test accuracy is 35.3% for the linear
model and 41.8% for the MLP. The linear model most often maps Shirt to Coat (190 cases) or
Pullover (185), while the MLP most often maps Shirt to T-shirt/top (247) or Pullover (147).
Pullover-to-Coat and Coat-to-Pullover are also common in both models. This pattern is
consistent with both encoders discarding explicit spatial locality by flattening the image.

The saved qualitative figures separate highly confident mistakes from low-confidence
mistakes and include representative correct predictions. Several confident errors exceed
99% predicted probability, so confidence alone is not a reliable indicator that either
current model is correct. These observations describe the present single-seed checkpoints;
they will be revisited after the CNN, GRU, and Transformer runs.

## Limitations and remaining work

These are preliminary five-epoch, single-seed results. They do not yet support a complete
architecture comparison. The final submission must add CNN, GRU, and Transformer results,
repeat runs where feasible, and discuss how spatial and sequential representations change
the observed trade-offs.
