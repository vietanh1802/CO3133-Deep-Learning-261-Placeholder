#!/bin/bash

# FASHION-MNIST
uv run python -m scripts.training.train_image_classification --data fashion_mnist --config configs/image_classification/fashion_mnist/linear.yaml
uv run python -m scripts.training.train_image_classification --data fashion_mnist --config configs/image_classification/fashion_mnist/mlp.yaml
uv run python -m scripts.training.train_image_classification --data fashion_mnist --config configs/image_classification/fashion_mnist/cnn.yaml
uv run python -m scripts.training.train_image_classification --data fashion_mnist --config configs/image_classification/fashion_mnist/gru.yaml
uv run python -m scripts.training.train_image_classification --data fashion_mnist --config configs/image_classification/fashion_mnist/transformer.yaml

# MNIST
uv run python -m scripts.training.train_image_classification --data mnist --config configs/image_classification/mnist/linear.yaml
uv run python -m scripts.training.train_image_classification --data mnist --config configs/image_classification/mnist/mlp.yaml
uv run python -m scripts.training.train_image_classification --data mnist --config configs/image_classification/mnist/cnn.yaml
uv run python -m scripts.training.train_image_classification --data mnist --config configs/image_classification/mnist/gru.yaml
uv run python -m scripts.training.train_image_classification --data mnist --config configs/image_classification/mnist/transformer.yaml

# CIFAR10
uv run python -m scripts.training.train_image_classification --data cifar10 --config configs/image_classification/cifar10/linear.yaml
uv run python -m scripts.training.train_image_classification --data cifar10 --config configs/image_classification/cifar10/mlp.yaml
uv run python -m scripts.training.train_image_classification --data cifar10 --config configs/image_classification/cifar10/cnn.yaml
uv run python -m scripts.training.train_image_classification --data cifar10 --config configs/image_classification/cifar10/gru.yaml
uv run python -m scripts.training.train_image_classification --data cifar10 --config configs/image_classification/cifar10/transformer.yaml
