# Adaptive Branch Weighting in Convolutional Neural Networks

An experimental computer-vision project exploring adaptive branch aggregation in convolutional neural networks. The work first evaluates a multi-branch CNN on CIFAR-10, then tests whether its learned weighting mechanism transfers to FashionMNIST through a controlled ablation study.

## Key results

- **CIFAR-10:** 93.36% best test accuracy after 40 epochs.
- **FashionMNIST:** all four aggregation strategies achieved more than 93.9% best test accuracy.
- **Best FashionMNIST result:** uniform branch averaging reached 94.45%, narrowly ahead of softmax-constrained weighting at 94.43% and learned weighting at 94.37%.
- The comparison suggests that the convolutional branches provide most of the representational capacity on FashionMNIST, while the weighting mechanism has a smaller secondary effect.

## Notebooks

- [`Part_A/cifar10_adaptive_cnn.ipynb`](Part_A/cifar10_adaptive_cnn.ipynb) develops and evaluates the adaptive multi-branch CNN on CIFAR-10.
- [`Part_B/fashionmnist_branch_weighting_ablation.ipynb`](Part_B/fashionmnist_branch_weighting_ablation.ipynb) compares learned, uniform, random-fixed and softmax-constrained branch weighting.

Both notebooks include saved training outputs and visualisations, so the reported experiments can be reviewed without rerunning the full training process.

## Methods

- PyTorch and Torchvision
- Multi-branch convolutional architecture
- Learned feature aggregation
- Data augmentation and label smoothing
- Learning-rate scheduling
- Controlled ablation study
- CIFAR-10 and FashionMNIST

## Running the notebooks

Install the dependencies:

```bash
pip install -r requirements.txt
```

The datasets are downloaded automatically by Torchvision when each notebook is run. A CUDA-capable GPU is recommended for training, but the saved outputs allow the project to be inspected without retraining.

## Author

Asli Akel
