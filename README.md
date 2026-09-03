# Custom CNN for Image Classification & Branch-Weighting Ablation (CIFAR-10, Fashion-MNIST)

This project explores how convolutional branches can be combined within a CNN. Part A develops the model on CIFAR-10. Part B then keeps the main architecture fixed and compares four branch-weighting strategies on FashionMNIST.

## Results

| Experiment | Weighting method | Best test accuracy |
|---|---|---:|
| CIFAR-10 | Learned FC | **93.36%** |
| FashionMNIST | Uniform | **94.45%** |
| FashionMNIST | Softmax FC | 94.43% |
| FashionMNIST | Learned FC | 94.37% |
| FashionMNIST | Random fixed | 93.94% |

The FashionMNIST results are close, with only 0.51 percentage points between the best and worst variants. In this experiment, the convolutional branches appear to provide most of the representational capacity, while the weighting method has a smaller effect.

## Notebooks

- [`Part_A/cifar10_adaptive_cnn.ipynb`](Part_A/cifar10_adaptive_cnn.ipynb) develops and evaluates the adaptive multi-branch CNN on CIFAR-10.
- [`Part_B/fashionmnist_branch_weighting_ablation.ipynb`](Part_B/fashionmnist_branch_weighting_ablation.ipynb) compares learned, uniform, random-fixed and softmax-constrained branch weighting.

Both notebooks contain the saved training outputs and visualisations. The experiments can therefore be reviewed without running the training again.

## Model structure

The model uses four multi-branch convolutional blocks. Channel depth increases from 64 to 512 while max-pooling reduces the spatial resolution:

```text
Input (3 × 32 × 32)
  → Multi-branch block (64 channels)  → MaxPool
  → Multi-branch block (128 channels) → MaxPool
  → Multi-branch block (256 channels) → MaxPool
  → Multi-branch block (512 channels)
  → Spatial average → Fully connected classifier → 10 classes
```

Each branch contains two convolutional layers with batch normalisation, ReLU and dropout. Part B changes only the way branch outputs are weighted, allowing the four variants to be compared under the same experimental conditions.

## Main methods

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

The datasets are downloaded automatically by Torchvision and are not stored in this repository. Run Part A before Part B if the full experimental sequence is being followed.

A CUDA-capable GPU is recommended. Part A trains for 40 epochs. Part B trains four variants for 15 epochs each and took approximately one hour on the GPU used for the saved run. Exact times may vary by hardware and software version.

Part B fixes the random seed at 42 and holds the optimiser, scheduler, data transforms, batch size and hardware conditions constant across the four variants.

## Author

Asli Akel
