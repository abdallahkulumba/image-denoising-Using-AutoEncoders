# Basic Autoencoder for Noise Reduction

This project implements a basic autoencoder using PyTorch to reduce noise in grayscale images. The autoencoder learns to map noisy images to their clean versions, effectively denoising them.

![Alt text](pic.png?raw=true "sample")

## Overview

An autoencoder is a type of neural network used for unsupervised learning. It compresses input data into a latent space and reconstructs it back to its original form. In this project, the autoencoder is used to denoise images by training it to reconstruct clean images from noisy inputs.

Key Features:

- **Dataset**: The MNIST dataset is used for demonstration, consisting of 28x28 grayscale images of digits.
- **Noise Addition**: Gaussian noise is added to the input images.
- **Visualization**: Visual comparison of noisy, clean, and denoised images after training.

## Dependencies

To run this project, you'll need the following libraries:

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib

Install the dependencies using pip:

```bash
pip install torch torchvision numpy matplotlib
```



## Model Architecture

The autoencoder consists of two main components:

* **Encoder** : Compresses the input image into a smaller latent representation.
* **Decoder** : Reconstructs the image from the latent representation.

### Encoder

* Convolutional layers reduce the spatial dimensions of the image while extracting features.
* Activation functions introduce non-linearity.

### Decoder

* Transposed convolutional layers expand the latent space back to the original image size.
* A sigmoid activation function outputs pixel values in the range [0, 1].

## Training

The training loop optimizes the Mean Squared Error (MSE) loss function, comparing the denoised image output with the original clean image. The model is trained using the Adam optimizer.

### Steps:

1. Add Gaussian noise to the training images.
2. Train the autoencoder to reconstruct clean images from the noisy input.
3. Visualize the denoising performance on test images.

## Future Work

Potential improvements to this project:

* Experiment with deeper architectures for better denoising performance.
* Test on other datasets (e.g., CIFAR-10, noisy real-world images).
* Add support for color image denoising.

## References

1. [PyTorch Documentation](https://pytorch.org/docs/)
2. [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
3. [Understanding Autoencoders](https://www.deeplearningbook.org/)

## License

This project is licensed under the MIT License.

## Happy Coding!
