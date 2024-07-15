---

# Image-Coloring-GAN

In this project, we input grayscale images and output RGB images.

**Data set used:** `102flowers.tgz`

## Network structure:

We used GAN.

*Note: We will mark convolutional layer by C.*

**Generator:** 
We used U-net architecture.  
![Generator Architecture](https://github.com/user-attachments/assets/ed6b7fc7-28b0-4b36-ad76-72eb156eae2a)

After each convolution, we applied batch normalization and ReLU activation.

**Discriminator:**
The discriminator is simpler, so we didn’t draw any diagrams for it.  
It has a sequence of layers as follows:  
\[ \text{SP}(C64) - \text{SP}(C128) - \text{SP}(C256) - \text{SP}(C512) - \text{SP}(C512) - \text{SP}(C1) \]  
SP = `spectral_norm()`

Spectral normalization stabilizes the training of discriminators in Generative Adversarial Networks (GANs) by rescaling the weight tensor.  
So, for a weight matrix \(W\), the spectral normalization is achieved by:

**Normalized Weight Matrix (Ws):**

\[ W_{sp} = \frac{W}{\sigma(W)} \]

Where \(\sigma(W)\) is the largest singular value of the matrix \(W\). The singular values are the square roots of the eigenvalues of the matrix \(W^t \cdot W\).

We also used Leaky ReLU with a slope of 0.2 and BatchNorm2d after each convolution layer.

## Loss Functions:

**Generator Loss:**
We used `nn.L1Loss()` to calculate the loss between the real and the generated image and `nn.BCELoss()` to evaluate how well we fool the discriminator.

\[ \text{Loss} = \text{L1Loss} \times \lambda + \text{gen\_loss\_discriminator} \]

In our case, \(\lambda = 50\). We experimented with various values and found the best results with \(\lambda = 50\). The purpose of setting \(\lambda\) to 50 is to give more weight to the loss achieved by L1Loss (real\_img : gen\_img), so the generator colors the images better while also considering how well the generated images fool the discriminator. This ensures the generator always tries to color the images while maintaining colored images that can fool the discriminator.

**Discriminator Loss:**
We used `nn.BCELoss()`.

Define the loss by:

\[ \text{discriminator\_loss} = \text{real\_img\_loss} + \text{fake\_img\_loss} \]

- real\_img\_loss: the loss of the discriminator on the real images.
- fake\_img\_loss: the loss of the discriminator on the generated images.

For both the discriminator and the generator, we used:

- **Optimizer:** `optim.Adam` with `lr: 0.0002`
- **Train Batch size:** 32 
- **Test Batch size:** 1
- **Num epochs:** 116

## Data Processing:

We used `cv2` and `torchvision` to manipulate the data. We read the data from the tgz file downloaded using the `tarfile` library. While reading it, we converted the data to grayscale and resized the images to \(128 \times 128\) using `cv2`, and turned the images into tensors using `torchvision` (since `cv2` images are in \( \text{Height} \times \text{Width} \times \text{Channels} \) format, but we need \( \text{Channels} \times \text{Height} \times \text{Width} \)).

The implementation is in the `read_data` function, which returns the RGB and Gray images along with the original image size.

We also implemented a class (`data`) that inherits from `Dataset` to make data access easier and less complicated. The function `get_data` reads the flower images and splits them into training and test sets.

## Results:

We checked the results using L1Loss with the original image. All the test images are outputted to a folder called `results`.

Our graphs:  
![Results Graph](https://github.com/user-attachments/assets/e3e5774d-c9c1-43b5-980b-93122b020c46)

---
