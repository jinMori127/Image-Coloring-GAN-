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
𝑆𝑃(𝐶64) − 𝑆𝑃(𝐶128) − 𝑆𝑃(𝐶256) − 𝑆𝑃(𝐶512) − 𝑆𝑃(𝐶512) − 𝑆𝑃(𝐶1) 
 
SP = `spectral_norm()`

---
Spectral normalization stabilizes the training of discriminators in Generative 
Adversarial Networks (GANs) by rescaling the weight tensor.
So for a weight matrix W the spectral normalization is achieved by:

**Normalized Weight Matrix (Ws):**

```math
W_{sp} = \frac{W}{\sigma(W)}
```

Where ```math (\sigma(W))``` is the largest singular value of the matrix \(W\). The singular values are the square root of the eigenvalues of the matrix ```math(W^t \cdot W)```.

---
We also used Leaky ReLU with a slope of 0.2 and BatchNorm2d after each convolution layer.

## Loss Functions:

**Generator Loss:**
We used `nn.L1Loss()` to calculate the loss between the real and the generated image and `nn.BCELoss()` to evaluate how well we fool the discriminator.

Loss = l1*lamda + gen_loss_discriminator

In our case lamda=50 we tried a lot of values and we got the best on 50
what is lamda 50 for? we want to give a bigger value to a loss achieved by
 L1Loss (real_img : gen_img) so the gen will color the images better taking in 
consideration how good the gen_img fooled the discriminator so he will 
Always try to color the images in the same time he maintain colored images 
That could fool the discriminator.

**Discriminator Loss:**
We used `nn.BCELoss()`.

Define the loss by:

define the loss by real_img_loss + fake_img_loss
-real_img_loss = the loss of the discriminator in the real images.
-fake_img_loss = the loss of the discriminator in the generated images

For both the discriminator and the generator, we used:

- **Optimizer:** `optim.Adam` with `lr: 0.0002`
- **Train Batch size:** 32 
- **Test Batch size:** 1
- **Num epochs:** 116

## Data Processing:

We used `cv2` and `torchvision` to manipulate the data. We read the data from the tgz file downloaded using the `tarfile` library. While reading it, we converted the data to grayscale and resized the images to \(128 \times 128\) using `cv2`, and turned the images into tensors using `torchvision` (because the cv2 image is 𝐻𝑒𝑖𝑔ℎ ∗
𝑊𝑖𝑑𝑡ℎ ∗ 𝑐ℎ𝑎𝑛𝑒𝑙𝑠 but we want 𝑐ℎ𝑎𝑛𝑒𝑙𝑠 ∗ 𝐻𝑒𝑖𝑔ℎ𝑡 ∗ 𝑊𝑖𝑑𝑡ℎ), implementation in 
`𝑟𝑒𝑎𝑑_𝑑𝑎𝑡𝑎`, in the end this function return the RGB and Gray image also the 
original image size.

We also implemented a class (`data`) that inherits from `Dataset` to make data access easier and less complicated. The function `get_data` reads the flower images and splits them into training and test sets.

## Results:

We checked the results using L1Loss with the original image. All the test images are outputted to a folder called `results`.

Our graphs:  
![Results Graph](https://github.com/user-attachments/assets/e3e5774d-c9c1-43b5-980b-93122b020c46)

---
---
## Brief look at some good results:

![image](https://github.com/user-attachments/assets/3fab8954-2b53-4d88-94f0-32ea65cbcd40)

for full view at the result: https://drive.google.com/drive/folders/1PRRA18SLu9jdqYRAUb0YRSE3hzhFGMqn?usp=sharing

---
