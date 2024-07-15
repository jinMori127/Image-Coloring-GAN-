# Image-Coloring-GAN-
In this project is to input gray scale images and output RGB images.

Data set used: 102flowers.tgz

Network structure:
We used GAN 
*Note: we will mark convolutional layer by C.
Generator: used U-net architecture 
![image](https://github.com/user-attachments/assets/ed6b7fc7-28b0-4b36-ad76-72eb156eae2a)

After each convolution we did batch norm and relu activation.

Discriminator:
The discriminator is a lot easier so we didn’t draw any drawings for it.
it has a sequence of layers like this:
𝑆𝑃(𝐶64) − 𝑆𝑃(𝐶128) − 𝑆𝑃(𝐶256) − 𝑆𝑃(𝐶512) − 𝑆𝑃(𝐶512) − 𝑆𝑃(𝐶1) 
SP = spectral_norm() 
Spectral normalization stabilizes the training of discriminators in Generative 
Adversarial Networks (GANs) by rescaling the weight tensor.
So for a weight matrix W the spectral normalization is achieved by:
Sure, here it is for you to copy:

---
**Normalized Weight Matrix (Ws)**:

\[ W_{sp} = \frac{W}{\sigma(W)} \]

Where \(\sigma(W)\) is the largest singular value of the matrix \(W\). The singular values are the square roots of the eigenvalues of the matrix \(W^t \cdot W\).

---
We also used after each conv layer leaky Relu with slop of 0.2 and a BatchNorm2d.
generator_loss:
we used the nn.L1Loss() to calculate the loss between the real and the 
generated image and nn.BCELoss() to calculate how good we fool the 
discriminator.
Loss = l1*lamda + gen_loss_discriminator
In our case lamda=50 we tried a lot of values and we got the best on 50
what is lamda 50 for? we want to give a bigger value to a loss achieved by
 L1Loss (real_img : gen_img) so the gen will color the images better taking in 
consideration how good the gen_img fooled the discriminator so he will 
Always try to color the images in the same time he maintain colored images 
That could fool the discriminator.

discriminator_loss : used a nn.BCELoss()

define the loss by real_img_loss + fake_img_loss
real_img_loss = the loss of the discriminator in the real images.
fake_img_loss = the loss of the discriminator in the generated images.

For both of the dis and gen we used Optimizer : optim.Adam with lr: 0.0002
train Bach size:32 , test Bach size: 1, num_epochs: 116

**data processing:** 

we used cv2 and torchvision to manipulate the data , we read the data from the 
tgz file that we downloaded from the link using the tarfile library, while reading it 
we turn the data to gray scale and resize the image to 128𝑋128 using cv2, and 
turn the image to a tensor using torchvision (because the cv2 image is 𝐻𝑒𝑖𝑔ℎ ∗
𝑊𝑖𝑑𝑡ℎ ∗ 𝑐ℎ𝑎𝑛𝑒𝑙𝑠 but we want 𝑐ℎ𝑎𝑛𝑒𝑙𝑠 ∗ 𝐻𝑒𝑖𝑔ℎ𝑡 ∗ 𝑊𝑖𝑑𝑡ℎ), implementation in 
𝑟𝑒𝑎𝑑_𝑑𝑎𝑡𝑎, in the end this function return the RGB and Gray image also the 
original image size.
Also we implemented a class (𝑑𝑎𝑡𝑎) that inherits from Dataset to make the 
access easier and not complicated.
We implemented a function called get_data that reads the flower images and 
then split the images to training and test set. 

**Results:**

We check the results using l1_loss with the original image also all the test images 
are outputted to a folder called results. 
Our graphs:

![image](https://github.com/user-attachments/assets/e3e5774d-c9c1-43b5-980b-93122b020c46)

