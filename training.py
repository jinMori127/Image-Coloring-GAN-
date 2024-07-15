
from Networks import Generator,Discriminator
from data_reader import get_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2


#create the model and load them 
reset  = False
Genrator_path = 'Genrator.pth'
Disciminator_path = 'Disciminator.pth'
discriminator_loss_array_path = 'discriminator_loss_array.npy'
discriminator_epoch_loss_array_path = 'discriminator_epoch_loss_array.npy'
generator_epoch_loss_array_path ='generator_epoch_loss_array.npy'
generator_loss_array_path ='generator_loss_array.npy'
test_loss_array_path = 'test_loss_array.npy'
test_output_path = 'results'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
try: 
    current_Genrator = Generator()
    current_Discriminator = Discriminator()
    current_Genrator.load_state_dict(torch.load(Genrator_path,map_location=device))
    current_Discriminator.load_state_dict(torch.load(Disciminator_path,map_location=device))
    print('models loaded')
except:
    print('problem loading the models')
    current_Genrator = Generator()
    current_Discriminator = Discriminator()

try:
    generator_loss_array = np.load(generator_loss_array_path)
    discriminator_loss_array = np.load(discriminator_loss_array_path)
    test_loss_array = np.load(test_loss_array_path)
    generator_epoch_loss_array = np.load(generator_epoch_loss_array_path)
    discriminator_epoch_loss_array = np.load(discriminator_epoch_loss_array_path)
    print('array_loaded')
except:
    discriminator_epoch_loss_array = np.array([])
    generator_epoch_loss_array = np.array([])
    generator_loss_array = np.array([])
    discriminator_loss_array = np.array([])
    test_loss_array = np.array([])

    print('problem with loading the array')

if reset:
    print('reseting models and arries')
    current_Genrator = Generator()
    current_Discriminator = Discriminator()
    discriminator_epoch_loss_array = np.array([])
    generator_epoch_loss_array = np.array([])
    generator_loss_array = np.array([])
    discriminator_loss_array = np.array([])
    test_loss_array = np.array([])


current_Genrator = current_Genrator.to(device)
current_Discriminator = current_Discriminator.to(device)

train_data, test_data = get_data()
train_loader = DataLoader(train_data, batch_size=32, shuffle=True,pin_memory=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True,pin_memory=True)


num_epochs = 115

# Binary Cross Entropy Loss for the discriminator
bce_loss = nn.BCELoss()

# Mean Squared Error Loss for the generator
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

def discriminator_loss(real_output, fake_output):
    real_labels = torch.ones_like(real_output)
    fake_labels = torch.zeros_like(fake_output)
    
    # MSE loss for real and fake images
    real_loss = bce_loss(real_output, real_labels)
    fake_loss = bce_loss(fake_output, fake_labels)
    
    # Average of real and fake losses
    total_loss = (real_loss + fake_loss)
    return total_loss

# Generator loss as per the specified combination
def generator_loss(discriminator_fake_output, fake_images, real_images, lambda_l1=50):
    # L1 loss between the generated images and the real images
    l1 = l1_loss(fake_images, real_images) * lambda_l1
    
    # Treat generated images as real (label them as 1) for the discriminator's judgment
    true_labels = torch.ones_like(discriminator_fake_output)
    # Discriminator loss for the generated images
    gen_loss_discriminator = bce_loss(discriminator_fake_output, true_labels)
    
    # Combine the L1 loss and discriminator's judgment
    total_loss = l1 + gen_loss_discriminator
    return total_loss


generator_optimizer = optim.Adam(current_Genrator.parameters(), lr=0.0002)
discriminator_optimizer = optim.Adam(current_Discriminator.parameters(), lr=0.0002)


def train_step(gen_Y, dis_YUV, generator, discriminator, gen_optimizer, disc_optimizer,i):
    # Zero the parameter gradients
    gen_optimizer.zero_grad()
    disc_optimizer.zero_grad()
    
    # Generate an image -> G(x)
    generated_images = generator(gen_Y)
    #generated_images = torch.cat([gen_Y, generated_images], dim=1)
    real_generated_image = generated_images#create_images_from_YUV(generated_images)
    
    # Optionally save the generated image
    #######################
    if i%100==0:
        save_image(real_generated_image, 'test_RGB.jpg')
        save_image(real_generated_image[:,0,:,:].unsqueeze(1), 'test_R.jpg')
        save_image(real_generated_image[:,1,:,:].unsqueeze(1), 'test_G.jpg')
        save_image(real_generated_image[:,2,:,:].unsqueeze(1), 'test_B.jpg')

        save_image(dis_YUV, 'real_RGB.jpg')
        save_image(dis_YUV[:,0,:,:].unsqueeze(1), 'real_E.jpg')
        save_image(dis_YUV[:,1,:,:].unsqueeze(1), 'real_G.jpg')
        save_image(dis_YUV[:,2,:,:].unsqueeze(1), 'real_B.jpg')

        
    ########################
    # Discriminator outputs for real and generated images
    real_output = discriminator(dis_YUV)
    generated_output = discriminator(generated_images.detach())
    
    # Calculate losses
    disc_loss = discriminator_loss(real_output, generated_output)
    # Ensure to pass the correct parameters according to generator_loss's signature
    gen_loss = generator_loss(generated_output, generated_images, dis_YUV)


    # Perform backpropagation
    gen_loss.backward(retain_graph=True) 
    gen_optimizer.step()
    
    # Zero gradients before backward pass for discriminator to avoid accumulation
    disc_optimizer.zero_grad()
    disc_loss.backward()
    disc_optimizer.step()
    
    # Print losses
    print(f'Generator loss: {gen_loss.item()}')
    print(f'Discriminator loss: {disc_loss.item()}')
    
    return gen_loss.item(), disc_loss.item()
    

for epoch in range(num_epochs):
    if epoch %2 == 0:
        print('testing')
        current_Genrator.eval()
        current_Discriminator.eval()
        current_loss = 0 
        batches_number = 0 
        with torch.no_grad():
            for i,batch in enumerate(test_loader):
                batches_number +=1
                images_rbg,images_gray,real_size = batch
                images_gray = images_gray.to(device)
                real_images = images_rbg.cpu()
                save_image(real_images,os.path.join(test_output_path,f'{i}_ground_truth.jpg'))
                save_image(images_gray,os.path.join(test_output_path,f'{i}_gray.jpg'))

                results = current_Genrator(images_gray)
                #results = torch.cat([gen_Y,results],dim=1)
                real_results = results.cpu()
                save_image(real_results,os.path.join(test_output_path,f'{i}_results.jpg'))
                current_mse_loss = l1_loss(real_results,real_images)
                current_loss+=current_mse_loss.cpu().item()
            test_loss_array = np.append(test_loss_array,current_loss/batches_number)
            
        current_Genrator.train()
        current_Discriminator.train()  
        
    print(f"epoch: {epoch}")
    current_Discriminator_loss = 0
    current_Genrator_loss = 0
    batch_count = 0 
    for i,batch in enumerate(train_loader):
        batch_count +=1

        images_rbg,images_gray,real_size = batch
        images_gray = images_gray.to(device)
        dis_YUV = images_rbg.to(device)

        gen_loss,disc_loss = train_step(images_gray,dis_YUV,current_Genrator,current_Discriminator,generator_optimizer,discriminator_optimizer,i)
        generator_loss_array = np.append(generator_loss_array,gen_loss)
        discriminator_loss_array = np.append(discriminator_loss_array,disc_loss)
        current_Discriminator_loss+=disc_loss
        current_Genrator_loss +=gen_loss
    current_Genrator_loss = current_Genrator_loss/batch_count
    current_Discriminator_loss = current_Discriminator_loss/batch_count
    generator_epoch_loss_array = np.append(generator_epoch_loss_array,current_Genrator_loss)

    discriminator_epoch_loss_array = np.append(discriminator_epoch_loss_array,current_Discriminator_loss)

    np.save(generator_epoch_loss_array_path,generator_epoch_loss_array)
    np.save(discriminator_epoch_loss_array_path,discriminator_epoch_loss_array)
    np.save(generator_loss_array_path,generator_loss_array)
    np.save(discriminator_loss_array_path,discriminator_loss_array)
    np.save(test_loss_array_path,test_loss_array)
    torch.save(current_Genrator.state_dict(),Genrator_path)
    torch.save(current_Discriminator.state_dict(),Disciminator_path) 
    # Plotting
    plt.figure(figsize=(15, 25))
    # First subplot
    plt.subplot(5, 1, 1)  # 3 rows, 1 column, 1st subplot
    plt.plot(range(1,len(generator_loss_array)+1),generator_loss_array, 'r-')
    plt.title('generator_loss')

    # Second subplot
    plt.subplot(5, 1, 2)  # 3 rows, 1 column, 2nd subplot
    plt.plot(range(1,len(discriminator_loss_array)+1),discriminator_loss_array, 'r-')
    plt.title('discriminator_loss')

    # Third subplot
    plt.subplot(5, 1, 3)  # 3 rows, 1 column, 3rd subplot
    plt.plot(range(1,len(test_loss_array)+1),test_loss_array, 'r-')
    plt.title('test_loss')

    # forth subplot
    plt.subplot(5, 1, 4)  # 3 rows, 1 column, 3rd subplot
    plt.plot(range(1,len(generator_epoch_loss_array)+1),generator_epoch_loss_array, 'r-')
    plt.title('generator_epoch_loss')

    # fifth subplot
    plt.subplot(5, 1, 5)  # 3 rows, 1 column, 3rd subplot
    plt.plot(range(1,len(discriminator_epoch_loss_array)+1),discriminator_epoch_loss_array, 'r-')
    plt.title('discriminator_epoch_loss')



    # Adjust the layout so there's no overlap
    plt.tight_layout()

    # Show plot
    plt.show()  
    

