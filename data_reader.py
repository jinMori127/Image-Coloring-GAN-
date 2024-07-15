import tarfile
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class data(Dataset):
    def __init__(self, file_path='102flowers.tgz'):
        #the groud truth is our data without processing any thing
        self.rgb_images,self.gray_images ,self.real_size  = read_data(file_path)


    def __getitem__(self, index):
        return self.rgb_images[index],self.gray_images[index],self.real_size[index]
    
    def __len__(self):
        return len(self.real_size)
    
def get_data(file_name='102flowers.tgz',train_percentage = 0.9):
    #this function gets the file name and the traiin percentage of the data that you want 
    #and return trainSet and testSet
    torch.random.manual_seed(42)
    dataset = data(file_name)
    total_size = len(dataset)
    train_size = int(train_percentage * total_size) 
    test_size = total_size - train_size
    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset,test_dataset
    



def read_data(File_path='102flowers.tgz'):
    #if we are reading the flower database
    images_list_rbg = []
    images_list_YUV = []
    real_images_sizes = []
    if os.path.basename(File_path) == '102flowers.tgz' or os.path.basename(File_path) == 'food-101.tar.gz':

        # Open the .tgz file
        with tarfile.open(File_path, 'r:gz') as tgz:
            # Iterate through the members
            for member in tgz.getmembers():
                # Filter out directories and potentially non-image files based on file extensions
                if member.isfile() and member.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    # Extract the member file as a file-like object
                    fileobj = tgz.extractfile(member)
                    if fileobj:
                        # Read the image into a byte array
                        image_bytes = np.asarray(bytearray(fileobj.read()), dtype=np.uint8)
                        
                        # Decode the image from the byte array
                        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                        '''plt.imshow(image)
                        plt.show()
                        image_numpy = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)
                        plt.imshow(image_numpy)
                        plt.show()'''
                        
                        # Append the image to the list if successfully loaded
                        if image is not None:
                            real_images_sizes.append([image.shape[0],image.shape[1]])
                            image_rgb = cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(128,128))
                            image_gray = cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(128,128))


                            #image_rgb = cv2.resize(image,(128,128))
                            transform = transforms.ToTensor()

                            # Convert the image to PyTorch tensor
                            image_tensor = transform(image_rgb)
                            images_list_rbg.append(image_tensor)

                            image_tensor = transform(image_gray)
                            images_list_YUV.append(image_tensor)

    return images_list_rbg,images_list_YUV,real_images_sizes

def return_image_to_orginal_size(image,real_size):
    transform = transforms.Resize(real_size)
    return transform(image)

