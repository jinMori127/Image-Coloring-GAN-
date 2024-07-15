import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        )
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.final_conv = nn.Conv2d(64,3, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        ###
        #conv6 = self.conv6(conv5)
        ###


        
        # Decoder
        up4 = self.upconv4(conv5)
        #up4 = self.upconv4(dec5)
        dec4 = torch.cat((conv4, up4), dim=1)
        dec4 = self.dec4(dec4)
        
        up3 = self.upconv3(dec4)
        dec3 = torch.cat((conv3, up3), dim=1)
        dec3 = self.dec3(dec3)
        
        up2 = self.upconv2(dec3)
        dec2 = torch.cat((conv2, up2), dim=1)
        dec2 = self.dec2(dec2)
        
        up1 = self.upconv1(dec2)
        dec1 = torch.cat((conv1, up1), dim=1)
        dec1 = self.dec1(dec1)
        
        out = self.final_conv(dec1)
        return out


class Discriminator(nn.Module):
    #Discriminator with enhancements for better performance
    def __init__(self, large=False):
        super(Discriminator, self).__init__()

        self.conv1 = spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.2)  # Adjusted slope

        self.conv2 = spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)  # Adjusted slope
        self.dropout2 = nn.Dropout(0.3)  # Introduce dropout

        self.conv3 = spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)  # Adjusted slope

        self.conv4 = spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.2,inplace=True)  # Adjusted slope

        self.conv5 = spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.LeakyReLU(0.2,inplace=True)  # Adjusted slope

        self.conv6 = spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=0, bias=False))
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.LeakyReLU(0.2,inplace=True)  # Adjusted slope

        self.conv7 = spectral_norm(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False))


    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.dropout2(h)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu3(h)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu4(h)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu5(h)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.relu6(h)

        h = self.conv7(h)
        h = torch.sigmoid(h)  # Using torch.sigmoid for clarity

        return h.flatten(start_dim=1)


