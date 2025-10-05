import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader, TensorDataset ,Dataset
from torchvision import transforms
from PIL import Image
import os
import shutil



class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=31):
        super(UNetGenerator, self).__init__()
        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(512,1024)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(1024, 512)  # Input channels = 512 (from upconv) + 512 (from enc4)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(512, 256)  # Input channels = 256 (up) + 256 (enc3)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256, 128)  # Input channels = 128 (up) + 128 (enc2)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(128, 64)

        # Output
        self.final =nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_c,out_c,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec1_up = self.upconv1(bottleneck)
        dec1_cat = torch.cat((dec1_up, enc4), dim=1)
        dec1 = self.decoder1(dec1_cat)

        dec2_up = self.upconv2(dec1)
        dec2_cat = torch.cat((dec2_up, enc3), dim=1)
        dec2 = self.decoder2(dec2_cat)

        dec3_up = self.upconv3(dec2)
        dec3_cat = torch.cat((dec3_up, enc2), dim=1)
        dec3 = self.decoder3(dec3_cat)

        dec4_up = self.upconv4(dec3)
        dec4_cat = torch.cat((dec4_up, enc1), dim=1)
        dec4 = self.decoder4(dec4_cat)

        return self.final(dec4)
class Discriminator(nn.Module):
    def __init__(self, in_channels= 3+31):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1,padding=1)
        )
    def forward(self, x):
        return self.model(x)

def training(num_epochs=30, batch_size=16, num_samples=100):
    """
    Main training loop. Generates synthetic data and trains the GAN.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Generate synthetic data using NumPy
    print(f"Generating {num_samples} synthetic data samples...")
    rgb_data_np = np.random.rand(num_samples, 256, 256, 3).astype(np.float32)
    hsi_data_np = np.random.rand(num_samples, 256, 256, 31).astype(np.float32)

    # Transpose to PyTorch format (N, C, H, W)
    rgb_data_np = rgb_data_np.transpose((0, 3, 1, 2))
    hsi_data_np = hsi_data_np.transpose((0, 3, 1, 2))

    # 2. Convert to PyTorch Tensors and create DataLoader
    rgb_tensors = torch.from_numpy(rgb_data_np)
    hsi_tensors = torch.from_numpy(hsi_data_np)
    dataset = TensorDataset(rgb_tensors, hsi_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Synthetic data ready.")

    # 3. Initialize models, optimizers, and loss functions
    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)

    g_opt = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    lambda_l1 = 100

    print("--- Starting Training ---")
    for epoch in range(num_epochs):
        for i, (rgb_img, hsi_img) in enumerate(dataloader):
            rgb_img = rgb_img.to(device)
            hsi_img = hsi_img.to(device)

            # --- Train Discriminator ---
            d_opt.zero_grad()
            # Real images
            real_out = discriminator(torch.cat((rgb_img, hsi_img), dim=1))
            real_labels = torch.ones(real_out.size(), device=device) # Labels must match output shape
            d_loss_real = criterion_gan(real_out, real_labels)
            # Fake images
            fake_hsi = generator(rgb_img)
            fake_out = discriminator(torch.cat((rgb_img, fake_hsi.detach()), dim=1))
            fake_labels = torch.zeros(fake_out.size(), device=device) # Labels must match output shape
            d_loss_fake = criterion_gan(fake_out, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            d_opt.step()

            # --- Train Generator ---
            g_opt.zero_grad()
            fake_out_for_g = discriminator(torch.cat((rgb_img, fake_hsi), dim=1))
            g_loss_gan = criterion_gan(fake_out_for_g, real_labels) # Generator tries to fool discriminator
            g_loss_l1 = criterion_l1(fake_hsi, hsi_img)
            g_loss = g_loss_gan + lambda_l1 * g_loss_l1
            g_loss.backward()
            g_opt.step()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], "
                      f"D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}")

    print("--- Training Complete ---")

    # 4. Save the trained generator model
    model_save_path = "generator_model.pth"
    torch.save(generator.state_dict(), model_save_path)
    print(f"Generator model saved to {model_save_path}")


if __name__ == "__main__":
    # Start training with a small dataset for demonstration
    training(num_epochs=10, batch_size=4, num_samples=40)