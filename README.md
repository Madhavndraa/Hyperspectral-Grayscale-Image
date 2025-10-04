# Hyperspectral-Grayscale-Image
A PyTorch U-Net GAN for spectral reconstruction. This tool converts standard 3-channel RGB images into 31-channel hyperspectral datacubes. Visualize individual spectral bands as "Hyperspectral Grayscale Images" to analyze material properties and features invisible in the original photo.

**📖 About The Project**

This project addresses the challenge of spectral reconstruction—estimating a rich, multi-band hyperspectral datacube from a simple RGB image. While a standard camera captures light in only three broad bands (Red, Green, and Blue), a hyperspectral sensor captures data across many narrow spectral bands, revealing detailed information about the material properties of objects in the scene.

Our model learns this mapping from RGB to HSI. The term "Hyperspectral Grayscale Image" refers to the visualization of a single channel from the generated hyperspectral cube. Each of these grayscale images represents the scene's appearance at a specific, narrow wavelength, allowing you to see features that are invisible in the original color image.

**Key Features**

1. RGB to Hyperspectral Conversion: Translates a standard (W, H, 3) image into a (W, H, 31) hyperspectral datacube.

2. U-Net Generator: Utilizes a U-Net architecture for the generator, which excels at preserving spatial details during the image-to-image translation task.

3. GAN Framework: Employs a Generative Adversarial Network to train the generator, pushing it to create realistic and plausible spectral data.

4. Individual Band Visualization: Allows for the extraction and saving of any of the 31 generated spectral bands as a grayscale image for analysis.

🛠️ How It Works
The project is built around a Conditional GAN (cGAN) architecture, specifically a Pix2Pix-like model:

1. Generator: A U-Net model takes a 3-channel RGB image as input. Through its encoder-decoder structure with skip connections, it generates a corresponding 31-channel hyperspectral image.

2. Discriminator: A PatchGAN discriminator receives both the input RGB image and either a real HSI (during training) or a fake, generated HSI. It learns to distinguish between the real and fake pairs, forcing the    Generator to produce higher-quality results.

3. Loss Functions: The training is guided by a combination of two losses:

   a. Adversarial Loss: Pushes the generator to create outputs that can fool the discriminator.

   b. L1 Loss (MAE): Ensures the generated HSI is structurally similar to the ground truth HSI, acting as a direct regression objective.


