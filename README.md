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

# 🛠️ How It Works

The project is built around a Conditional GAN (cGAN) architecture, specifically a Pix2Pix-like model:

1. Generator: A U-Net model takes a 3-channel RGB image as input. Through its encoder-decoder structure with skip connections, it generates a corresponding 31-channel hyperspectral image.

2. Discriminator: A PatchGAN discriminator receives both the input RGB image and either a real HSI (during training) or a fake, generated HSI. It learns to distinguish between the real and fake pairs, forcing the    Generator to produce higher-quality results.

3. Loss Functions: The training is guided by a combination of two losses:

   a. Adversarial Loss: Pushes the generator to create outputs that can fool the discriminator.

   b. L1 Loss (MAE): Ensures the generated HSI is structurally similar to the ground truth HSI, acting as a direct regression objective.

# ⚙️ Installation & Usage

Follow these steps to set up the environment and run the project. The process involves two main stages: training the model and then using it for inference.

Step 1: Initial Setup
Clone the repository:
Open your terminal or command prompt and clone the repository to your local machine.

> **Bash**
```
git clone https://github.com/Madhavndraa/Hyperspectral-Grayscale-Image.git
cd Hyperspectral-Grayscale-Image
```

**Install Dependencies:**
This project requires PyTorch, Torchvision, and Pillow. You can install them using pip.

> **Bash**
```
pip install torch torchvision numpy Pillow
```
Prepare Your Dataset:
Make sure you have your RGB and Hyperspectral image pairs organized in a Dataset directory as expected by the UGAN.py script.

Step 2: Train the Model
First, you need to train the Generative Adversarial Network. This step will create the model file (generator_model.pth) that is used to generate new images.

**▶️ Run the UGAN.py script:**

> **Bash**
```
python UGAN.py
```
This script will start the training process.

Once training is complete, a file named generator_model.pth will be saved in your project directory.

Step 3: Generate the Hyperspectral Image
After the model has been trained and generator_model.pth is created, you can now use it to convert any RGB image into a hyperspectral image.

Configure the Input:
Open the HSI.py file and change the Input_Image_Path variable to the path of the RGB image you want to convert.


> **Inside HSI.py**
```
Input_Image_Path = 'path/to/your/image.jpg' # <-- CHANGE THIS
```

Run the HSI.py script:
Execute the script from your terminal.

> **Bash**
```
python HSI.py
```

This script will load the trained model (generator_model.pth).

It will process your input image and save a single channel of the generated hyperspectral image (e.g., hsi_channel_15.png) in your project directory. You can change which channel is saved by modifying the channel_to_view variable in the script.
