import torch
from torchvision import transforms
from  PIL import Image
import os
import sys
from UGAN import UNetGenerator
import numpy as np

def generate_HSI(rgb_img_path,model_path):
    """
    Loads Generator model and uses it to generate HSI.
    Args:
         rgb_img_path (string): Path to the image to be loaded.
         model_path (string): Path to the model to be loaded.
    Returns:
        torch.tensor: Tensor representing Generated HSI.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} does not exist.")
        return None

    if not os.path.exists(rgb_img_path):
        print(f"Error: Image {rgb_img_path} does not exist.")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    generator = UNetGenerator(in_channels=3,out_channels=31).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device,weights_only=False))
    generator.eval() # Set model to Evaluation mode

    # Image Transformations
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    try:
        rgb_img = Image.open(rgb_img_path).convert('RGB')
        transformed_img = transform(rgb_img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error: loading and processing image: {e}")
        return None

    print(f"Generating HSI...")
    # to disable gradient calculations for inference
    with torch.no_grad(): generated_hsi = generator(transformed_img)

    print(f"HSI generated: {generated_hsi}")
    return generated_hsi.squeeze(0).cpu()

if __name__ == "__main__":
    Model_File_Path = "generator_model.pth"

    # Input Image
    Input_Image_Path = 'Dataset/test/PotatoEarlyBlight1.JPG'
    # Generate the HSI
    hsi_tensor = generate_HSI(Input_Image_Path, Model_File_Path)

    if hsi_tensor is not None:
        print("--- HSI Generation Successful! ---")
        print(f"Shape of the generated HSI tensor: {hsi_tensor.shape}")

    if hsi_tensor is not None:
        print("--- HSI Generation Successful! ---")
        print(f"Shape of the generated HSI tensor: {hsi_tensor.shape}")

        # --- VISUALIZATION CODE ---
        channel_to_view = 15
        print(f"Saving channel {channel_to_view} as a grayscale image...")

        single_channel = hsi_tensor[channel_to_view]
        numpy_image = single_channel.cpu().numpy()
        numpy_image = ((numpy_image - numpy_image.min()) / (numpy_image.max() - numpy_image.min()) * 255).astype(
            np.uint8)

        output_image = Image.fromarray(numpy_image, 'L')
        output_image.save(f"hsi_channel_{channel_to_view}.png")

        print(f"Successfully saved hsi_channel_{channel_to_view}.png")