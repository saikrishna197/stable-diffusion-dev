import torch
import torch.nn.functional as F

def generate_diffusion_images(image, sigma, n_iterations):
    # Convert image to PyTorch tensor
    image = torch.from_numpy(image).float()

    # Initialize list to store diffusion images
    diffusion_images = []

    # Create Gaussian filter kernel
    kernel = torch.tensor(F.gaussian_kernel(sigma=sigma))

    # Loop through number of iterations
    for i in range(n_iterations):
        # Apply Gaussian filter
        filtered_image = F.conv2d(image.unsqueeze(0), kernel.unsqueeze(0))
        # Append filtered image to list
        diffusion_images.append(filtered_image.squeeze())
        # Set image to filtered image for next iteration
        image = filtered_image.squeeze()

    return diffusion_images
