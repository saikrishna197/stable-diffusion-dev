import torch
import torch.nn.functional as F

def generate_diffusion_images(image, sigma, n_iterations):
    # Convert image to PyTorch tensor
    image = torch.from_numpy(image)

    # Initialize list to store diffusion images
    diffusion_images = []

    # Loop through number of iterations
    for i in range(n_iterations):
        # Apply Gaussian filter
        filtered_image = F.gaussian_blur(image, sigma)
        # Append filtered image to list
        diffusion_images.append(filtered_image)
        # Set image to filtered image for next iteration
        image = filtered_image

    return diffusion_images

# Example usage
image = plt.imread("example.jpg")
sigma = 1
n_iterations = 5
diffusion_images = generate_diffusion_images(image, sigma, n_iterations)
