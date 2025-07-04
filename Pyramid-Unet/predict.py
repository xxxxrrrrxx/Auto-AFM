from net import *
from data import *
from torchvision.utils import save_image
from PIL import Image
import cv2
import subprocess

# Load the UNet model
net = UNet().cuda()
_input = r'C:\Users\17105\Desktop\data-transmit\1.tif.tif'  # Input image path
weights = 'params/Pyramid.pth'  # Pre-trained weights
net.load_state_dict(torch.load(weights))

# Process input image using custom transformation and resize function
img = keep_image_size_open(_input)
img_data = transform(img).cuda()
img_data = torch.unsqueeze(img_data, dim=0)

# Pass image through the network
out = net(img_data)

# Save the network output
save_image(out, 'result/result.jpg')

# Open and manipulate the output image
image = Image.open('result/result.jpg')

# Crop and resize the image as needed
cropped_image = image.crop((0, 0, 256, 161))
resized_image = cropped_image.resize((1936, 1216))

# Save the resized image
resized_image.save('result/output_image.jpg')

# Define paths for the mask and output images
mask_path = 'result/output_image.jpg'
output_path = 'result/nucleus_boundary/path_to_original_image.jpg'


# Function to draw contours on the original image using mask
def draw_contours(image_path, mask_path, output_path):
    # Load original and mask images
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    # Convert mask to grayscale
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary mask
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over each contour and draw it if area is greater than 600
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 600:
            cv2.drawContours(img, [contour], -1, (255, 0, 0), 2)

    # Save the image with contours drawn
    cv2.imwrite(output_path, img)


# Call the function to draw contours
draw_contours(_input, mask_path, output_path)

# Define image and threshold for binary conversion
image_path = "result/output_image.jpg"
output_binary_path = "result/binary_image.jpg"
threshold_value = 55


# Function to convert image to binary and save
def convert_to_binary_and_save(image_path, threshold, output_path):
    # Load the image and convert to grayscale
    image = cv2.imread(image_path, 0)

    # Apply thresholding to convert to binary image
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Save the binary image
    cv2.imwrite(output_path, binary_image)


# Call the function to convert and save as binary image
convert_to_binary_and_save(image_path, threshold_value, output_binary_path)

# Run the external Python script for template matching
subprocess.run(['python', 'template_matching.py'])


