import numpy as np
import cv2
from PIL import Image
def augment_images(image_list):
  augmented_images = [augment_image(image) for image in image_list]
  pre_aug_sample = image_list[:5]
  aug_sample =augmented_images[:5]

  for i,(pre_aug,post_aug) in enumerate(zip(pre_aug_sample,aug_sample)):
    pre_aug.save(f'sample/pre_augmentation{i}.png')
    post_aug.save(f'sample/post_augmentation{i}.png')

  return augmented_images
    
def augment_image(image):
  image = np.array(image)
  #fft bandpass
  image = fourier_transform_bandpass(image,10,100)

  #sobel filter
  image = sobel_filter(image)

  #opening or closing
  image = image_closing(image)

  image = grayscale_to_rgb(image)

  # Convert NumPy array to PIL Image
  image = Image.fromarray(image.astype(np.uint8))

  return image



def fourier_transform_bandpass(image, low_radius, high_radius):
  # Convert the image to grayscale
  if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    gray = image
  
  # Perform Fourier Transform
  f = np.fft.fft2(gray)
  fshift = np.fft.fftshift(f)  # Shift zero frequency component to the center
  
  # Create a mask for the bandpass filter
  rows, cols = gray.shape
  crow, ccol = rows // 2, cols // 2  # Center of the image
  mask = np.zeros((rows, cols))
  
  # Apply the bandpass filter mask
  mask[crow - high_radius:crow + high_radius, ccol - high_radius:ccol + high_radius] = 1
  mask[crow - low_radius:crow + low_radius, ccol - low_radius:ccol + low_radius] = 0
  
  # Apply the mask to the frequency domain
  fshift = fshift * mask
  
  # Inverse Fourier Transform to get the filtered image
  f_ishift = np.fft.ifftshift(fshift)
  img_back = np.fft.ifft2(f_ishift)
  img_back = np.abs(img_back)  # Get the magnitude of the result
  
  return np.uint8(img_back)

def sobel_filter(image):
  # Convert to grayscale if necessary
  if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    gray = image
  # Apply Sobel operator for edge detection (both x and y gradients)
  sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
  sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
  
  # Calculate the gradient magnitude
  sobel_edges = cv2.magnitude(sobel_x, sobel_y)
  
  # Convert the result to 8-bit image
  sobel_edges = np.uint8(np.absolute(sobel_edges))
  
  return sobel_edges


def image_opening(image, kernel_size=(5, 5)):
  # Convert to grayscale if necessary
  if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    gray = image
  
  # Define the structuring element (kernel)
  kernel = np.ones(kernel_size, np.uint8)
  
  # Perform morphological opening (erosion followed by dilation)
  opened_image = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
  
  return opened_image

def image_closing(image, kernel_size=(5, 5)):
  # Convert to grayscale if necessary
  if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    gray = image
  
  # Define the structuring element (kernel)
  kernel = np.ones(kernel_size, np.uint8)
  
  # Perform morphological closing (dilation followed by erosion)
  closed_image = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
  
  return closed_image

def grayscale_to_rgb(image):
  # Stack the grayscale image across 3 channels (copying the grayscale into R, G, and B)
  rgb_image = np.stack([image] * 3, axis=-1)
  return rgb_image