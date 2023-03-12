import numpy as np
import cv2
from PIL import Image
from skimage import io
import re

class Normalize:

    def __init__(self, filepath):
        """
        Normalize an image
        """
        self.filepath = filepath

    def normalize_image(self):
        """Open an image file and convert it to a tensor."""
        # Open the image file
        img = io.imread(self.filepath)
        # Convert the image to a numpy array
        img_array = np.array(img)
        # Normalize the image tensor
        img_tensor = img_array / 255.0
        return img_tensor

    def cv2_based_normalize(self):
        # Load the image
        img = cv2.imread(self.filepath)
        # Normalize the image
        norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Display the original image
        return norm_image