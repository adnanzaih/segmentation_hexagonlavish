import numpy as np
from sklearn.cluster import KMeans
from skimage import io
from skimage.color import rgb2lab, lab2rgb


class KMeanClustering:

    def __init__(self, image):
        """
        K-Means Clustering is an image segmentation technique that groups similar pixels together based on their color or intensity values. 
        The basic idea behind this algorithm is to divide the image into k clusters, where each cluster represents a group of pixels with similar characteristics.
        """
        self.image = image

    def kmeanclustering(self):
        # Load the image
        image = self.image

        # Convert the image to the L*a*b* color space
        image_lab = rgb2lab(image[:,:,:3])

        # Reshape the image to a 2D array of pixels
        pixels = np.reshape(image_lab, (image.shape[0] * image.shape[1], 3))

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=3).fit(pixels)

        # Get the cluster labels for all pixels
        labels = kmeans.labels_

        # Create an empty image with the same shape as the original
        segmented_image = np.zeros((image.shape[0], image.shape[1], 3))

        # Fill in the segmented image with the cluster colors
        for i in range(image.shape[0] * image.shape[1]):
            segmented_image[i // image.shape[1], i % image.shape[1]] = kmeans.cluster_centers_[labels[i]]

        # Convert the segmented image back to RGB
        segmented_image = lab2rgb(segmented_image)

        return segmented_image
