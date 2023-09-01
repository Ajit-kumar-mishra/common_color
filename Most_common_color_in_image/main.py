import cv2
import numpy as np
from sklearn.cluster import KMeans

def find_common_colors(image_path, num_colors=5):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Use K-Means clustering to find the most common colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    
    # Get the RGB values of the cluster centers
    common_colors = kmeans.cluster_centers_
    
    return common_colors.astype(int)

if __name__ == "__main__":
    image_path = "god.jpg"  # Replace with the path to your image
    num_colors = 5  # You can adjust the number of common colors you want to find
    
    common_colors = find_common_colors(image_path, num_colors)
    
    print("Most common colors:")
    for color in common_colors:
        print(f"RGB: {color}")
