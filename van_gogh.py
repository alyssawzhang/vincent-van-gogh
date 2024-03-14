# Import various libraries
from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans
from scipy.spatial import distance
import pandas as pd
import requests
import numpy as np
import colorgram


# Load the input CSV file into a DataFrame
csv_file = r'van_gogh_input.csv'  # Can replace with applicable file path
data = pd.read_csv(csv_file, encoding='ISO-8859-1')

# Create empty lists to store the results
dominant_color = []
saturation = []
brightness = []
yellowness = []
color_diversity = []

# Function to calculate color variables (aka. features or metrics) for a single image
def process_image(row):
    image_url = row['Image Address']
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        total_pixels = image.width * image.height

        # Calculate RGB Dominant Color with k-means clustering and RGB color space
        label_to_color = {0: 'R', 1: 'G', 2: 'B'}
        rgb_image = np.array(image)
        pixels = rgb_image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3, n_init=10)
        kmeans.fit(pixels)

        # Calculate Saturation with average and HSV color space
        hsv_image = image.convert('HSV')
        saturation_values = hsv_image.split()[1].getdata()
        total_saturation = sum(saturation_values)
        average_saturation = total_saturation / total_pixels

        # Calculate Brightness with average and HSV color space
        brightness_values = hsv_image.split()[2].getdata()
        total_brightness = sum(brightness_values)
        average_brightness = total_brightness / total_pixels

        # Calculate Yellowness with average and CIELAB color space
        lab_image = image.convert('LAB')
        b_values = [lab[2] for lab in lab_image.getdata()]
        total_yellowness = sum(b_values)
        average_yellowness = total_yellowness / total_pixels

        # Calculate Color Diversity with average Euclidean distance between colors in colorgram palette
        colorgram_image = colorgram.extract(image, 10)  # Extract palette of 10 colors

        color_distances = []
        for i, color1 in enumerate(colorgram_image):
            for j, color2 in enumerate(colorgram_image):
                if i != j:
                    dist = distance.euclidean(color1.rgb, color2.rgb)
                    color_distances.append(dist)

        average_color_diversity = sum(color_distances) / len(color_distances) # Average "Color Diversity score"

        # Store results by appending to lists
        dominant_color.append(label_to_color[kmeans.labels_[0]])
        saturation.append(average_saturation)
        brightness.append(average_brightness)
        yellowness.append(average_yellowness)
        color_diversity.append(average_color_diversity)

    except Exception as e:
        dominant_color.append(None)
        saturation.append(None)
        brightness.append(None)
        yellowness.append(None)
        color_diversity.append(None)

# Process images one by one and append results to the lists
for index, row in data.iterrows():
    process_image(row)

# Add new columns to the DataFrame
data['Dominant Color'] = dominant_color
data['Saturation'] = saturation
data['Brightness'] = brightness
data['Yellowness'] = yellowness
data['Color Diversity'] = color_diversity

# Save the results to a new CSV file
output_file = 'van_gogh_output.csv'  # Can replace with desired output file path
data.to_csv(output_file, index=False)
