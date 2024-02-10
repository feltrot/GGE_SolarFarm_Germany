
# get Landsat data
import landsatxplore.api
import os
import requests
import zipfile
from io import BytesIO
import arcpy
import numpy as np
from sklearn.cluster import KMeans

# Function to download Landsat images
def download_landsat_images(api_key, output_path, latitude, longitude, start_date, end_date):
    api = landsatxplore.api.API(api_key)
    
    scenes = api.search(
        dataset='LANDSAT_8_C1',
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=20
    )
    
    # Download the first scene (you can modify this logic based on your requirements)
    scene_id = scenes[0]['entityId']
    api.download(scene_id, output_path)
    api.logout()

# Function to unzip downloaded Landsat images
def unzip_landsat_images(zip_path, output_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)

# Function to classify images using K-means clustering
def classify_image(image_path, num_classes=5):
    # Use arcpy to read raster data
    raster = arcpy.Raster(image_path)
    # Get the RGB bands (change band numbers if needed)
    bands = [raster.bandCount - i for i in range(3)]
    image_data = arcpy.RasterToNumPyArray(raster, arcpy.Point(raster.extent.XMin, raster.extent.YMin), raster.width, raster.height)

    # Reshape the data to 2D array (pixel values for each band)
    flat_image = np.dstack([image_data[:, :, i] for i in bands]).reshape((-1, len(bands)))

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_classes, random_state=42).fit(flat_image)
    labels = kmeans.labels_

    # Reshape labels to match the original image dimensions
    classified_image = labels.reshape(image_data.shape[:2])

    return classified_image

# Example usage:
api_key = "<YOUR_USGS_API_KEY>"
output_path = "path_to_save_downloaded_images"
latitude = 37.7749
longitude = -122.4194
start_date = '2022-01-01'
end_date = '2022-12-31'

# Download Landsat images
download_landsat_images(api_key, output_path, latitude, longitude, start_date, end_date)

# Unzip downloaded images
zip_file_path = os.path.join(output_path, "downloaded_scene.zip")
unzip_landsat_images(zip_file_path, output_path)

# Classify the image
scene_folder = os.listdir(output_path)[0]  # Assuming there is only one scene in the folder
scene_path = os.path.join(output_path, scene_folder, "path_to_raster.tif")  # Modify this based on your actual data
classified_image = classify_image(scene_path)

# Use the classified image for further analysis or visualization






# download MODIS data and extract climate data
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from pyproj import Proj, transform

# Function to download MODIS LST image
def download_modis_lst(latitude, longitude, date):
    url = f'https://modis.ornl.gov/rst/api/v1/MOD11A1/subset?latitude={latitude}&longitude={longitude}&date={date}'
    response = requests.get(url)
    return response.content

# Function to extract temperature from MODIS LST image
def extract_temperature(modis_image):
    # Assuming the data is in GeoTIFF format
    img = Image.open(BytesIO(modis_image))
    temperature_data = np.array(img)

    # Extracting temperature values for a specific location (modify indices based on your data)
    temperature_value = temperature_data[50, 50]  # Example: extracting value at pixel (50, 50)

    return temperature_value

# Example usage:
latitude = 48.4724
longitude = 8.6626
date = '2022-01-01'

modis_image = download_modis_lst(latitude, longitude, date)
temperature_value = extract_temperature(modis_image)

print(f"Temperature at {latitude}, {longitude} on {date}: {temperature_value} K")
