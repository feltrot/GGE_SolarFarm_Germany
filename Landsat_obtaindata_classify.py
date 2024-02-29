
# get Landsat data
import landsatxplore.api
import os
import requests
import zipfile
from io import BytesIO
import arcpy
import numpy as np
from sklearn.cluster import KMeans




# download landsat images (following this link: https://towardsdatascience.com/downloading-landsat-satellite-images-with-python-a2d2b5183fb7)
from landsatxplore.api import API
import requests

# Your USGS  credentials
username = "ftrott"
password = "rimhy1-tiwzoh-wepnIh"

# Initialize a new API instance
api = API(username, password)

# Perform a request
response = api.request(endpoint="dataset-catalogs")
print(response)



# Search for Landsat TM scenes
scenes = api.search(
    dataset='landsat_ot_c2_l2',
    latitude=51.5226,
    longitude=7.8414,
    start_date='2023-05-01',
    end_date='2023-08-31',
    max_cloud_cover=50
)

# log out
api.logout()



import pandas as pd

# Create a DataFrame from the scenes
df_scenes = pd.DataFrame(scenes)
df_scenes = df_scenes[['display_id','wrs_path', 'wrs_row','satellite','cloud_cover','acquisition_date']]
df_scenes.sort_values('acquisition_date', ascending=False, inplace=True)

# set main directory
directory = '/Users/felixtrotter/Documents/Training/Projects/GGE_Solar/Data_samples'

# load EarthEcplorer from landsat package
from landsatxplore.earthexplorer import EarthExplorer
import os

# Initialize the API
ee = EarthExplorer(username, password)

# loop through each satellite file stored in the dataframe
for i in df_scenes["display_id"]:
    print(i)
    # Select the first scene
    ID = i

    # Download the scene 
    try: 
        ee.download(ID, output_dir=directory)
        print('{} succesful'.format(ID))
        
    # Additional error handling
    except:
        if os.path.isfile(os.path.join(directory, '/{}.tar').format(ID)):
            print('{} error but file exists'.format(ID))
        else:
            print('{} error'.format(ID))

ee.logout()

# import package to extract data from compressed tar folder
import tarfile

# loop through each file
for file in os.listdir(directory):
    print(file)
    # ignore hidden files
    if not file.endswith('DS_Store'):
        # Extract files from tar archiveßß
        tar = tarfile.open(os.path.join(directory, file))
        tar.extractall(directory)
        tar.close()


# we need to give each satelite image that we create from the bands a 
# CRS. Here we use EPSG 4326 = WGS84
# load packages
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.plot import plotting_extent
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling


# set crs
dst_crs = 'EPSG:4326'

# create function that can reproject rasters
def reproject_raster(in_path, out_path):
# reproject raster to project crs
    with rasterio.open(in_path) as src:
        src_crs = src.crs
        transform, width, height = calculate_default_transform(src_crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()

        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height})

        with rasterio.open(out_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
        return(out_path)




import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import rioxarray
from affine import Affine
import mercantile

import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

# create directory oif not yet exists
dir_RGB_images = os.path.join(directory, 'RGB_images')
if not os.path.exists(dir_RGB_images):
    os.makedirs(dir_RGB_images)

RBG_ls = []
for ii in df_scenes["display_id"]:
    print(ii)
    # Select the first scene
    ii = "LC09_L2SP_197024_20230826_20230828_02_T1"
    ID = ii
    # Load Blue (B2), Green (B3) and Red (B4) bands
    B2 = tiff.imread(os.path.join(directory, '{}_SR_B2.TIF').format(ID, ID))
    B3 = tiff.imread(os.path.join(directory, '{}_SR_B3.TIF').format(ID, ID))
    B4 = tiff.imread(os.path.join(directory, '{}_SR_B4.TIF').format(ID, ID))

    # Stack and scale bands
    RGB = np.dstack((B4, B3, B2))
    RGB = np.clip(RGB*0.0000275-0.2, 0, 1)

    # Clip to enhance contrast
    RGB = np.clip(RGB,0,0.2)/0.2

    # write CRS
    #RGB.rio.write_crs("epsg:4326", inplace=True)

    # save image as tif file in new directory
    tiff.imwrite(dir_RGB_images + '/' + 'RGB__' + ii + '.tif', RGB)

    #rds = rioxarray.open_rasterio(dir_RGB_images + '/' + 'RGB__' + ii + '.tif')
    #rds.rio.write_crs("epsg:4326", inplace=True)
    #rds_4326 = rds.rio.reproject("EPSG:4326")
    #rds_4326.rio.to_raster(dir_RGB_images + '/' + 'RGB__' + ii + '.tif', RGB)
    with rasterio.open(dir_RGB_images + '/' + 'RGB__' + ii + '.tif') as src:
        with WarpedVRT(src, crs='EPSG:4326',
                    resampling=Resampling.bilinear) as vrt:

            # Determine the destination tile and its mercator bounds using
            # functions from the mercantile module.
            dst_tile = mercantile.tile(*vrt.lnglat(), 9)
            left, bottom, right, top = mercantile.xy_bounds(*dst_tile)

            # Determine the window to use in reading from the dataset.
            dst_window = vrt.window(left, bottom, right, top)

            # Read into a 3 x 512 x 512 array. Our output tile will be
            # 512 wide x 512 tall.
            data = vrt.read(window=dst_window, out_shape=(3, 512, 512))

            # Use the source's profile as a template for our output file.
            profile = vrt.profile
            profile['width'] = 512
            profile['height'] = 512
            profile['driver'] = 'GTiff'

            # We need determine the appropriate affine transformation matrix
            # for the dataset read window and then scale it by the dimensions
            # of the output array.
            dst_transform = vrt.window_transform(dst_window)
            scaling = Affine.scale(dst_window.height / 512,
                                dst_window.width / 512)
            dst_transform *= scaling
            profile['transform'] = dst_transform

            # Write the image tile to disk.
            with rasterio.open(dir_RGB_images + '/' + 'RGB__' + ii + '.tif', 'w', **profile) as dst:
                dst.write(data)

    # append each RGB image to the list
    #RBG_ls.append(RGB)
    # Display RGB image
    #fig, ax = plt.subplots(figsize=(10, 10))
    #plt.imshow(RGB)
    #ax.set_axis_off()







ii = "LC09_L2SP_197024_20230826_20230828_02_T1"
ID = ii
# Load Blue (B2), Green (B3) and Red (B4) bands
B2 = tiff.imread(os.path.join(directory, '{}_SR_B2.TIF').format(ID, ID))
B3 = tiff.imread(os.path.join(directory, '{}_SR_B3.TIF').format(ID, ID))
B4 = tiff.imread(os.path.join(directory, '{}_SR_B4.TIF').format(ID, ID))

# Stack and scale bands
RGB = np.dstack((B4, B3, B2))
RGB = np.clip(RGB*0.0000275-0.2, 0, 1)

# Clip to enhance contrast
RGB = np.clip(RGB,0,0.2)/0.2

# write CRS
#RGB.rio.write_crs("epsg:4326", inplace=True)

# save image as tif file in new directory
tiff.imwrite(dir_RGB_images + '/' + 'RGB__' + ii + '.tif', RGB)



from osgeo import gdal

file = dir_RGB_images + '/' + 'RGB__' + ii + '.tif'
outFileName = dir_RGB_images + '/' + 'RGB__' + ii + '__mod' + '.tif'
ds = gdal.Open(file)
band = ds.GetRasterBand(1)
arr = band.ReadAsArray()
[rows, cols] = arr.shape
arr_min = arr.min()
arr_max = arr.max()
arr_mean = int(arr.mean())
arr_out = np.where((arr < arr_mean), 10000, arr)
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(outFileName, cols, rows, 1, gdal.GDT_UInt16)
outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(ds.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(arr_out)
outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!
outdata = None
band=None
ds=None



































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









import os
from glob import glob
import subprocess

import rasterio


# Prepare paths.
input_pattern = 'data/eudem/*.TIF'
input_paths = sorted(glob(input_pattern))
assert input_paths
vrt_path = 'data/eudem-vrt/eudem.vrt'
output_dir = 'data/eudem-buffered/'
os.makedirs(output_dir, exist_ok=True)

# EU-DEM specific options.
tile_size = 1_000_000
buffer_size = 50

for input_path in input_paths:

    # Get tile bounds.
    with rasterio.open(input_path) as f:
        bottom = int(f.bounds.bottom)
        left = int(f.bounds.left)

    # For EU-DEM only: round this partial tile down to the nearest tile_size.
    if left == 943750:
        left = 0

    # New tile name in SRTM format.
    output_name = 'N' + str(bottom).zfill(7) + 'E' + str(left).zfill(7) + '.TIF'
    output_path = os.path.join(output_dir, output_name)

    # New bounds.
    xmin = left - buffer_size
    xmax = left + tile_size + buffer_size
    ymin = bottom - buffer_size
    ymax = bottom + tile_size + buffer_size

    # EU-DEM tiles don't cover negative locations.
    xmin = max(0, xmin)
    ymin = max(0, ymin)

    # Do the transformation.
    cmd = [
        'gdal_translate',
        '-a_srs', 'EPSG:3035',  # EU-DEM crs.
        '-co', 'NUM_THREADS=ALL_CPUS',
        '-co', 'COMPRESS=DEFLATE',
        '-co', 'BIGTIFF=YES',
        '--config', 'GDAL_CACHEMAX','512',
        '-projwin', str(xmin), str(ymax), str(xmax), str(ymin),
        vrt_path, output_path,
    ]
    r = subprocess.run(cmd)
    r.check_returncode()



import elevation

# Set the bounding box for the region in Germany you are interested in
bbox = (7.8250615, 51.5097488, 7.8812025, 51.5315566)  # (min_lon, min_lat, max_lon, max_lat)

# Download the DEM data
elevation.clip(bounds=bbox, output='dem.tif')

print("DEM downloaded successfully.")
