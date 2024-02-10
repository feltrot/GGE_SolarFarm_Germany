
import ee
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

# Authenticate to the Earth Engine servers
ee.Initialize()

# Function to calculate solar radiance statistics for a region of interest
def calculate_solar_radiance_statistics(image):
    # Extract solar radiance band (change band name based on your dataset)
    radiance_band = image.select(['SOLAR_RADIANCE'])

    # Reduce the region (mean, total, and standard deviation)
    stats = radiance_band.reduceRegion(
        reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True),
        geometry=roi,
        scale=30,  # Adjust scale based on your requirements
        bestEffort=True
    )

    return stats

# Location coordinates
latitude = 48.4724
longitude = 8.6626

# Define the region of interest as a point
roi = ee.Geometry.Point([longitude, latitude])

# Date range
start_date = '2023-07-01'
end_date = '2023-09-30'

# Function to create a square polygon around a point
def create_square_polygon(point, size):
    x, y = point.x, point.y
    half_size = size / 2
    return Polygon([(x - half_size, y - half_size),
                    (x + half_size, y - half_size),
                    (x + half_size, y + half_size),
                    (x - half_size, y + half_size)])

# Download daily solar radiance images
dataset = (ee.ImageCollection('your_solar_radiance_dataset')
           .filterBounds(roi)
           .filterDate(start_date, end_date))

# Extract statistics for each image
image_stats = dataset.map(calculate_solar_radiance_statistics)

# Convert Earth Engine results to a Pandas DataFrame
data = pd.DataFrame(image_stats.getInfo().get('features'))

# Extract mean and stdDev values
data['mean_radiance'] = data['properties'].apply(lambda x: x['SOLAR_RADIANCE_mean'])
data['std_dev_radiance'] = data['properties'].apply(lambda x: x['SOLAR_RADIANCE_stdDev'])

# Find the top and bottom 100 pixels by mean radiance
top_100_pixels = data.nlargest(100, 'mean_radiance')
bottom_100_pixels = data.nsmallest(100, 'mean_radiance')

# Convert top and bottom 100 pixels to GeoDataFrames
top_gdf = gpd.GeoDataFrame(top_100_pixels, geometry=top_100_pixels['geometry'].apply(Polygon))
bottom_gdf = gpd.GeoDataFrame(bottom_100_pixels, geometry=bottom_100_pixels['geometry'].apply(Polygon))

# Save GeoDataFrames to shapefiles (you may need to adjust the paths)
top_gdf.to_file("path_to_save_top_100_pixels.shp")
bottom_gdf.to_file("path_to_save_bottom_100_pixels.shp")

# Display the results
print("Top 100 Pixels:")
print(top_gdf[['longitude', 'latitude', 'mean_radiance', 'std_dev_radiance']])
print("\nBottom 100 Pixels:")
print(bottom_gdf[['longitude', 'latitude', 'mean_radiance', 'std_dev_radiance']])
