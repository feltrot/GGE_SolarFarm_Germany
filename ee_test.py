

# import Earth Engine 
import ee
ee.Authenticate()

ee.Initialize()

# Load landsat image
Landsat8 = ee.Image('LANDSAT/LC08/C01/T1_TOA/LC08_170052_20170108').select(['B4', 'B3', 'B2'])

# create region
region = ee.Geometry.Rectangle(7.8176353, 51.5306785, 7.8751888, 51.5116292)

# Export to drive
task = ee.batch.Export.image.toDrive(**{
    'image': Landsat8,
    'description': 'imagetoDrive_L8',
    'folder' : 'ExampleData_Schafhausen',
    'scale' : 30,
    'region' : region.getInfo()['coordinates'] 
})
task.start()