from osgeo import gdal
import numpy as np

# Input and output file paths
input_file = './ms-data/class_10_val/map_images/Industrial_473.tif'  # Replace with the actual input file path
output_file = "<output>"  # Replace with the actual output file path

# Open the input dataset
dataset = gdal.Open(input_file)

# Read bands 4, 3, and 2 (assuming these are the RGB bands)
band4 = dataset.GetRasterBand(4).ReadAsArray().astype(np.float32)  # Red
band3 = dataset.GetRasterBand(3).ReadAsArray().astype(np.float32)  # Green
band2 = dataset.GetRasterBand(2).ReadAsArray().astype(np.float32)  # Blue

# Define the input and output ranges for scaling
input_min = 0
input_max = 2750
output_min = 1
output_max = 255

# Apply scaling to each band
def scale_band(band, input_min, input_max, output_min, output_max):
    scaled_band = np.clip(((band - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min,
                          output_min, output_max)
    return scaled_band.astype(np.uint8)

band4_scaled = scale_band(band4, input_min, input_max, output_min, output_max)
band3_scaled = scale_band(band3, input_min, input_max, output_min, output_max)
band2_scaled = scale_band(band2, input_min, input_max, output_min, output_max)



# Create the output dataset in JPEG format
driver = gdal.GetDriverByName('JPEG')
output_dataset = driver.Create(output_file, dataset.RasterXSize, dataset.RasterYSize, 3, gdal.GDT_Byte, ['QUALITY=100'])

# Set NoData value to 0 (optional, if needed)
output_dataset.GetRasterBand(1).SetNoDataValue(0)
output_dataset.GetRasterBand(2).SetNoDataValue(0)
output_dataset.GetRasterBand(3).SetNoDataValue(0)

# Write the scaled bands to the output dataset
output_dataset.GetRasterBand(1).WriteArray(band4_scaled)  # Red channel
output_dataset.GetRasterBand(2).WriteArray(band3_scaled)  # Green channel
output_dataset.GetRasterBand(3).WriteArray(band2_scaled)  # Blue channel

# Flush and close the datasets
output_dataset.FlushCache()
output_dataset = None
dataset = None

print("Conversion and scaling completed successfully.")

