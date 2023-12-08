import rasterio
from rasterio.enums import Resampling
import os
from os.path import join
def raster_resize(local_name, output_local_name, raster_resize_factor, dtype):
    if dtype == 'uint8':
        resample = Resampling.nearest
    else:
        resample = Resampling.bilinear
    with rasterio.open(local_name) as dataset:
        data = dataset.read(
            out_shape=(
                dataset.count,
                dataset.height // raster_resize_factor,
                dataset.width // raster_resize_factor),
            resampling=resample)

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        profile = dataset.profile.copy()

        profile.update({
            'dtype': dtype,
            'height': dataset.height // raster_resize_factor,
            'width': dataset.width // raster_resize_factor,
            'transform': transform})

        with rasterio.open(output_local_name, 'w', **profile) as dst:
            dst.write(data)

in_path = "/mnt/rawdata/aza/ortophoto/tiff"
out_path = "/mnt/artifacts/assets/vpr/aza_tiff_small"
raster_resize_factor = 4
dtype_str = 'uint8'
files = [f for f in os.listdir(in_path) if f.endswith(".tif")]
print(files)
for f in files:
    raster_resize(join(in_path,f), join(out_path,f),
                  raster_resize_factor, dtype_str)
