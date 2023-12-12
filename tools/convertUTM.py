import csv
from pyproj import Proj, transform
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

def convert_utm_to_lat_lon(utm_coords):
    """
    Convert UTM coordinates to latitude and longitude.

    Parameters:
    utm_coords (list of tuples): List of tuples containing UTM coordinates in the format (Easting, Northing, Elevation).

    Returns:
    list of tuples: List of tuples containing latitude and longitude.
    """
    # Define the UTM zone 36N coordinate system
    utm_proj = Proj(proj='utm', zone=36, ellps='WGS84', south=False)

    # Define the WGS84 latitude/longitude coordinate system
    wgs84_proj = Proj(proj='latlong', datum='WGS84')

    # Convert UTM coordinates to WGS84 latitude/longitude
    lat_lon_coords = [transform(utm_proj, wgs84_proj, easting, northing) for easting, northing, _ in utm_coords]

    return lat_lon_coords


def find_transformation(src_points, dst_points):
    # Calculate centroids
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)

    # Center the points around the centroid
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid

    # Compute the covariance matrix
    H = np.dot(src_centered.T, dst_centered)

    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H)

    # Calculate rotation
    rotation = np.dot(Vt.T, U.T)

    # Special reflection case
    if np.linalg.det(rotation) < 0:
       Vt[2, :] *= -1
       rotation = np.dot(Vt.T, U.T)

    # Calculate translation
    translation = dst_centroid - np.dot(rotation, src_centroid)

    return rotation, translation

def apply_transformation_to_obj(file_path, rotation, translation, output_path):
    transformed_lines = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex line
                vertex = np.array(list(map(float, line.strip().split()[1:])))
                # Apply rotation and translation
                transformed_vertex = np.dot(rotation, vertex) + translation
                transformed_line = 'v ' + ' '.join(map(str, transformed_vertex)) + '\n'
                transformed_lines.append(transformed_line)
            elif line.startswith('f '):  # Face line
                # Keep face line unchanged
                transformed_lines.append(line)
            else:
                # You can add handling for other line types (like texture coordinates or normals) if needed
                pass

    with open(output_path, 'w') as file:
        file.writelines(transformed_lines)

def apply_transformation_to_anchors(file_path, rotation, translation, output_path):
    transformed_lines = []

    with open(file_path, 'r') as file:
        for line in file:
            if not line[0].isdigit():
                continue

            vertex = np.array(list(map(float, line.strip().split(",")[2:5])))
            # Apply rotation and translation
            transformed_vertex = np.dot(rotation, vertex) + translation
            transformed_lines.append(transformed_vertex)

    with open(output_path, 'w') as file:
        file.write("X,Y,Z,\n")
        for line in transformed_lines:
            for item in line:
                file.write(str(item))
                file.write(",")
            file.write("\n")


# UTM coordinates (Easting, Northing, Elevation) for each waypoint
utm = [
    (658908.576, 3508150.229, 90.612),  # Waypoint 1
    (658900.122, 3508144.225, 90.416), # Waypoint 2
    (658893.982, 3508154.062, 90.518) # Waypoint 3
]


# these xyz values are only for scan AaronScan_3Points_2023-12-05T11-09-58/Anchors.txt 
# for a different scan, you need to read in the anchors
xyz = [
    (0,0.276004,0.592119,-0.508760), # Anchor 1
    (1,0.706804,0.510343,9.811332), # Anchor 2
    (2,-11.117970,0.537205,9.370130) # Anchor 3
]

src_origin = xyz[0]
dst_origin = utm[0]

# Convert UTM and XYZ lists to numpy arrays for the transformation function
# Assuming the first element in each xyz tuple is an identifier and not a coordinate
src_points = np.array([np.subtract(point[1:], src_origin[1:]) for point in xyz])
dst_points = np.array([np.subtract(point, dst_origin) for point in utm])


# Convert and print the latitude/longitude coordinates
lat_lon_coordinates = convert_utm_to_lat_lon(utm)
for idx, (lat, lon) in enumerate(lat_lon_coordinates, start=1):
    print(f"Waypoint {idx}: Latitude {lat}, Longitude {lon}")

rotation, translation = find_transformation(src_points, dst_points)

# input_obj = 'AaronScan_3Points_2023-12-05T11-09-58/combined_mesh.obj'
# output_obj = 'AaronScan_3Points_2023-12-05T11-09-58/world oriented_mesh.obj' 
# apply_transformation_to_obj(input_obj, rotation, translation, output_obj)

input_obj = '/DATA/AaronScan_3Points_2023-12-05T11-09-58/Anchors.txt'
output_obj = '/DATA/AaronScan_3Points_2023-12-05T11-09-58/Anchors_world_oriented.txt' 
apply_transformation_to_anchors(input_obj, rotation, translation, output_obj)

# def main(input_obj, output_obj, utm_coords, xyz_coords):
#     # Convert UTM and XYZ lists to numpy arrays for the transformation function
#     src_points = np.array([point[1:] for point in xyz_coords])
#     dst_points = np.array(utm_coords)

#     # Convert and print the latitude/longitude coordinates
#     lat_lon_coordinates = convert_utm_to_lat_lon(utm_coords)
#     for idx, (lat, lon) in enumerate(lat_lon_coordinates, start=1):
#         print(f"Waypoint {idx}: Latitude {lat}, Longitude {lon}")

#     rotation, translation = find_transformation(src_points, dst_points)
#     apply_transformation_to_obj(input_obj, rotation, translation, output_obj)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Apply geometric transformation to OBJ file.')
#     parser.add_argument('input_obj', type=str, help='Input OBJ file path')
#     parser.add_argument('output_obj', type=str, help='Output OBJ file path')
#     parser.add_argument('--utm', nargs='+', type=float, help='UTM coordinates (Easting, Northing, Elevation) for each waypoint', required=True)
#     parser.add_argument('--xyz', nargs='+', type=float, help='XYZ coordinates for each waypoint', required=True)

#     args = parser.parse_args()

#     # Processing UTM and XYZ coordinates
#     utm_coords = list(zip(args.utm[::3], args.utm[1::3], args.utm[2::3]))
#     xyz_coords = list(zip(args.xyz[::4], args.xyz[1::4], args.xyz[2::4], args.xyz[3::4]))

#     main(args.input_obj, args.output_obj, utm_coords, xyz_coords)


