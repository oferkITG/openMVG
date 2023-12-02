import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np

# Function to write PLY file
def write_ply(filename, points):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write("{} {} {}\n".format(point[0], point[1], point[2]))

# Reading the data
file_path = r"C:\work_itg\data\2023-11027 Aaron iphone open src app\2023-11-27T09-37-23\sfm_out\sfm_data.json"  # Replace with the path to your file
ply_filename = file_path+'.ply'  # Replace with your desired output path

# data = pd.read_csv(file_path, header=None)
# data.columns = ['Timestamp', 'X', 'Z', 'Y', 'Value1', 'Value2', 'Value3', 'Value4']
f = open(file_path)
sfm_data = json.load(f)
extrinsics = sfm_data['extrinsics']
x = np.array([])
y = np.array([])
z = np.array([])
for cam in extrinsics:
    x = np.append(x, [cam['value']['center'][0]])
    y = np.append(y, [cam['value']['center'][1]])
    z = np.append(z, [cam['value']['center'][2]])


# Calculate the range for each axis
max_range = np.array([x.max()-x.min(), 
                      y.max()-y.min(), 
                      z.max()-z.min()]).max() / 2.0

# Find the center point for each axis
mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5

# Creating a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the limits for each axis to the same range
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.scatter(x, y, z, c='blue', marker='o', s=1)
ax.set_title('3D Plot of AR Poses')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')

# Export to PLY
#write_ply(ply_filename, data[['X', 'Y', 'Z']].values)

plt.show(block=True)
