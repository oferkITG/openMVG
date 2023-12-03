import matplotlib
matplotlib.use('TkAgg')
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


def get_pose_xz(file_path):

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

    return x, y, z

if __name__ == '__main__':

    # Reading the data
    rec_file_path = "/DATA/ITG/ios_logger_noDepth/2023-11-27T09-37-23-small/sfm_out/sfm_data.json"  # Replace with the path to your file
    gt_file_path = "/DATA/ITG/ios_logger_noDepth/2023-11-27T09-37-23-small/sfm_out/sfm_data_gt.json"  # Replace with the path to your file


    x, y, z = get_pose_xz(rec_file_path)
    x_gt, y_gt, z_gt = get_pose_xz(gt_file_path)


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
    
    ax.scatter(x, y, z, c='blue', marker='o', s=10)  # Increase the marker size to 10
    ax.scatter(x_gt, y_gt, z_gt, c='red', marker='^', s=10)  # Increase the marker size to 10
    ax.set_title('3D Plot of AR Poses')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Export to PLY
    #write_ply(ply_filename, data[['X', 'Y', 'Z']].values)

    plt.show(block=True)
