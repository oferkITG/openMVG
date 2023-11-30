import json
from scipy.spatial.transform import Rotation as R

def make_timestamp_frameIdx_dict(Frames_filepath: str):

    with open(Frames_filepath, 'r') as f:
        lines = f.readlines()
        frames_idx_dict = {}
        for ind, line in enumerate(lines):
            line = line.split(',')
            print(line)
            timestamp = line[0]
            frames_idx_dict[ind] = timestamp

    return frames_idx_dict 

def make_AR_poses_dict(ARposes_filepath: str):

    with open(ARposes_filepath, 'r') as f:
        lines = f.readlines()
        arposes = {}
        for ind, line in enumerate(lines):
            print(line)
            timestamp, x, y, z, qw, qx, qy, qz = line.split(',')
            arposes[timestamp] = [timestamp, x, y, z, qw, qx, qy, qz]

    return arposes
    

def add_poses(Frames_filepath, ARposes_filepath, sfm_data_file, output_file):
    # match frames to arposes by timestamp
    frames = make_timestamp_frameIdx_dict(Frames_filepath)
    arposes = make_AR_poses_dict(ARposes_filepath)
    
    # load sfm data generated from openMVG
    with open(sfm_data_file, 'r') as f:
        sfm_data = json.load(f)
            
    for ind, view in enumerate(sfm_data["views"]):
        key = view["key"]
        frame_id = view["value"]["ptr_wrapper"]["data"]["filename"].split(".")[0][len("frame"):]
        id_view = view["value"]["ptr_wrapper"]["data"]["id_view"]
        id_intrinsic = view["value"]["ptr_wrapper"]["data"]["id_intrinsic"]
        id_pose = view["value"]["ptr_wrapper"]["data"]["id_pose"]

        # get ARposes data according to frame_id
        timestamp, x, y, z, qw, qx, qy, qz = arposes[frames[int(frame_id)]]
        
        r = R.from_quat([float(qx), float(qy), float(qz), float(qw)])  
        r = r.as_matrix()
        r = r.tolist()
        pose_dict = {
            "key": key,
            "value": {
                    "rotation": r,
                    "center": [float(x), float(y), float(z)]
            }
                
        }

        sfm_data["extrinsics"].append(pose_dict)


    with open(output_file, '+a') as f:
        f.write(json.dumps(sfm_data, indent=4))
        f.write("\n")

if __name__ == '__main__':
    # create a dictionary of timestamps and frame indices
    Frames_filepath = '/DATA/ITG/ios_logger_noDepth/2023-11-27T09-37-23/Frames.txt'
    ARposes_filepath = '/DATA/ITG/ios_logger_noDepth/2023-11-27T09-37-23/ARposes.txt'

    # SfM data file to add poses to
    sfm_data_file = '/DATA/ITG/ios_logger_noDepth/2023-11-27T09-37-23/openMVG_out/matches/sfm_data.json'
    output_file = '/DATA/ITG/ios_logger_noDepth/2023-11-27T09-37-23/openMVG_out/matches/sfm_data_test.json'

    add_poses(Frames_filepath, ARposes_filepath, sfm_data_file, output_file)
    
        
        
    