import argparse

import numpy as np

from src.geo_utils.UtmFrameModel import *
import matplotlib.pyplot as plt
import pickle
import os
import cv2
'''
Given ISD folder and dataset of orthophoto tiles,
for each ISD file, sample poses around center camera pose and generate views
'''

def generate_views(orthophoto_path, dsm_file, isd_path, output_path,
                   shelef_path=None,
                   max_shift_east=1500, shift_step_east=500,
                   max_shift_north=100, shift_step_north=100,
                   resize_model_scale=4,
                   fov_factor=1.0, resize_orthophoto_scale=0.25,
                   plot=False
                   ):
    # Get image identifiers and corresponding files
    isd_files = [f for f in os.listdir(isd_path) if f.endswith(".shelef")]
    if shelef_path is not None:
        shelef_img_ids = [f.split(".")[0] for f in os.listdir(shelef_path)]
        isd_files = [f for f in isd_files if f.split(".")[0] in shelef_img_ids]
    img_ids = [f.split(".")[0] for f in isd_files]

    # Read orthophoto folder
    print("Create the OrthophotoProvider from {}".format(orthophoto_path))
    ortho_provider = OrthophotoProvider(orthophoto_path)
    # Read dsm file
    print("Create the DSMProvider from {}".format(dsm_file))
    # dsm_provider = DSMProvider(dsm_file, GeoTiffDirType.SingleFileType)
    dsm_provider = DSMProvider(None, GeoTiffDirType.SingleValue, 80)
    view_filenames = []
    for i, img_id in enumerate(img_ids):
        print("Start generating views for image: {}".format(img_id))
        curr_dataset_path = os.path.join(output_path, img_id)
        os.makedirs(curr_dataset_path, exist_ok=True)
        # Read and create the original camera model
        print("Create the original camera model")
        utm_camera, angles, Kinv, utm_target, img_size = import_shelef_json(os.path.join(
            isd_path, isd_files[i]))
        original_frame_model = UtmFrameModel(utm_camera, angles, Kinv, img_size)
        original_frame_model.load_dsm(dsm_provider)

        # Sample and generate views
        print("Sample and generate views")
        view_num = 0
        look_at_transformations = {}
        for curr_shift_east in np.arange(-max_shift_east,
                                         max_shift_east + 1, shift_step_east):
            for curr_shift_north in np.arange(-max_shift_north,
                                              max_shift_north + 1, shift_step_north):
                # Create a new view
                view_num += 1

                # Compute the shifted target
                new_utm_target = utm_target + np.array([curr_shift_east, curr_shift_north, 0])
                new_utm_target = np.concatenate(
                    [new_utm_target[:2], [original_frame_model.use_dsm.get_height_vec(new_utm_target)]])

                # Compute the transformation for saving along with the generated view
                look_transformation = calc_lookat_matrix(new_utm_target, utm_camera)

                # Load the model
                new_model = UtmFrameModel.from_lookat(utm_camera, new_utm_target, Kinv, img_size)

                # Resize model
                new_model = new_model.resize_factor(resize_model_scale)

                # Apply the FOV factor
                curr_size = new_model.image_size
                new_model.image_size = [int(curr_size[0] * (1 + 2 * fov_factor)),
                                        int(curr_size[1] * (1 + 2 * fov_factor))]
                H = np.array([[1, 0, -curr_size[0] * fov_factor],
                              [0, 1, -curr_size[1] * fov_factor], [0, 0, 1]])
                new_model.Kinv = new_model.Kinv @ H

                try:
                    # Load the orthophoto
                    new_model.load_ortho(ortho_provider)

                    # resize orthophoto only if shape > 10000
                    ortho_shape = new_model.use_ortho.image.shape
                    if np.min(ortho_shape[:2]) > 10000:
                        new_model.use_ortho = new_model.use_ortho.resize(
                            int(ortho_shape[0] * resize_orthophoto_scale)
                            , int(ortho_shape[1] * resize_orthophoto_scale))
                    new_model.load_dsm(dsm_provider)
                    new_view_res, _ = new_model.quick_ortho_view()

                    # save
                    view_id = "{}_{}".format(img_id, view_num)
                    view_filename = os.path.join(curr_dataset_path, view_id + ".PNG")
                    view_filenames.append(view_filename)
                    cv2.imwrite(view_filename, new_view_res)
                    look_at_transformations[view_id] = {"look_transformation": look_transformation,
                                                        "fov": fov_factor}
                    print("view {} saved to {}".format(view_num, view_filename))
                    if plot:
                        new_view = new_view_res[:, :, ::-1]  # BGR to RGB
                        plt.imshow(new_view)
                        plt.show()

                except KeyError as e:
                    print("Cannot generate view # {}".format(view_num))
                    print("Tile missing: {}".format(e))

        output_pkl_file = os.path.join(curr_dataset_path, "view_transformations.pkl")
        with open(output_pkl_file, "wb") as fp:
            pickle.dump(look_at_transformations, fp)

class CoarseHypotesis:
    params : np.array
    img_path : str
    encoding : np.array
    image_size : list
    Kinv: np.array
    fov_factor : float
    resize_model_scale : float
    def __init__(self, params_, image_size_, Kinv_, fov_, resize_model_scale_, img_path_):
        self.params = params_
        self.image_size = image_size_
        self.Kinv = Kinv_
        self.fov_factor = fov_
        self.resize_model_scale = resize_model_scale_
        self.image_path = img_path_
        self.encoding = None


def generate_candidates(hypo_cache, orthophoto_path, dsm_file, isd_path, vpr_cache_path,
                       shelef_path=None,
                       max_shift_east=1500, shift_step_east=500,
                       max_shift_north=100, shift_step_north=100,
                       resize_model_scale=4,
                       fov_factor=1.0, resize_orthophoto_scale=0.25,
                       plot=False
                       ):
    # Get image identifiers and corresponding files
    isd_files = [f for f in os.listdir(isd_path) if f.endswith(".shelef")]
    if shelef_path is not None:
        shelef_img_ids = [f.split(".")[0] for f in os.listdir(shelef_path)]
        isd_files = [f for f in isd_files if f.split(".")[0] in shelef_img_ids]
    img_ids = [f.split(".")[0] for f in isd_files]

    # Read orthophoto folder
    print("Create the OrthophotoProvider from {}".format(orthophoto_path))
    ortho_provider = OrthophotoProvider(orthophoto_path)
    # Read dsm file
    print("Create the DSMProvider from {}".format(dsm_file))
    # dsm_provider = DSMProvider(dsm_file, GeoTiffDirType.SingleFileType)
    dsm_provider = DSMProvider(None, GeoTiffDirType.SingleValue, 80)
    hypo_cache_params = [x.params for k,x in hypo_cache.items()]
    hypo_cache_names = list(hypo_cache.keys())
    look_at_hypo_id = dict()
    for i, img_id in enumerate(img_ids):
        print("Start generating views for image: {}".format(img_id))
        os.makedirs(vpr_cache_path, exist_ok=True)
        # Read and create the original camera model
        print("Create the original camera model")
        origin_camera, angles, Kinv, utm_target_org, img_size = import_shelef_json(os.path.join(
            isd_path, isd_files[i]))
        original_frame_model = UtmFrameModel(origin_camera, angles, Kinv, img_size)
        original_frame_model.load_dsm(dsm_provider, 2000)

        # Sample and generate views
        print("Generate Candidates")
        view_num = 0
        look_at_transformations = dict()
        look_at_hypo_id[img_id] = []
        hypo_image_list = []
        for curr_shift_east in np.arange(-max_shift_east,
                                         max_shift_east + 1, shift_step_east):
            for curr_shift_north in np.arange(-max_shift_north,
                                              max_shift_north + 1, shift_step_north):
                # Create a new view
                view_num += 1

                # Compute the shifted target
                new_utm_target = original_frame_model.use_dsm.get_UTM(utm_target_org + np.array([curr_shift_east, curr_shift_north, 0]))

                current_params = np.concatenate([origin_camera, new_utm_target])
                norm_error = []
                if len(hypo_cache_params)>0:
                    norm_error = np.abs((np.array(hypo_cache_params)-current_params)/np.array([100, 100., 100, 200, 200, 200]))
                if len(norm_error) > 0 and np.min(norm_error.max(axis=1)) < 1:
                    look_at_hypo_id[img_id].append(hypo_cache_names[np.argmin(norm_error.max(axis=1))])
                else:
                    hypo_name = 'Hypo_%s'% len(hypo_cache_names)
                    assert ( hypo_name not in hypo_cache_names)
                    new_model = UtmFrameModel.from_lookat(origin_camera, new_utm_target, Kinv, img_size)\
                        .resize_factor(resize_model_scale).enlarge_fov(fov_factor)
                    new_model.load_ortho(ortho_provider).load_dsm(dsm_provider)
                    new_view_res, _ = new_model.quick_ortho_view()

                    view_filename = os.path.join(vpr_cache_path, hypo_name + ".PNG")
                    cv2.imwrite(view_filename, new_view_res)
                    hypo_image_list.append(new_view_res)
                    hypo_cache[hypo_name] = CoarseHypotesis(current_params, img_size, Kinv, fov_factor, resize_model_scale, view_filename)
                    look_at_hypo_id[img_id].append(hypo_name)
                    hypo_cache_params.append(current_params)
                    hypo_cache_names.append(hypo_name)

    return look_at_hypo_id, hypo_image_list





if 0:
    # Compute the transformation for saving along with the generated view
    look_transformation = calc_lookat_matrix(new_utm_target, utm_camera)

    # Load the model
    new_model = UtmFrameModel.from_lookat(utm_camera, new_utm_target, Kinv, img_size)

    # Resize model
    new_model = new_model.resize_factor(resize_model_scale)

    # Apply the FOV factor
    curr_size = new_model.image_size
    new_model.image_size = [int(curr_size[0] * (1 + 2 * fov_factor)),
                            int(curr_size[1] * (1 + 2 * fov_factor))]
    H = np.array([[1, 0, -curr_size[0] * fov_factor],
                  [0, 1, -curr_size[1] * fov_factor], [0, 0, 1]])
    new_model.Kinv = new_model.Kinv @ H

    try:
        # Load the orthophoto
        new_model.load_ortho(ortho_provider)

        # resize orthophoto only if shape > 10000
        ortho_shape = new_model.use_ortho.image.shape
        if np.min(ortho_shape[:2]) > 10000:
            new_model.use_ortho = new_model.use_ortho.resize(
                int(ortho_shape[0] * resize_orthophoto_scale)
                , int(ortho_shape[1] * resize_orthophoto_scale))
        new_model.load_dsm(dsm_provider)
        new_view_res, _ = new_model.quick_ortho_view()

        # save
        view_id = "{}_{}".format(img_id, view_num)
        view_filename = os.path.join(curr_dataset_path, view_id + ".PNG")
        cv2.imwrite(view_filename, new_view_res)
        look_at_transformations[view_id] = {"look_transformation": look_transformation,
                                            "fov": fov_factor}
        print("view {} saved to {}".format(view_num, view_filename))
        if plot:
            new_view = new_view_res[:, :, ::-1]  # BGR to RGB
            plt.imshow(new_view)
            plt.show()

    except KeyError as e:
        print("Cannot generate view # {}".format(view_num))
        print("Tile missing: {}".format(e))

    output_pkl_file = os.path.join(curr_dataset_path, "view_transformations.pkl")
    with open(output_pkl_file, "wb") as fp:
        pickle.dump(look_at_transformations, fp)
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("orthophoto_path",
                            help="path to full orthophoto dataset")
    arg_parser.add_argument("dsm_file",
                            help="path to the dsm file")
    arg_parser.add_argument("isd_path", help="a path to a dataset with isd files")
    arg_parser.add_argument("output_path", help="output path")
    arg_parser.add_argument("--shelef_path", help="path to folder with shelef images to filter isd")
    arg_parser.add_argument("--max_shift_east", type=int, default=1500)
    arg_parser.add_argument("--shift_step_east", type=int, default=500)
    arg_parser.add_argument("--max_shift_north", type=int, default=150)
    arg_parser.add_argument("--shift_step_north", type=int, default=150)
    arg_parser.add_argument("--resize_model_scale", type=int, default=4)
    arg_parser.add_argument("--resize_orthophoto_scale", type=float, default=0.25)
    arg_parser.add_argument("--fov_factor", type=float, default=1)
    arg_parser.add_argument("--plot", action='store_true')

    args = arg_parser.parse_args()
    generate_views(args.orthophoto_path, args.dsm_file,
                   args.isd_path, args.output_path,
                   args.shelef_path,
                   args.max_shift_east,
                   args.shift_step_east,
                   args.max_shift_north,
                   args.shift_step_north,
                   args.resize_model_scale,
                   args.fov_factor,
                   args.resize_orthophoto_scale,
                   args.plot
                   )

