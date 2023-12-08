import argparse
import torch
from utils.vpr_utils import get_nearest_neighbors
from loaders.VPRDataset import VPRDataset
from models.NetVLAD import NetVLAD
from options import options
import pickle
import os
from src.geo_utils.UtmFrameModel import *
'''
shelef_toy output/tiff/tiff_tiff_eigenplaces_encoding.pth 
/mnt/rawdata/aza/ortophoto/tiff/
/mnt/rawdata/aza/DSM/dsm_trex_w84utm36_12m.tif
isd/
--encoder eigenplaces
'''

def localize_dataset(orthophoto_path, dsm_file, isd_path,
                     dataset_path, ref_file,
                     encoder_type, encoder_options,
                     device_id, k, sim_metric):

    if torch.cuda.is_available():
        device_id = device_id
    else:
        device_id = 'cpu'
    device = torch.device(device_id)
    print("Localizing a shelef dataset with {}".format(encoder_type))
    print("Create and load the model for image encoding and retrieval")
    if encoder_type == "cosplace":
        # https://github.com/gmberton/CosPlace
        backbone = encoder_options["backbone"]
        dim = encoder_options["dim"]
        transform = encoder_options["transforms"]
        net = torch.hub.load("gmberton/cosplace", "get_trained_model",
                             backbone=backbone, fc_output_dim=dim)
    elif encoder_type == "netvlad":
        net = NetVLAD()
        net.load_state_dict(torch.load(encoder_options["weights"], map_location=device_id))
        transform = encoder_options["transforms"]
        dim = encoder_options["dim"]
    elif encoder_type == "eigenplaces":
        # https://github.com/gmberton/EigenPlaces
        backbone = encoder_options["backbone"]
        dim = encoder_options["dim"]
        transform = encoder_options["transforms"]
        net = torch.hub.load("gmberton/eigenplaces",
                               "get_trained_model", backbone=backbone, fc_output_dim=dim)
    else:
        raise NotImplementedError

    print("Create the dataset and the loader")
    loader_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 4}
    dataset = VPRDataset(dataset_path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, **loader_params)

    localization_results = {}
    print("Load the encoding of the reference dataset from: {}".format(ref_file))
    ref = torch.load(ref_file)

    # Get image identifiers and corresponding files
    isd_files = [f for f in os.listdir(isd_path) if f.endswith(".shelef")]
    shelef_img_ids = [f.split(".")[0] for f in os.listdir(dataset_path)]
    isd_files = [f for f in isd_files if f.split(".")[0] in shelef_img_ids]

    # Read orthophoto folder
    print("Create the OrthophotoProvider from {}".format(orthophoto_path))
    ortho_provider = OrthophotoProvider(orthophoto_path)
    # Read dsm file
    print("Create the DSMProvider from {}".format(dsm_file))
    # dsm_provider = DSMProvider(dsm_file, GeoTiffDirType.SingleFileType)
    dsm_provider = DSMProvider(None, GeoTiffDirType.SingleValue, 80)
    with torch.no_grad():
        net.eval()
        net.to(device)
        for i, minibatch in enumerate(loader):
            imgs = minibatch.get('img').to(device)
            global_desc = net(imgs)


            query_id = loader.dataset.img_ids[i]
            ref_encoding = ref["encoding"].to(device)
            ref_ids = ref["img_ids"]
            ref_files = ref["img_files"]
            ref_metadata = ref["metadata"]
            ref_dataset_path = ref["dataset_path"]

            # find the relevant ids for the img
            utm_camera, angles, Kinv, utm_target, img_size = import_shelef_json(os.path.join(
                isd_path, isd_files[i]))
            frame_model = UtmFrameModel(utm_camera, angles, Kinv, img_size)
            frame_model.load_dsm(dsm_provider)
            matching_tiles = ortho_provider.get_matching_tiles(frame_model.get_frame_roi())
            matching_tiles = [os.path.basename(f).split(".")[0] for f in matching_tiles]
            matching_tiles_idx = []
            for tile_id in matching_tiles:
                matching_tiles_idx.append(ref_ids.index(tile_id))
            knn, scores = get_nearest_neighbors(global_desc, ref_encoding[matching_tiles_idx, :], k,
                                                metric=sim_metric, return_scores=True)


            scores = scores.cpu().numpy()
            print("#"*10)
            print("Localization Results for Query: {}".format(query_id))
            matches = {}
            for j in range(k):
                nn_id = np.array(ref_ids)[matching_tiles_idx][knn[j]]
                print("neighbor #{} [{:.4f}] - id: {}".format(j+1, scores[j], nn_id))
                if ref_metadata is None:
                    transformation = None
                else:
                    ref_metadata[nn_id]
                matches[j] = {nn_id:{"score":scores[j],
                                     "transformation":transformation,
                                     "file":os.path.join(ref_dataset_path,
                                                         np.array(ref_files)[matching_tiles_idx][knn[j]])
                                     }}

            localization_results[query_id] = matches
            # convert to 4x4 and save

    # save localization results
    output_pkl_file = os.path.join(dataset_path, "localization_results.pkl")
    with open(output_pkl_file, "wb") as fp:
        pickle.dump(localization_results, fp)
    print("Localization results were saved to: {}".format(output_pkl_file))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset_path", help="path to the physical location of the target dataset")
    arg_parser.add_argument("reference_file", help="path to the physical location of the reference encoding")
    arg_parser.add_argument("orthophoto_path",
                            help="path to full orthophoto dataset")
    arg_parser.add_argument("dsm_file",
                            help="path to the dsm file")
    arg_parser.add_argument("isd_path", help="a path to a dataset with isd files")
    arg_parser.add_argument("--encoder", default="eigenplaces",
                            help="supported models: netvlad, cosplace, eigenplaces")
    arg_parser.add_argument("--device", default="cuda:0",
                            help="device identifier")
    arg_parser.add_argument("--sim_metric", default="cosine_sim")
    arg_parser.add_argument("--k", default=3, type=int)
    args = arg_parser.parse_args()

    encoder_options = options[args.encoder]
    localize_dataset(args.orthophoto_path, args.dsm_file, args.isd_path,
                     args.dataset_path, args.reference_file,
                     args.encoder, encoder_options,
                     args.device, args.k, args.sim_metric
                     )

