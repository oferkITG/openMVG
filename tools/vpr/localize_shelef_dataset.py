import argparse
import torch
from tools.vpr.utils.vpr_utils import get_nearest_neighbors
from tools.vpr.loaders.VPRDataset import VPRDataset
from tools.vpr.models.NetVLAD import NetVLAD
from tools.vpr.options import options
import pickle
import os

def localize_dataset(dataset_path, reference_path,
        encoder_type, encoder_options,
        device_id, k, sim_metric):
    if torch.cuda.is_available():
        device_id = device_id
    else:
        device_id = 'cpu'
    device = torch.device(device_id)

    print("Localizing a shelef dataset with {}".format(encoder_type))
    encoder_options = encoder_options
    print("Create and load the model for image encoding and retrieval")
    if encoder_type in ("cosplace", "cosplace_reduce_shelef_x4"):
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
    load_per_img = True
    if os.path.isfile(reference_path):
        load_per_img = False
        ref_file = reference_path
        print("Load the encoding of the reference dataset from: {}".format(ref_file))
        ref = torch.load(ref_file)
    with torch.no_grad():
        net.eval()
        net.to(device)
        for i, minibatch in enumerate(loader):
            imgs = minibatch.get('img').to(device)
            global_desc = net(imgs)

            query_id = loader.dataset.img_ids[i]

            if load_per_img:
                ref_file = os.path.join(os.path.join(reference_path, query_id)
                                        ,"{}_{}_encoding.pth".format(query_id, encoder_type))
                if not os.path.exists(ref_file):
                    print("No encoding available for {}, skipping".format(query_id))
                    continue

                ref = torch.load(ref_file)
                if len(ref["img_ids"]) == 0:
                    print("Empty encoding for {}, skipping".format(query_id))
                    continue

                print("Loaded the encoding of the reference dataset from: {}".format(ref_file))

            ref_encoding = ref["encoding"].to(device)
            ref_ids = ref["img_ids"]
            ref_files = ref["img_files"]
            ref_metadata = ref["metadata"]
            ref_dataset_path = ref["dataset_path"]
            knn, scores = get_nearest_neighbors(global_desc, ref_encoding, k,
                                                metric=sim_metric, return_scores=True)

            scores = scores.cpu().numpy()
            print("#"*10)
            print("Localization Results for Query: {}".format(query_id))
            matches = {}
            for j in range(k):
                nn_id = ref_ids[knn[j]]
                print("neighbor #{} [{:.4f}] - id: {}".format(j+1, scores[j], nn_id))
                if ref_metadata is None:
                    metadata = None
                else:
                    metadata = ref_metadata[nn_id]
                matches[j] = {nn_id:{"score":scores[j],
                                     "metadata":metadata,
                                     "file":os.path.join(ref_dataset_path,ref_files[knn[j]])
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
    arg_parser.add_argument("reference_path", help="path to the physical location of the reference dataset")
    arg_parser.add_argument("--encoder", default="eigenplaces",
                            help="supported models: netvlad, cosplace, eigenplaces")
    arg_parser.add_argument("--device", default="cuda:0",
                            help="device identifier")
    arg_parser.add_argument("--sim_metric", default="cosine_sim")
    arg_parser.add_argument("--k", default=3, type=int)
    args = arg_parser.parse_args()

    encoder_options = options[args.encoder]
    localize_dataset(args.dataset_path, args.reference_path,
            args.encoder, encoder_options,
            args.device, args.k, args.sim_metric)


