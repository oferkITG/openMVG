import argparse
import torch
from utils.vpr_utils import get_nearest_neighbors
from loaders.VPRDataset import VPRDataset
from models.NetVLAD import NetVLAD
from options import options
import numpy as np
'''
example command:
data/assets/vpr/dataset1/Shelef_AAA_0504_511 output/dataset1/dataset1_Tiles_netvlad_encoding.pth --encoder netvlad
data/assets/vpr/dataset1/Shelef_AAA_0504_511 output/dataset1/dataset1_Tiles_cosplace_encoding.pth --encoder cosplace
data/assets/vpr/dataset1/Shelef_AAA_0504_511 output/dataset1/dataset1_Tiles_eigenplaces_encoding.pth --encoder eigenplaces
'''
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset_path", help="path to the physical location of the traget dataset")
    arg_parser.add_argument("reference_path", help="a .pth file with encoding and meta data")
    arg_parser.add_argument("--encoder", default="eigenplaces",
                            help="supported models: netvlad, cosplace, eigenplaces")
    arg_parser.add_argument("--device", default="cuda:0",
                            help="device identifier")
    arg_parser.add_argument("--sim_metric", default="cosine_sim")
    arg_parser.add_argument("--k", default=3, type=int)
    args = arg_parser.parse_args()
    if torch.cuda.is_available():
        device_id = args.device
    else:
        device_id = 'cpu'
    device = torch.device(device_id)

    encoder_type = args.encoder
    print("Localizing a shelef dataset with {}".format(encoder_type))
    encoder_options = options[encoder_type]
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
    dataset = VPRDataset(args.dataset_path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, **loader_params)

    print("Load the encoding of the reference dataset from: {}".format(args.reference_path))
    ref = torch.load(args.reference_path)
    ref_encoding = ref["encoding"].to(device)
    metadata = ref["metadata"]
    ref_ids = ref["img_ids"]
    i = 0
    k = args.k
    mapk = {1:[],2:[],3:[]}
    top_scores = np.zeros((k, len(dataset)))
    with torch.no_grad():
        net.eval()
        net.to(device)
        for i, minibatch in enumerate(loader):
            imgs = minibatch.get('img').to(device)
            global_desc = net(imgs)
            knn, scores = get_nearest_neighbors(global_desc, ref_encoding, k,
                                                metric=args.sim_metric, return_scores=True)

            scores = scores.cpu().numpy()
            query_id = loader.dataset.img_ids[i]
            query_metadata = loader.dataset.metadata[query_id]
            matching_tiles = query_metadata.get("list_of_tiles")
            print("#"*10)
            print("Query: {}, list of tiles:{}".format(query_id, matching_tiles))
            found_match = False
            matches = np.zeros(k)
            for j in range(k):
                top_scores[j,i] = scores[j]
                nn_id = ref_ids[knn[j]]
                print("neighbor #{} [{:.4f}] - id: {}".format(j+1, scores[j], nn_id))
                if len(matching_tiles) > 0:
                    if nn_id in matching_tiles:
                        matches[j] = 1

            if len(matching_tiles) > 0:
                for j in range(k):
                    if np.sum(matches[:(j+1)]) > 0:
                        mapk[j+1].append(1)
                    else:
                        mapk[j+1].append(0)

            print("#"*10)

    for k, matches in mapk.items():
        m = np.sum(np.array(matches))
        n = len(matches)
        p = m*1.0/n
        print("{}/{}(MAP-{}:{:.2f}) found a match in the top-{} results".format(m, n, k, p , k))
        print("Mean score of matches in place {}: {:.3f}".format(k, np.mean(top_scores[k-1])))

