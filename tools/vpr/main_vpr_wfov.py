import argparse
from encode_dataset import encode_dataset
from options import options
from localize_shelef_tile_with_hints import localize_dataset
# example arguments
'''
/mnt/artifacts/assets/vpr/aza_tiff_small
/mnt/rawdata/aza/DSM/dsm_trex_w84utm36_12m.tif
isd/
shelef_toy
'''

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("orthophoto_path",
                            help="path to full orthophoto dataset")
    arg_parser.add_argument("dsm_file",
                            help="path to the dsm file")
    arg_parser.add_argument("isd_path", help="a path to a dataset with isd files")
    arg_parser.add_argument("shelef_path", help="path to folder with shelef images to filter isd")
    arg_parser.add_argument("reference_file", help="path to the physical location of the reference encoding")
    arg_parser.add_argument("--sim_metric", default="cosine_sim")
    arg_parser.add_argument("--k", default=3, type=int)
    arg_parser.add_argument("--online_only", action='store_true')
    arg_parser.add_argument("--device", default="cuda:0",
                            help="device identifier")
    arg_parser.add_argument("--encoder", default="eigenplaces",
                            help="supported models: netvlad, cosplace, eigenplaces")

    args = arg_parser.parse_args()
    encoder_options = options[args.encoder]
    if not args.online_only:
        encode_dataset(args.output_path, args.output_path,
                       args.encoder, encoder_options,
                       True, args.device)

    localize_dataset(args.orthophoto_path, args.dsm_file, args.isd_path,
                     args.shelef_path, args.reference_file,
                     args.encoder, encoder_options,
                     args.device, args.k, args.sim_metric
                     )