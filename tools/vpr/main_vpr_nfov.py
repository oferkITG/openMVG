import argparse
from generate_synt_views import generate_views
from encode_dataset import encode_dataset
from options import options
from localize_shelef_dataset import localize_dataset
# example arguments
'''
/mnt/artifacts/assets/vpr/aza_tiff_small
/mnt/rawdata/aza/DSM/dsm_trex_w84utm36_12m.tif
isd/
/mnt/artifacts/assets/vpr/output/b1
shelef_b1
'''

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("orthophoto_path",
                            help="path to full orthophoto dataset")
    arg_parser.add_argument("dsm_file",
                            help="path to the dsm file")
    arg_parser.add_argument("isd_path", help="a path to a dataset with isd files")
    arg_parser.add_argument("output_path", help="output path")
    arg_parser.add_argument("shelef_path", help="path to folder with shelef images to filter isd")
    arg_parser.add_argument("--max_shift_east", type=int, default=1500)
    arg_parser.add_argument("--shift_step_east", type=int, default=500)
    arg_parser.add_argument("--max_shift_north", type=int, default=150)
    arg_parser.add_argument("--shift_step_north", type=int, default=150)
    arg_parser.add_argument("--resize_model_scale", type=int, default=4)
    arg_parser.add_argument("--resize_orthophoto_scale", type=float, default=0.25)
    arg_parser.add_argument("--fov_factor", type=float, default=1)
    arg_parser.add_argument("--device", default="cuda:0",
                            help="device identifier")
    arg_parser.add_argument("--encoder", default="cosplace",
                            help="supported models: netvlad, cosplace, eigenplaces")
    arg_parser.add_argument("--single_folder", action='store_true')
    arg_parser.add_argument("--online_only", action='store_true')
    arg_parser.add_argument("--force_encode", action='store_true')
    arg_parser.add_argument("--sim_metric", default="cosine_sim")
    arg_parser.add_argument("--k", default=3, type=int)

    args = arg_parser.parse_args()
    encoder_options = options[args.encoder]
    if not args.online_only:

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
                       False
                       )


        encode_dataset(args.output_path, args.output_path,
                           args.encoder, encoder_options,
                           False, args.device)
    elif args.force_encode:
        encode_dataset(args.output_path, args.output_path,
                       args.encoder, encoder_options,
                       False, args.device)

    localize_dataset(args.shelef_path, args.output_path,
                     args.encoder, encoder_options,
                     args.device, args.k, args.sim_metric)