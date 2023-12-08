from tools.vpr.utils.vpr_utils import netvlad_transform, cosplace_transform
options = {
    "netvlad":
        {
            "weights":"weights/pretrained_vgg16_pitts30k_netvlad_from_matlab.pth",
            "transforms":
                {
                    "shelef_transform": netvlad_transform(2000),
                    "orthophoto_transform": netvlad_transform(2000),
                },
            "dim":4096

        },
    "cosplace":
        {
            "backbone":"ResNet50",
            "dim":2048,
            "transforms":
                {
            "shelef_transform": cosplace_transform(None),
            "orthophoto_transform":cosplace_transform(None),
                }

        },
    "cosplace_small_res":
        {
            "backbone": "ResNet50",
            "dim": 2048,
            "transforms":
                {
                    "shelef_transform": cosplace_transform([480, 640]),
                    "orthophoto_transform": cosplace_transform(None),
                }

        },
    "eigenplaces":
        {
            "backbone":"ResNet50",
            "dim":2048,
            "transforms":
                {
                    "shelef_transform": cosplace_transform(None), #cosplace_transform(1000),
                    "orthophoto_transform": cosplace_transform(None)#cosplace_transform(2000),
                }

        }


}