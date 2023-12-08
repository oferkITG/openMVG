import pickle
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

input_dataset = "shelef_toy/"
with open("shelef_toy/localization_results.pkl", 'rb') as f:
    res = pickle.load(f)

for query_id, loc in res.items():
    imgs_to_plot = []
    query_img = join(input_dataset, query_id+".JPG")
    print(query_img)
    imgs_to_plot.append(query_img)
    for i, match in loc.items():
        print("Neighbor {}".format(i+1))
        for nn_id, props in match.items():
            print("Id: {}".format(nn_id))
            for k, v in props.items():
                print("{}:{}".format(k, v))
                if k == "file":
                    imgs_to_plot.append(v)

    fig, axes = plt.subplots(1, 2)
    for j in range(2):
        img = mpimg.imread(imgs_to_plot[j])
        print(img.shape)
        #plt.tight_layout()
        axes[j].imshow(img)
        axes[j].set_axis_off()
        #axes[j].axes.set_visible(False)
    plt.show()
