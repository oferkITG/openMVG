import os
import torch
import sys
sys.path.append("../")
from utils.vpr_utils import get_nearest_neighbors
import numpy as np

output_path = "../output/Aza/"
data_ids = [f for f in os.listdir((output_path))]
img_ids = []
cluster_ids = []
encoding = []
for did in data_ids:
    data_path = os.path.join(output_path, did)
    data = torch.load(data_path)
    encoding += data["encoding"]
    img_files = data["img_files"]
    img_ids += data["img_ids"]
    cluster_ids += [did.replace(".pth", "") for i in range(len(data["img_ids"]))]

encoding = torch.stack(encoding)
print(encoding.shape)
k = 3
matched = np.zeros(len(img_ids))
for i, img_id in enumerate(img_ids):
    print("#" * 10)
    print("query id: {} - cluster id: {}".format(img_id, cluster_ids[i]))
    query = encoding[i].unsqueeze(0)
    knn, scores = get_nearest_neighbors(query, encoding, encoding.shape[0],
                                        metric='cosine_sim', return_scores=True)
    knn = knn[1:] # remove self
    scores = scores[1:] # remove self
    for j, l in enumerate(knn[:k]):
        nn_id = img_ids[l]
        nn_cluster_id = cluster_ids[l]
        if nn_cluster_id == cluster_ids[i]:
            matched[i] = 1


        print("neighbor #{} [{:.3f}] - id: {}, cluster_id: {}".format(j, scores[j],
                                                                   nn_id, nn_cluster_id))
    print("#"*10)

print("Statistics for {} images within {} clusters".format(len(img_ids), len(np.unique(np.array(cluster_ids)))))
print("{} matched to cluster within {} neighbors".format(np.mean(matched),k))



