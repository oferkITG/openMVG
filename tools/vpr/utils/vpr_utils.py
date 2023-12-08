from torchvision import transforms
import torch
import numpy as np
import logging


class ToRGBTensor(object):
    def __call__(self, img):
        img = np.moveaxis(np.array(img), [0,1,2], [1, 2, 0])
        img = torch.from_numpy(img).to(torch.float32)
        return img

    def __repr__(self):
        return 'vpr.ToRGBTensor'

def netvlad_transform(img_size, augmentations=[]):
    """
        Composes a data transformation for embedding
        Note: this will generate an RGB range image (not [0,1]) to optimize netvlad performance
        :param img_size: (int or None) if integer is given the image is resized to (img_size, img_size)
        :param: augmentations: (list<transformations>) a list of augmentation transformations to apply
        :return: the transformation to apply
    """
    if img_size is None:
        resize_transform = []
    else:
        resize_transform = [transforms.Resize((img_size, img_size))]
    transforms_list = [transforms.ToPILImage(), *resize_transform, *augmentations,
                       ToRGBTensor(), transforms.Normalize(
        mean=[123.68, 116.779, 103.939], std=[1.0, 1.0, 1.0])]
    transform = transforms.Compose(transforms_list)
    return transform

def cosplace_transform(img_size):
    if img_size is None:
        resize_transform = []
    else:
        resize_transform = [transforms.Resize((img_size[0], img_size[1]))]
    transforms_list = [transforms.ToPILImage(), *resize_transform, transforms.ToTensor(),
                       transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    transform = transforms.Compose(transforms_list)
    return transform

def base_transform(img_size):
    transforms_list = [transforms.ToPILImage(),
                       transforms.Resize((img_size, img_size)),
                                         ToRGBTensor()]
    transform = transforms.Compose(transforms_list)
    return transform
def get_nearest_neighbors(query_vec, db_vec, k, metric='cosine_sim', return_scores=False):
    """
    Fetch the indices of the k nearest neighbors in the given dataset's embedding
    :param query_vec: (torch.tensor) an embedding of a query
    :param db_vec: (torch.tensor) an embedding of a dataset
    :param k: (int) the number of requested nearest-neighbors
    :param metric: (str) the metric to use for ranking the neighbors
                   (corrcoeff for Pearson's corr., l2 for L2-norm)
    :return: list of indices
    """
    if metric == 'l2':
        distances = torch.norm(db_vec - query_vec, dim=1)
        knn_indices = np.argsort(distances.cpu().numpy())[:k]
        knn_scores = -distances[knn_indices]
    elif metric == 'pearson_corr':
        db_vec = db_vec.cpu().numpy()
        query_vec = query_vec.cpu().numpy()
        n = db_vec.shape[0]
        correlations = np.zeros(n)
        for i in range(n):
            correlations[i] = np.corrcoef(query_vec[0], db_vec[i])[0,1]
        knn_indices = np.argsort(-correlations, )[:k]
        knn_scores = correlations[knn_indices]
    elif metric == 'normalized_corr':
        query_vec = query_vec.view(-1)
        query_vec = (query_vec - query_vec.mean())
        db_means = db_vec.mean(dim=1, keepdim=True)
        db_vec = db_vec - db_means
        norm_corr = (torch.matmul(db_vec, query_vec) /
                     (torch.norm(db_vec, dim=1) * torch.norm(query_vec))).cpu().numpy()
        knn_indices = np.argsort(-norm_corr)[:k]
        knn_scores = norm_corr[knn_indices]
    elif metric == 'cosine_sim':
        similarities = (torch.matmul(db_vec, query_vec.transpose(0,1)).view(-1)) / \
                       ((torch.norm(db_vec, dim=1))*torch.norm(query_vec))
        knn_indices = np.argsort(-similarities.cpu().numpy())[:k]
        knn_scores = similarities[knn_indices]
    else:
        logging("metric {} not supported".format(metric))
        raise NotImplementedError
    if return_scores:
        return knn_indices, knn_scores
    else:
        return knn_indices


def get_similarities(query_vec, db_vec, metric='cosine_sim', return_scores=False):
    if metric == 'l2':
        distances = torch.norm(db_vec - query_vec, dim=1)
        distances = distances.cpu().numpy()
        sorted_inds = np.argsort(distances)
        sorted_dist = -distances[sorted_inds]
    elif metric == 'pearson_corr':
        db_vec = db_vec.cpu().numpy()
        query_vec = query_vec.cpu().numpy()
        n = db_vec.shape[0]
        correlations = np.zeros(n)
        for i in range(n):
            correlations[i] = np.corrcoef(query_vec[0], db_vec[i])[0,1]
        sorted_inds = np.argsort(-correlations, )
        sorted_dist = correlations[sorted_inds]
    elif metric == 'normalized_corr':
        query_vec = query_vec.view(-1)
        query_vec = (query_vec - query_vec.mean())
        db_means = db_vec.mean(dim=1, keepdim=True)
        db_vec = db_vec - db_means
        norm_corr = (torch.matmul(db_vec, query_vec) /
                     (torch.norm(db_vec, dim=1) * torch.norm(query_vec))).cpu().numpy()
        sorted_inds = np.argsort(-norm_corr)
        sorted_dist = norm_corr[sorted_inds]
    elif metric == 'cosine_sim':
        similarities = (torch.matmul(db_vec, query_vec.transpose(0,1)).view(-1)) / \
                       ((torch.norm(db_vec, dim=1))*torch.norm(query_vec))
        sorted_inds = np.argsort(-similarities.cpu().numpy())
        sorted_dist = similarities[sorted_inds]
    else:
        logging("metric {} not supported".format(metric))
        raise NotImplementedError
    if return_scores:
        return sorted_inds, sorted_dist
    else:
        return sorted_inds

