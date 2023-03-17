import numpy as np
import torch
import pandas as pd
from sklearn.cluster import KMeans
from .BT import get_weights, get_sampled_indices
from .generalized_BT import get_data_distribution, manipulate_data_distribution

def full_label_data(df, tasks):
    """filter the instances with all required labels

    Args:
        df (pd.DataFrame): a DataFrame containing data instances
        tasks (list): a list of names of target columns

    Returns:
        np.array: an array of boolean values indicating whether or not each row meets the requirement.
    """
    selected_rows = np.array([True]*len(df))
    for task in tasks:
        selected_rows = selected_rows & df[task].notnull().to_numpy()
    return selected_rows

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split

        self.X = []
        self.y = []
        self.protected_label = []
        self.instance_weights = []
        self.adv_instance_weights = []
        self.regression_label = []
        self.mask = None

        self.load_data()

        self.regression_init()
        
        self.X = np.array(self.X)
        if self.mask is not None:
            self.mask = np.array(self.mask)
            
        if len(self.X.shape) == 3:
            self.X = np.concatenate(list(self.X), axis=0)
        self.y = np.array(self.y).astype(int)
        self.protected_label = np.array(self.protected_label).astype(int)

        self.subsample_data()
        self.remove_clusters()

        self.manipulate_data_distribution()

        self.balanced_training()

        self.adv_balanced_training()

        if self.split == "train":
            self.adv_decoupling()

        print("Loaded data shapes: {}, {}, {}".format(self.X.shape, self.y.shape, self.protected_label.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.mask is not None:
            return self.X[index], self.y[index], self.protected_label[index], self.instance_weights[index], self.adv_instance_weights[index], self.regression_label[index], self.mask[index]
        return self.X[index], self.y[index], self.protected_label[index], self.instance_weights[index], self.adv_instance_weights[index], self.regression_label[index]
    
    def load_data(self):
        pass

    def subsample_data(self):
        'Subsamples train split'
        if self.split == "train" and self.args.subsample_perc < 1.0:
            subsample_len = int(len(self.X) * self.args.subsample_perc)
            self.X = self.X[:subsample_len]
            self.y = self.y[:subsample_len]
            self.protected_label = self.protected_label[:subsample_len]
            self.regression_label = self.regression_label[:subsample_len]
            if self.mask is not None:
                self.mask = self.mask[:subsample_len]

    def remove_clusters(self):
        'Cluster train set and remove several clusters'
        if self.split == "train" and self.args.num_remove_clusters > 0:
            assert self.args.num_remove_clusters < self.args.num_clusters, "Number of clusters should be greater than number of clusters to remove"
            cluster_alg = KMeans(n_clusters=self.args.num_clusters)
            clusters = cluster_alg.fit_predict(self.X)
            clusters_to_remove = np.random.choice(np.arange(self.args.num_clusters),
                                                  self.args.num_remove_clusters,
                                                  replace=False)
            for cluster in clusters_to_remove:
                clusters = np.where(clusters != cluster, clusters, -100)
            clusters = np.where(clusters == -100, False, True)
            self.X = self.X[clusters]
            self.y = self.y[clusters]
            self.protected_label = self.protected_label[clusters]
            self.regression_label = self.regression_label[clusters]
            if self.mask is not None:
                self.mask = self.mask[clusters]

    def remove_similar(self, another_dataset, another_centroid=None):
        from scipy import spatial
        if another_centroid is None:
            another_centroid = np.mean(another_dataset.X, axis=0)
        scores = []
        for el in self.X:
            scores += [1 - spatial.distance.cosine(el, another_centroid)]
        print(np.min(scores), np.max(scores), np.mean(scores), np.median(scores))
        top_scores = np.sort(scores)[:int((1 - self.args.remove_percent) * len(scores))]
        print(np.min(top_scores), np.max(top_scores), np.mean(top_scores), np.median(top_scores))
        top_score_indices = np.argsort(scores)[:int((1 - self.args.remove_percent) * len(scores))]
        self.X = self.X[top_score_indices]
        self.y = self.y[top_score_indices]
        self.protected_label = self.protected_label[top_score_indices]
        self.regression_label = self.regression_label[top_score_indices]
        if self.mask is not None:
            self.mask = self.mask[top_score_indices]

    def cluster_and_remove(self, another_dataset):
        # Cluster another dataset, choose random cluster and remove most similar to this cluster samples
        cluster_alg = KMeans(n_clusters=self.args.num_clusters)
        clusters = cluster_alg.fit_predict(another_dataset.X)
        clusters_to_remove = np.random.choice(np.arange(self.args.num_clusters),
                                              1, replace=False)
        another_centroid = cluster_alg.cluster_centers_[clusters_to_remove]
        self.remove_similar(another_dataset, another_centroid)

    def manipulate_data_distribution(self):
        if self.args.GBT and self.split == "train":
            # Get data distribution
            distribution_dict = get_data_distribution(y_data=self.y, g_data=self.protected_label)

            selected_index = manipulate_data_distribution(
                default_distribution_dict = distribution_dict, 
                N = self.args.GBT_N, 
                GBTObj = self.args.GBTObj, 
                alpha = self.args.GBT_alpha)

            self.X = self.X[selected_index]
            self.y = self.y[selected_index]
            self.protected_label = self.protected_label[selected_index]

    def balanced_training(self):
        if (self.args.BT is None) or (self.split != "train"):
            # Without balanced training
            self.instance_weights = np.array([1 for _ in range(len(self.protected_label))])
        else:
            assert self.args.BT in ["Reweighting", "Resampling", "Downsampling"], "not implemented"

            assert self.args.BTObj in ["joint", "y", "g", "stratified_y", "stratified_g", "EO"], "not implemented"
            """
            reweighting each training instance 
                joint:          y,g combination, p(g,y)
                y:              main task label y only, p(y)
                g:              protected label g only, p(g)
                stratified_y:   balancing the g for each y, p(g|y), while keeping the y distribution
                stratified_g:   balancing the y for each g, p(y|g)
                EO:             balancing the g for each y, p(g|y)
            """

            if self.args.BT == "Reweighting":
                self.instance_weights = get_weights(self.args.BTObj, self.y, self.protected_label)

            elif self.args.BT in ["Resampling", "Downsampling"]:

                selected_index = get_sampled_indices(self.args.BTObj, self.y, self.protected_label, method = self.args.BT)

                X = [self.X[index] for index in selected_index]
                self.X = np.array(X)
                y = [self.y[index] for index in selected_index]
                self.y = np.array(y)
                _protected_label = [self.protected_label[index] for index in selected_index]
                self.protected_label = np.array(_protected_label)
                self.instance_weights = np.array([1 for _ in range(len(self.protected_label))])

            else:
                raise NotImplementedError
        return None

    def adv_balanced_training(self):
        if (self.args.adv_BT is None) or (self.split != "train"):
            # Without balanced training
            self.adv_instance_weights = np.array([1 for _ in range(len(self.protected_label))])
        else:
            assert self.args.adv_BT in ["Reweighting"], "not implemented"

            assert self.args.adv_BTObj in ["joint", "y", "g", "stratified_y", "stratified_g", "EO"], "not implemented"
            """
            reweighting each training instance 
                joint:          y,g combination, p(g,y)
                y:              main task label y only, p(y)
                g:              protected label g only, p(g)
                stratified_y:   balancing the g for each y, p(g|y)
                stratified_g:   balancing the y for each g, p(y|g)
            """

            if self.args.adv_BT == "Reweighting":
                self.adv_instance_weights = get_weights(self.args.adv_BTObj, self.y, self.protected_label)
            else:
                raise NotImplementedError
        return None

    def adv_decoupling(self):
        """Simulating unlabelled protected labels through assigning -1 to instances.

        Returns:
            None
        """
        if self.args.adv_decoupling and self.args.adv_decoupling_labelled_proportion < 1:
            self.adv_instance_weights[
                np.random.rand(len(self.protected_label)) > self.args.adv_decoupling_labelled_proportion
                ] = -1
        else:
            pass
        return None

    def regression_init(self):
        if not self.args.regression:
            self.regression_label = np.array([0 for _ in range(len(self.protected_label))])
        else:
            # Discretize variable into equal-sized buckets
            if self.split == "train":
                bin_labels, bins = pd.qcut(self.y, q=self.args.n_bins, labels=False, duplicates = "drop", retbins = True)
                self.args.regression_bins = bins
            else:
                bin_labels = pd.cut(self.y, bins=self.args.regression_bins, labels=False, duplicates = "drop", include_lowest = True)
            bin_labels = np.nan_to_num(bin_labels, nan=0)
            
            # Reassign labels
            self.regression_label, self.y = np.array(self.y), bin_labels