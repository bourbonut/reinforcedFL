"""
Functions to partition data
"""

import warnings, random, pickle


def generate_IID_parties(dataset, k_nodes, path, **kwargs):
    """
    Generate IID data (random shuffle) for each node.

    Args:
        dataset (dict[str, VisionDataset]):
            "training" for training data and "test" for test data
            where data must be `VisionDataset`
        k_nodes (int): Number of node
        path (Path): Folder to save data for nodes
    """

    msg = "Training data and test data have not the same number of labels"
    assert dataset["training"].classes == dataset["test"].classes, msg
    size_train = len(dataset["training"])
    size_test = len(dataset["test"])
    num_labels = len(dataset["training"].classes)
    mtr, mte = (size_train // k_nodes, size_test // k_nodes)  # samples per node

    # Shuffle indices to select random samples
    train_indices = list(range(size_train))
    test_indices = list(range(size_test))
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    for i in range(k_nodes):
        rtr = size_train % k_nodes if i + 1 == k_nodes else 0
        rte = size_test % k_nodes if i + 1 == k_nodes else 0
        # Select indices for each node and get the data
        node_train_indices = train_indices[mtr * i : mtr * (i + 1) + rtr]
        node_test_indices = test_indices[mte * i : mte * (i + 1) + rte]

        # Generate data
        node_data = [[], [], [], []]
        c = 0
        for key in dataset:
            data = dataset[key]
            indices = node_train_indices if key == "training" else node_test_indices
            for sample, label in (data[i] for i in indices):
                node_data[c].append(sample)
                node_data[c + 1].append(label)
            c += 2

        # Now put it all in an npz
        name_file = "nodes-" + str(i + 1) + ".pkl"
        with open(path / name_file, "wb") as file:
            pickle.dump(node_data, file)


# def generate_non_IID_label_parties(
#     dataset, nb_per_node, node_folder, labels_per_client=1
# ):
#     """
#     Generate non IID data: each node receive samples from labels_per_client different labels.
#
#     Args:
#         nb_per_node (list[int]): Number of samples per client
#         node_folder (str): Folder to save nodes' data
#         labels_per_client (int): Number of labels per client's dataset
#     """
#
#     dataset_path = os.path.join("examples", "datasets")
#     if not os.path.exists(dataset_path):
#         os.makedirs(dataset_path)
#     # (x_train, y_train), (x_test, y_test) = load_mnist(download_dir=dataset_path)
#     x_train, y_train, x_test, y_test = (
#         dataset["x_train"],
#         dataset["y_train"],
#         dataset["x_test"],
#         dataset["y_test"],
#     )
#
#     num_train = np.shape(y_train)[0]
#     num_test = np.shape(y_test)[0]
#     unique_labels = np.unique(y_test)
#     num_labels = np.shape(unique_labels)[0]
#
#     if sum(nb_per_node) > num_train:
#         logger.info(
#             f"Warning the sum of samples per parties is superior to the total number of samples {sum(nb_per_node)} > {num_train}"
#         )
#         raise InvalidConfigurationException("Nb_per_node")
#
#     # Shuffle indices for each label separately
#     indices_per_label_train = {
#         l: np.where(y_train == l)[0] for l in list(unique_labels)
#     }
#     indices_per_label_test = {l: np.where(y_test == l)[0] for l in list(unique_labels)}
#     for l in list(unique_labels):
#         rng.shuffle(indices_per_label_train[l])
#         rng.shuffle(indices_per_label_test[l])
#
#     # Create nodes' datasets
#     for i in range(len(nb_per_node)):
#         # if len(unique_labels) < labels_per_client:
#         #     unique_labels = np.unique(y_train)
#         # Select labels_per_client number of labels randomly
#         rng.shuffle(unique_labels)
#         if len(unique_labels) <= labels_per_client:
#             node_labels = unique_labels
#         else:
#             node_labels = unique_labels[:labels_per_client]
#         # unique_labels = unique_labels[labels_per_client:]
#
#         random_integers = rng.integers(0, nb_per_node[i], labels_per_client)
#         while np.sum(random_integers) <= 0:
#             random_integers = rng.integers(0, nb_per_node[i], labels_per_client)
#
#         # Select indices for each node and get the data
#         node_train_indices = np.array([], dtype=int)
#         node_test_indices = np.array([], dtype=int)
#         for l in range(labels_per_client):
#             nbr_sambles_n = int(
#                 nb_per_node[i] * float(random_integers[l]) / np.sum(random_integers)
#             )
#
#             if nbr_sambles_n >= len(indices_per_label_train[node_labels[l]]):
#                 nbr_sambles_n = len(indices_per_label_train[node_labels[l]])
#                 unique_labels = unique_labels[unique_labels != node_labels[l]]
#
#                 new_rdm_int = (
#                     float(nbr_sambles_n * np.sum(random_integers)) / nb_per_node[l]
#                 )
#                 if len(random_integers) > l + 1:
#                     random_integers[l + 1] += random_integers[l] - new_rdm_int
#
#             node_train_indices = np.concatenate(
#                 (
#                     node_train_indices,
#                     indices_per_label_train[node_labels[l]][:nbr_sambles_n],
#                 )
#             )
#             indices_per_label_train[node_labels[l]] = indices_per_label_train[
#                 node_labels[l]
#             ][nbr_sambles_n:]
#
#             nbr_sambles_n_test = int(num_test * nbr_sambles_n / num_train)
#             if nbr_sambles_n_test >= len(indices_per_label_test[node_labels[l]]):
#                 nbr_sambles_n_test = len(indices_per_label_test[node_labels[l]])
#
#             node_test_indices = np.concatenate(
#                 (
#                     node_test_indices,
#                     indices_per_label_test[node_labels[l]][:nbr_sambles_n_test],
#                 )
#             )
#             indices_per_label_test[node_labels[l]] = indices_per_label_test[
#                 node_labels[l]
#             ][nbr_sambles_n_test:]
#
#         x_train_pi = x_train[node_train_indices]
#         y_train_pi = y_train[node_train_indices]
#         x_test_pi = x_test[node_test_indices]
#         y_test_pi = y_test[node_test_indices]
#
#         # Now put it all in an npz
#         name_file = "data_party" + str(i + 1) + ".npz"
#         name_file = os.path.join(node_folder, name_file)
#         np.savez(
#             name_file,
#             x_train=x_train_pi,
#             y_train=y_train_pi,
#             x_test=x_test_pi,
#             y_test=y_test_pi,
#         )
#
#         print_statistics(i, x_test_pi, x_train_pi, num_labels, y_train_pi)
#
#         logger.info(f"Finished! :) Data saved in {node_folder}")
#
#     generate_histograms(node_folder, num_labels)
#
#
# def generate_non_IID_swapped_label_clusters(
#     dataset, nb_per_node, node_folder, swapped_labels=[(1, 7), (4, 9), (3, 8), (2, 5)]
# ):
#     """
#     Generate non IID data: create len(swapped_labels) clusters in which two labels are swapped out.
#     Each node's data is shuffled from one of this cluster (at least one node per cluster)
#
#     Args:
#         nb_per_node (list[int]): Number of samples per client
#         node_folder (str): Folder to save nodes' data
#         swapped_labels list(tuple(int)): List of tuples defining for each cluster which 2 labels will be swapped out
#     """
#
#     dataset_path = os.path.join("examples", "datasets")
#     if not os.path.exists(dataset_path):
#         os.makedirs(dataset_path)
#     # (x_train, y_train), (x_test, y_test) = load_mnist(download_dir=dataset_path)
#     x_train, y_train, x_test, y_test = (
#         dataset["x_train"],
#         dataset["y_train"],
#         dataset["x_test"],
#         dataset["y_test"],
#     )
#
#     num_train = np.shape(y_train)[0]
#     # logger.info(num_train)
#     num_test = np.shape(y_test)[0]
#     # logger.info(num_test)
#     num_labels = np.shape(np.unique(y_test))[0]
#
#     # Shuffle indices to select random samples
#     train_indices = np.arange(num_train)
#     rng.shuffle(train_indices)
#     test_indices = np.arange(num_test)
#     rng.shuffle(test_indices)
#
#     if sum(nb_per_node) > num_train:
#         logger.info(
#             f"Warning the sum of samples per parties is superior to the total number of samples {sum(nb_per_node)} > {num_train}"
#         )
#         raise InvalidConfigurationException("Nb_per_node")
#
#     def verify_cluster(cluster_per_client):
#         objective_count = len(nb_per_node) // len(swapped_labels)
#         return_value = True
#         for c in range(len(swapped_labels)):
#             if (
#                 np.count_nonzero(cluster_per_client == c) >= objective_count
#                 and np.count_nonzero(cluster_per_client == c) <= objective_count + 1
#             ):
#                 return_value = return_value * True
#             else:
#                 return_value = return_value * False
#         return return_value
#
#     # Assign a cluster to each client (at least one client per cluster)
#     cluster_per_client = rng.choice(len(swapped_labels), len(nb_per_node))
#
#     while not verify_cluster(cluster_per_client):
#         cluster_per_client = rng.choice(len(swapped_labels), len(nb_per_node))
#
#     for i in range(len(nb_per_node)):
#         # Select indices for each node and get the data
#         node_train_indices = train_indices[: nb_per_node[i]]
#         train_indices = train_indices[nb_per_node[i] :]
#         num_node_test = int(num_test * nb_per_node[i] / num_train)
#         node_test_indices = test_indices[:num_node_test]
#         test_indices = test_indices[num_node_test:]
#
#         x_train_pi = x_train[node_train_indices]
#         y_train_pi = y_train[node_train_indices]
#         x_test_pi = x_test[node_test_indices]
#         y_test_pi = y_test[node_test_indices]
#
#         indices_train_to_swap = dict()
#         indices_test_to_swap = dict()
#
#         for l in range(len(swapped_labels[cluster_per_client[i]])):
#             indices_train_to_swap[
#                 int(swapped_labels[cluster_per_client[i]][1 - l])
#             ] = list(
#                 np.where(y_train_pi == swapped_labels[cluster_per_client[i]][l])[0]
#             )
#             indices_test_to_swap[
#                 int(swapped_labels[cluster_per_client[i]][1 - l])
#             ] = list(np.where(y_test_pi == swapped_labels[cluster_per_client[i]][l])[0])
#
#         for l, indices in indices_train_to_swap.items():
#             y_train_pi[indices] = l
#         for l, indices in indices_test_to_swap.items():
#             y_test_pi[indices] = l
#
#         logger.info(
#             f"Node {i+1} of cluster {cluster_per_client[i]}: Label exchanged {swapped_labels[cluster_per_client[i]]}"
#         )
#
#         # Now put it all in an npz
#         name_file = "data_party" + str(i + 1) + ".npz"
#         name_file = os.path.join(node_folder, name_file)
#         np.savez(
#             name_file,
#             x_train=x_train_pi,
#             y_train=y_train_pi,
#             x_test=x_test_pi,
#             y_test=y_test_pi,
#         )
#
#         print_statistics(i, x_test_pi, x_train_pi, num_labels, y_train_pi)
#
#         logger.info(f"Finished! :) Data saved in {node_folder}")
#
#     data_clusters = {
#         int(i + 1): int(cluster_per_client[i]) for i in range(len(nb_per_node))
#     }
#     cluster_file = os.path.join(node_folder, "data_clusters.yml")
#     with open(cluster_file, "w") as outfile:
#         yaml.dump(data_clusters, outfile)
#
#     generate_histograms(node_folder, num_labels)
#
#
# def generate_non_IID_clusters_PCA_Kmeans(
#     dataset, nb_per_node, node_folder, variance=0.98, num_clusters=10
# ):
#     """
#     Generate non IID data: generates num_clusters clusters using PCA (Principal Component Analysis)
#     and K-means. Sub-samples nodes'data from these clusters samples.
#
#     Args:
#     nb_per_node (list[int]): Number of samples per client
#     node_folder (str): Folder to save nodes' data
#     """
#
#     from sklearn.decomposition import PCA
#     from sklearn.cluster import KMeans
#
#     x_train, y_train, x_test, y_test = (
#         dataset["x_train"],
#         dataset["y_train"],
#         dataset["x_test"],
#         dataset["y_test"],
#     )
#
#     num_train = np.shape(y_train)[0]
#     num_test = np.shape(y_test)[0]
#     unique_labels = np.unique(y_test)
#     num_labels = np.shape(unique_labels)[0]
#
#     if sum(nb_per_node) > num_train:
#         logger.info(
#             f"Warning the sum of samples per parties is superior to the total number of samples {sum(nb_per_node)} > {num_train}"
#         )
#         raise InvalidConfigurationException("Nb_per_node")
#
#     # Create clusters based on PCA and K-means algorithms based on:
#     # https://towardsdatascience.com/k-means-and-pca-for-image-clustering-a-visual-analysis-8e10d4abba40
#     x_train_pca = x_train.reshape(-1, x_train.shape[1] * x_train.shape[2])
#     x_test_pca = x_test.reshape(-1, x_test.shape[1] * x_test.shape[2])
#     pca = PCA(variance)
#     pca.fit(x_train_pca)
#     x_train_pca = pca.transform(x_train_pca)
#     x_test_pca = pca.transform(x_test_pca)
#
#     logger.info(f"PCA outputs dimensions: {pca.n_components_}")
#
#     num_clusters = min(num_clusters, len(nb_per_node))
#     k_means = KMeans(n_clusters=num_clusters)
#     k_means.fit(x_train_pca)
#     train_samples_clusters = k_means.predict(x_train_pca)
#     test_samples_clusters = k_means.predict(x_test_pca)
#
#     # Shuffle indices for each cluster separately
#     unique_clusters = np.unique(train_samples_clusters)
#     indices_per_cluster_train = {
#         l: np.where(train_samples_clusters == l)[0] for l in list(unique_clusters)
#     }
#     indices_per_cluster_test = {
#         l: np.where(test_samples_clusters == l)[0] for l in list(unique_clusters)
#     }
#
#     generate_clusters_histograms(
#         node_folder, y_train, indices_per_cluster_train, num_labels
#     )
#
#     for l in list(unique_clusters):
#         rng.shuffle(indices_per_cluster_train[l])
#         rng.shuffle(indices_per_cluster_test[l])
#
#     # Assign a cluster to each client (at least one client per cluster)
#     cluster_per_client = rng.choice(num_clusters, len(nb_per_node))
#     while not len(np.unique(cluster_per_client)) == num_clusters:
#         cluster_per_client = rng.choice(num_clusters, len(nb_per_node))
#
#     # Create dataset for each node based on cluster
#     for i in range(len(nb_per_node)):
#         # Select indices for each node and get the data
#         node_train_indices = indices_per_cluster_train[cluster_per_client[i]][
#             : nb_per_node[i]
#         ]
#
#         c = 0
#         history_node_train_indices = []
#         m = len(node_train_indices)
#         i_m = cluster_per_client[i]
#         while (
#             len(node_train_indices) < nb_per_node[i]
#         ):  # if all samples of this cluster has been distributed
#             cluster_per_client[i] = c
#             node_train_indices = indices_per_cluster_train[cluster_per_client[i]][
#                 : nb_per_node[i]
#             ]
#             history_node_train_indices.append(node_train_indices)
#             if len(node_train_indices) > m:
#                 i_m = c
#             c += 1
#             if c >= num_clusters:
#                 node_train_indices = history_node_train_indices[i_m]
#                 break
#
#         indices_per_cluster_train[cluster_per_client[i]] = indices_per_cluster_train[
#             cluster_per_client[i]
#         ][nb_per_node[i] :]
#         num_node_test = int(num_test * nb_per_node[i] / num_train)
#         node_test_indices = indices_per_cluster_test[cluster_per_client[i]][
#             :num_node_test
#         ]
#         indices_per_cluster_test[cluster_per_client[i]] = indices_per_cluster_test[
#             cluster_per_client[i]
#         ][num_node_test:]
#
#         x_train_pi = x_train[node_train_indices]
#         y_train_pi = y_train[node_train_indices]
#         x_test_pi = x_test[node_test_indices]
#         y_test_pi = y_test[node_test_indices]
#
#         # Now put it all in an npz
#         name_file = "data_party" + str(i + 1) + ".npz"
#         name_file = os.path.join(node_folder, name_file)
#         np.savez(
#             name_file,
#             x_train=x_train_pi,
#             y_train=y_train_pi,
#             x_test=x_test_pi,
#             y_test=y_test_pi,
#         )
#
#         logger.info(f"Cluster {cluster_per_client[i]}")
#         print_statistics(i, x_test_pi, x_train_pi, num_labels, y_train_pi)
#         logger.info(f"Finished! :) Data saved in {node_folder}")
#
#     data_clusters = {
#         int(i + 1): int(cluster_per_client[i]) for i in range(len(nb_per_node))
#     }
#     cluster_file = os.path.join(node_folder, "data_clusters.yml")
#     with open(cluster_file, "w") as outfile:
#         yaml.dump(data_clusters, outfile)
#
#     generate_histograms(node_folder, num_labels)
#
#
# def bench_partition(
#     dataset, n_parties, node_folder, partition, dataset_name="mnist", beta=0.5
# ):
#     """
#     Folowing, partition strategies from paper:
#     Li, Q., Diao, Y., Chen, Q., & He, B. (2021). Federated Learning on Non-IID Data Silos: An Experimental Study. http://arxiv.org/abs/2102.02079
#     """
#     x_train, y_train, x_test, y_test = (
#         dataset["x_train"],
#         dataset["y_train"],
#         dataset["x_test"],
#         dataset["y_test"],
#     )
#
#     n_train = np.shape(y_train)[0]
#     n_test = np.shape(y_test)[0]
#     num_labels = np.unique(y_test).shape[0]
#
#     if partition == "homo":
#         idxs = rng.permutation(n_train)
#         idxs_test = rng.permutation(n_test)
#         batch_idxs = np.array_split(idxs, n_parties)
#         batch_idxs_test = np.array_split(idxs_test, n_parties)
#         net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
#         net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_parties)}
#
#     elif partition == "noniid-labeldir":
#         min_size = 0
#         min_require_size = 10
#         K = 10
#         if dataset_name in ("celeba", "covtype", "a9a", "rcv1", "SUSY"):
#             K = 2
#             # min_require_size = 100
#
#         # np.random.seed(2020)
#         net_dataidx_map = {}
#         net_dataidx_map_test = {}
#
#         while min_size < min_require_size:
#             idx_batch = [[] for _ in range(n_parties)]
#             idx_batch_test = [[] for _ in range(n_parties)]
#             for k in range(K):
#                 idx_k = np.where(y_train == k)[0]
#                 rng.shuffle(idx_k)
#
#                 idx_k_test = np.where(y_test == k)[0]
#                 rng.shuffle(idx_k_test)
#
#                 proportions = rng.dirichlet(np.repeat(beta, n_parties))
#                 ## Balance
#                 proportions_train = np.array(
#                     [
#                         p * (len(idx_j) < n_train / n_parties)
#                         for p, idx_j in zip(proportions, idx_batch)
#                     ]
#                 )
#                 proportions_train = proportions_train / proportions_train.sum()
#                 proportions_train = (np.cumsum(proportions_train) * len(idx_k)).astype(
#                     int
#                 )[:-1]
#                 idx_batch = [
#                     idx_j + idx.tolist()
#                     for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions_train))
#                 ]
#                 min_size = min([len(idx_j) for idx_j in idx_batch])
#
#                 proportions_test = np.array(
#                     [
#                         p * (len(idx_j) < n_test / n_parties)
#                         for p, idx_j in zip(proportions, idx_batch_test)
#                     ]
#                 )
#                 proportions_test = proportions_test / proportions_test.sum()
#                 proportions_test = (
#                     np.cumsum(proportions_test) * len(idx_k_test)
#                 ).astype(int)[:-1]
#                 idx_batch_test = [
#                     idx_j + idx.tolist()
#                     for idx_j, idx in zip(
#                         idx_batch_test, np.split(idx_k_test, proportions_test)
#                     )
#                 ]
#
#         for j in range(n_parties):
#             rng.shuffle(idx_batch[j])
#             net_dataidx_map[j] = idx_batch[j]
#             rng.shuffle(idx_batch_test[j])
#             net_dataidx_map_test[j] = idx_batch_test[j]
#
#     elif partition > "noniid-#label0" and partition <= "noniid-#label9":
#         num = eval(partition[13:])
#         if dataset_name in ("celeba", "covtype", "a9a", "rcv1", "SUSY"):
#             num = 1
#             K = 2
#         else:
#             K = 10
#         if num == 10:
#             net_dataidx_map = {
#                 i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)
#             }
#             net_dataidx_map_test = {
#                 i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)
#             }
#             for i in range(10):
#                 idx_k = np.where(y_train == i)[0]
#                 rng.shuffle(idx_k)
#                 idx_k_test = np.where(y_test == i)[0]
#                 rng.shuffle(idx_k_test)
#                 split = np.array_split(idx_k, n_parties)
#                 split_test = np.array_split(idx_k_test, n_parties)
#                 for j in range(n_parties):
#                     net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
#                     net_dataidx_map_test[j] = np.append(
#                         net_dataidx_map_test[j], split_test[j]
#                     )
#         else:
#             times = [0 for i in range(10)]
#             contain = []
#             for i in range(n_parties):
#                 current = [i % K]
#                 times[i % K] += 1
#                 j = 1
#                 while j < num:
#                     ind = rng.integers(0, K - 1)
#                     if ind not in current:
#                         j = j + 1
#                         current.append(ind)
#                         times[ind] += 1
#                 contain.append(current)
#             net_dataidx_map = {
#                 i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)
#             }
#             net_dataidx_map_test = {
#                 i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)
#             }
#             for i in range(K):
#                 if times[i] > 0:
#                     idx_k = np.where(y_train == i)[0]
#                     idx_k_test = np.where(y_test == i)[0]
#
#                     rng.shuffle(idx_k)
#                     rng.shuffle(idx_k_test)
#                     split = np.array_split(idx_k, times[i])
#                     split_test = np.array_split(idx_k_test, times[i])
#                     ids = 0
#                     for j in range(n_parties):
#                         if i in contain[j]:
#                             net_dataidx_map[j] = np.append(
#                                 net_dataidx_map[j], split[ids]
#                             )
#                             net_dataidx_map_test[j] = np.append(
#                                 net_dataidx_map_test[j], split_test[ids]
#                             )
#                             ids += 1
#
#     elif partition == "iid-diff-quantity":
#         idxs = rng.permutation(n_train)
#         idxs_test = rng.permutation(n_test)
#         min_size = 0
#         while min_size < 10:
#             proportions = rng.dirichlet(np.repeat(beta, n_parties))
#             proportions = proportions / proportions.sum()
#             min_size = np.min(proportions * len(idxs))
#         proportions_train = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
#         proportions_test = (np.cumsum(proportions) * len(idxs_test)).astype(int)[:-1]
#         batch_idxs = np.split(idxs, proportions_train)
#         batch_idxs_test = np.split(idxs_test, proportions_test)
#         net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
#         net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_parties)}
#
#     # Now put it all in an npz
#     for i in net_dataidx_map.keys():
#         x_train_pi = x_train[net_dataidx_map[i]]
#         y_train_pi = y_train[net_dataidx_map[i]]
#         x_test_pi = x_test[net_dataidx_map_test[i]]
#         y_test_pi = y_test[net_dataidx_map_test[i]]
#
#         name_file = "data_party" + str(i + 1) + ".npz"
#         name_file = os.path.join(node_folder, name_file)
#         np.savez(
#             name_file,
#             x_train=x_train_pi,
#             y_train=y_train_pi,
#             x_test=x_test_pi,
#             y_test=y_test_pi,
#         )
