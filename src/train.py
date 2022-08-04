import sys
import os
import time
import logging
import datetime
import pickle
from collections import defaultdict

from pytz import timezone
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from scipy.spatial.distance import cdist

from datasets import ImageFolderWithPaths

from utils import (
    sfl_defaults,
    log_single_show_classification_metric,
)


tz = timezone("EST")
logger = logging.getLogger(__name__)
streamHandler = logging.StreamHandler(sys.stdout)
logger.addHandler(streamHandler)
file_handler = logging.FileHandler("log.txt")
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)


def train_siamese_embedding(
    model,
    train_dataloader,
    eval_dataloader,
    device,
    criterion,
    optimizer,
    num_epochs,
    scheduler=None,
    model_name=None,
    save_dir=None,
    plot_save_dir=None,
    save_iters=1
):
    """Train Benchmark model, ResNet50
    Args:
        model (torch.nn object): Model to be train
        train_dataloader (torch dataloader object): dataloader
        eval_dataloader (torch dataloader object): dataloader
        device (torch device object)
        criterion (torch loss object): loss function,
        optimizer (torch optimizer object): optimizer
        num_epochs (int): epochs to train the model
        scheduler (torch Scheduler object): scheduler for model training
        model_name (str): name of model
        save_dir (str): path to weights
        plot_save_dir (str): path to training plots
        save_iters (int): save every n iterations
    Returns:
        model (torch.nn object): trained model
    """

    train_loss_list = []
    eval_loss_list = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()

        running_loss = 0.0
        num_samp = 0
        for i, (data, labels) in enumerate(train_dataloader, 0):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * output.size(0)
            num_samp += len(labels)
            if scheduler is not None:
                scheduler.step()

        train_loss = running_loss / num_samp
        end_time = time.time()

        if epoch % save_iters == 0:
            model.eval()
            num_samp_eval = 0
            running_loss_eval = 0.0
            for i, (data, labels) in enumerate(eval_dataloader, 0):
                data, labels = data.to(device), labels.to(device)
                with torch.set_grad_enabled(False):
                    output = model(data)
                    loss = criterion(output, labels)
                running_loss_eval += loss.item() * output.size(0)
                num_samp_eval += len(labels)
            eval_loss = running_loss / num_samp_eval
            current_eval_loss = eval_loss
            train_loss_list.append(train_loss)
            eval_loss_list.append(eval_loss)
            if len(eval_loss_list) == 1:
                min_eval_loss = current_eval_loss
            if (model_name is not None) and (save_dir is not None) and current_eval_loss <= np.min(eval_loss_list):
                timestamp = str(datetime.datetime.now(tz)).replace(" ", "_")
                timestamp = timestamp.replace(":", "_")
                timestamp = timestamp.replace(".", "_")
                weight_name = f"checkpoint_ep_{epoch}_" + timestamp + ".pt"
                model_dir = os.path.join(save_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)
                torch.save(model, os.path.join(model_dir, weight_name))
                torch.save(model, os.path.join(
                    model_dir, "checkpoint_latest.pt"))

            logger.info(
                f"Epoch: {epoch} Train Loss: {train_loss:.4f} Eval Loss: {eval_loss:.4f} Train Spent: {end_time-start_time}s"
            )

    if (model_name is not None) and (plot_save_dir is not None):
        plot_dir = os.path.join(plot_save_dir, model_name)
        os.makedirs(plot_dir, exist_ok=True)

        sfl_defaults()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title("Optimization Curve\n\n")
        ax.set_xlabel("\nEpochs")
        ax.set_ylabel("Train Loss")
        fig.tight_layout()
        ax.plot(
            [i * 5 for i in range(len(train_loss_list))], train_loss_list, label="Train"
        )
        ax.plot(
            [i * 5 for i in range(len(eval_loss_list))], eval_loss_list, label="Val"
        )
        ax.legend()
        plt.savefig(os.path.join(
            plot_dir, "train_loss_plot_" + timestamp + ".png"))
        plt.close()

    return model


def evaluate_image_retrieval_SiameseNetwork(
    model,
    device,
    eval_loader,
    reference_embedding_path,
    save_similarity_pair_histo=False,
    expected_show=None,
    thres=0.5,
    eval_as_classification=False,
    mrr_type="min",
):
    """Mean Reciprocal Rank for evaluating Siamese Model
    Args:
        model (torch.nn object): Siamese Model
        device (torch.device object): cpu/GPU
        eval_loader (torch dataloader object): evaluation frame dataloader
        reference_embedding_path (str): path to reference embeddings(.pkl)
        save_similarity_pair_histo (boolean): save pairwise similarity figures for all classes
        expected_show (str): expected showname, ex. Big Bang Theory
        thres (float): similarity threshold for defining a match
        eval_as_classification (boolean): if True, then outputs will be binary(match or not match)
        mrr_type (str): types of Mean Reciprocal Rank, values can be min, max, mean

    Returns:
        mrr (float): Mean Reciprocal Rank
    """

    logger.info("Started Evaluation")

    with open(reference_embedding_path, "rb") as f:
        label_emb_tup = pickle.load(f)

    query_label_list = label_emb_tup[0]
    query_filename_list = label_emb_tup[2]
    query_emb_list = label_emb_tup[1]
    num_class = len(set(query_label_list))

    if eval_as_classification:
        expected_ref_id_list = [
            i for i, lb in enumerate(query_filename_list) if lb.split("/")[-2] == expected_show
        ]

        query_emb_list = np.array(query_emb_list)[expected_ref_id_list]
        single_show_cls_gt_list = []
        single_show_cls_pred_list = []

    model.eval()
    gt_label_list = []
    top_ranking_list = []
    if save_similarity_pair_histo:
        dist_vec_list = []
        key_label_list = []
    for inputs, labels, paths in eval_loader:
        gt_label_list.extend(labels.numpy())
        key_inputs = inputs.to(device)
        key_labels = labels.to(device)
        key_paths = paths

        with torch.set_grad_enabled(False):
            key_emb = model(key_inputs)
        dist_vec = cdist(key_emb.cpu(), np.array(query_emb_list), "cosine")

        if eval_as_classification:
            if len(query_emb_list) == 0:
                # Edge case handling for no face detected shows
                continue
            else:
                cos_sim = 1 - dist_vec
                max_cos_sim = np.max(cos_sim, axis=1)

            sub_pred_list = [1 if cs > thres else 0 for cs in max_cos_sim]
            sub_gt_list = [
                1 if lb.split("/")[-2] == expected_show else 0 for lb in key_paths
            ]
            single_show_cls_gt_list.extend(sub_gt_list)
            single_show_cls_pred_list.extend(sub_pred_list)
            continue

        s_i = np.argsort(dist_vec, axis=1)
        if save_similarity_pair_histo:
            dist_vec_list.append(dist_vec)
            key_label_list.extend(key_labels.cpu().numpy())

        for i in range(len(s_i)):
            query_ranking = s_i[i]
            ref_id_by_rank = np.array(query_label_list)[query_ranking]
            gt = key_labels[i]
            rank_match_id = []
            for rank, ref_id in enumerate(ref_id_by_rank):
                if gt == ref_id:
                    rank_match_id.append(rank)
            if mrr_type == "min":
                top_rank = rank_match_id[0]
            elif mrr_type == "max":
                top_rank = rank_match_id[-1]
            elif mrr_type == "mean":
                top_rank = np.mean(rank_match_id)

            top_ranking_list.append(top_rank)

    if eval_as_classification:
        # Get all metrics
        # Edge case handling for no face detected shows
        if len(single_show_cls_pred_list) == 0:
            return 0, 0, 0, 0
        full_eval_metric = log_single_show_classification_metric(
            single_show_cls_gt_list,
            single_show_cls_pred_list,
            expected_show,
        )

        return full_eval_metric

    if save_similarity_pair_histo:
        pass

    float_rank = (np.array(top_ranking_list) + 1).astype(float)
    for cls_id in range(num_class):
        class_float_rank = float_rank[[lb == cls_id for lb in gt_label_list]]
        cls_mrr = np.sum(np.reciprocal(class_float_rank)) / \
            len(class_float_rank)
        logger.info(f"cls_mrr for class {cls_id}:{cls_mrr}")

    mrr = np.sum(np.reciprocal(float_rank)) / len(float_rank)
    logger.info(f"Mean Reciprocal Rank : {mrr}")

    return mrr


def inference_siamese_network(
    model,
    device,
    inf_loader,
    reference_embedding_path,
    expected_show,
    thres,
    face_inf=False
):
    """Inference with Siamese Model
    Args:
        model (torch.nn object): Siamese Model
        device (torch.device object): cpu/GPU
        inf_loader (torch dataloader object): inference frame dataloader
        reference_embedding_path (str): path to reference embeddings
        thres (float): threshold for a "match"
        face_inf (boolean): true is this function is used to do inference on faces
    Returns:
        final_pred_list (np.ndarray): array of frame predictions
        len_predict (int): number of matched samples, used for time tracking
    """
    pred_list = []

    with open(reference_embedding_path, "rb") as f:
        label_emb_tup = pickle.load(f)

    query_label_list = label_emb_tup[0]
    query_emb_list = label_emb_tup[1]
    query_filename_list = label_emb_tup[2]

    expected_ref_id_list = [
        i for i, lb in enumerate(query_filename_list) if lb.split("/")[-2] == expected_show
    ]
    ref_path = [query_filename_list[rid]for rid in expected_ref_id_list]
    expected_ref_emb = np.array(query_emb_list)[expected_ref_id_list]

    model.eval()

    single_show_cls_pred_list = []
    ref_min_dist_ind_list = []
    for inputs, labels, paths in inf_loader:
        key_inputs = inputs.to(device)
        key_labels = labels.to(device)

        with torch.set_grad_enabled(False):
            key_emb = model(key_inputs)

            dist_vec = cdist(key_emb.cpu(), expected_ref_emb, "cosine")
            cos_sim = 1 - dist_vec
            max_cos_sim = np.max(cos_sim, axis=1)
            ref_min_dist_ind_list.extend(np.argmax(cos_sim, axis=1))
            sub_pred_list = [1 if cs > thres else 0 for cs in max_cos_sim]
            single_show_cls_pred_list.extend(sub_pred_list)

    if face_inf:
        result_list = []
        for i, pred in enumerate(single_show_cls_pred_list):
            if pred == 1:
                from_frame_id = os.path.basename(
                    paths[i].split(".")[0].split("face")[-1])
                if "_" in from_frame_id:
                    from_frame_id = from_frame_id.split("_")[0]

                result_list.append(
                    (ref_path[ref_min_dist_ind_list[i]], int(from_frame_id)))

    else:
        result_list = [(ref_path[ref_min_dist_ind_list[i]], i)
                       for i, pred in enumerate(single_show_cls_pred_list) if pred == 1]
    len_predict = len(single_show_cls_pred_list)
    return result_list, len_predict


def evaluate_all_frames(
        batch_size,
        frame_path,
        saved_model,
        device,
        num_workers,
        seed_worker,
        generator,
        num_ref=5,
        plot_simlarity=False):
    """Mean Reciprocal Rank for evaluating Siamese Model
    Args:
        batch_size (int): number of samples in a batch
        frame_path (str): path to evaluation frame dir
        saved_model (torch.nn object): Siamese Model
        device (torch.device object): cpu/GPU
        num_workers (int): number of workers for torch dataloader
        seed_worker (fucntion):
        generator (torch generator object): generator for dataloader
        num_ref (int): number of references to save in ref_path_dict
        plot_simlarity (boolean): true if save similairy plots
    Returns:
        ref_path_dict (dict): a dict with showname as keys, intellidently selected frames as value
    """
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                0.229, 0.224, 0.225]),
        ]
    )
    train_set = ImageFolderWithPaths(root=frame_path, transform=transform)
    filename_tup_list = train_set.imgs

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, worker_init_fn=seed_worker, generator=generator)
    class_id_mapping = train_set.class_to_idx
    id_class_mapping = {}
    for k, v in class_id_mapping.items():
        id_class_mapping[v] = k

    model = saved_model
    model.to(device)
    model.eval()

    np_list = []
    label_list = []
    paths_list = []
    frame_dict = defaultdict(list)
    frame_path_dict = defaultdict(list)
    ref_path_dict = defaultdict(list)

    # get stacked enbeddings for each show in train
    for i, (inputs, labels, paths) in enumerate(tqdm(train_loader)):

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            key_emb = model(inputs)
            np_list.append(key_emb.cpu().numpy())
            label_list.extend(labels.cpu().numpy())
            paths_list.extend(paths)

    np_arr = np.vstack(np_list)
    for i in range(len(label_list)):
        frame_dict[label_list[i]].append(np_arr[i])
        frame_path_dict[label_list[i]].append(paths_list[i])

    for key, val in frame_dict.items():
        frame_dict[key] = np.vstack(val)

    for target in class_id_mapping.keys():
        logger.info(f"processing {target} reference frames")

        current = frame_dict[class_id_mapping[target]]
        dist_vec_self = cdist(current, current, 'cosine')
        upper_tri_ind = np.triu_indices(len(current), 1)
        dist_vec_self = dist_vec_self[upper_tri_ind]

        if plot_simlarity:
            fig, axs = plt.subplots(figsize=(7, 3))
            sns.kdeplot(dist_vec_self.reshape(-1),
                        label=f"{target}", color=plt.cm.tab20(i), ax=axs)
            plt.legend(loc=(1.04, 0))
            axs.set_xlabel('Pair Wise Similarity', fontsize=20)
            axs.set_ylabel('Density', fontsize=20)
            axs.set_title(f"{target}", fontsize=20)
            plt.tight_layout()
            os.makedirs("frame_case_study", exist_ok=True)
            plt.savefig(
                f"frame_case_study/{target}_frame_pairwise_sim.png", bbox_inches='tight')
        # generate reference dictionary
        # get other character embeddings, and self to other dist
        mean_dist_per_target_frame_list = []
        for i in range(len(frame_dict)):
            if i != class_id_mapping[target]:
                others = frame_dict[i]
                dist_vec = cdist(current, others, 'cosine')
                # mean dist between current target frames and the other show frames across all images in "this other show"
                mean_dist_per_target_frame_list.append(
                    np.mean(dist_vec, axis=1))
                if plot_simlarity:
                    sns.kdeplot(dist_vec.reshape(-1),
                                label=f"{id_class_mapping[i]}", color=plt.cm.tab20(i), ax=axs)
        final_mean_dist_per_target_frame = np.mean(
            mean_dist_per_target_frame_list, axis=0)
        max_mean_dist_frame_id = np.argsort(
            final_mean_dist_per_target_frame)[::-1][:num_ref]
        top_k_target_frames = [frame_path_dict[class_id_mapping[target]][fid]
                               for fid in max_mean_dist_frame_id]
        ref_path_dict[target] = top_k_target_frames

    return ref_path_dict
