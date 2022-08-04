# -*- coding: utf-8 -*-
import os
import glob
import sys
import time
import shutil
import json
import ast
import logging
import datetime
from pytz import timezone
import toml
import random
import warnings

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from facenet_pytorch import (
    InceptionResnetV1,
)

from datasets import (
    BalancedBatchSampler,
    HardNegativePairSelector,
    seed_worker,
    ImageFolderWithPaths
)
from losses import OnlineContrastiveLoss
from preprocess import preprocess
from train import (
    inference_siamese_network,
    train_siamese_embedding,
)

# ------------------------------------------------------------------------------#
#                              GENERAL SETTINGS                                 #
# ------------------------------------------------------------------------------#
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
warnings.filterwarnings("ignore")
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
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"using device {device}")
config = toml.load("config/config.toml")
# ------------------------------------------------------------------------------#
#                                 PARAMETERS                                   #
# ------------------------------------------------------------------------------#

TASK = config.get("TASK").get("TASK")
EXPECTED_SHOW = config.get("TASK").get("EXPECTED_SHOW")
# Data parameters
VID_INFO = config.get("DATA").get("VID_INFO")
RAW_VIDEO_DIR = config.get("DATA").get("RAW_VIDEO_DIR")
PROCESSED_VIDEO_DIR = config.get("DATA").get("PROCESSED_VIDEO_DIR")
VIDEO_DIR_TRAIN = config.get("DATA").get("VIDEO_DIR_TRAIN")
VIDEO_DIR_EVAL = config.get("DATA").get("VIDEO_DIR_EVAL")
FRAME_DIR_TRAIN = config.get("DATA").get("FRAME_DIR_TRAIN")
FRAME_DIR_EVAL = config.get("DATA").get("FRAME_DIR_EVAL")
FACE_DIR_TRAIN = config.get("DATA").get("FACE_DIR_TRAIN")
FACE_DIR_EVAL = config.get("DATA").get("FACE_DIR_EVAL")
AUDIO_DIR_TRAIN = config.get("DATA").get("AUDIO_DIR_TRAIN")
AUDIO_DIR_EVAL = config.get("DATA").get("AUDIO_DIR_EVAL")
CUSTOM_FACE_DIR_TRAIN = config.get("DATA").get("CUSTOM_FACE_DIR_TRAIN")
CUSTOM_FACE_DIR_EVAL = config.get("DATA").get("CUSTOM_FACE_DIR_EVAL")
# Preprocessing parameters
RECREATE_VID_SPLIT = ast.literal_eval(
    config.get("PREPROCESS").get("RECREATE_VID_SPLIT")
)
RECREATE_FRAMES = ast.literal_eval(
    config.get("PREPROCESS").get("RECREATE_FRAMES"))
RECREATE_REFERENCE_FRAME = ast.literal_eval(
    config.get("PREPROCESS").get("RECREATE_REFERENCE_FRAME")
)
RECREATE_REFERENCE_NAMES = ast.literal_eval(
    config.get("PREPROCESS").get("RECREATE_REFERENCE_NAMES")
)
REFERENCE_FRAME_PATH = config.get("PREPROCESS").get("REFERENCE_FRAME_PATH")
REFERENCE_FRAME_EMBEDDING_PATH = config.get("PREPROCESS").get(
    "REFERENCE_FRAME_EMBEDDING_PATH"
)
REFERENCE_FACE_PATH = config.get("PREPROCESS").get("REFERENCE_FACE_PATH")
REFERENCE_FACE_EMBEDDING_PATH = config.get("PREPROCESS").get(
    "REFERENCE_FACE_EMBEDDING_PATH"
)
REFERENCE_NAME_DICT_PATH = config.get(
    "PREPROCESS").get("REFERENCE_NAME_DICT_PATH")
REFERENCE_PROB_DICT_PATH = config.get(
    "PREPROCESS").get("REFERENCE_PROB_DICT_PATH")
N_SECS_TRAIN = config.get("PREPROCESS").get("N_SECS_TRAIN")
N_SECS_EVAL = config.get("PREPROCESS").get("N_SECS_EVAL")
N_SECS_INF = config.get("PREPROCESS").get("N_SECS_INF")
FACE_ONLY = ast.literal_eval(config.get("PREPROCESS").get("FACE_ONLY"))
FACE_CROP = ast.literal_eval(config.get("PREPROCESS").get("FACE_CROP"))
NUM_REF = int(config.get("PREPROCESS").get("NUM_REF"))

# Model parameters
MODEL_NAME = config.get("MODEL").get("MODEL_NAME")
MODEL_DIR = config.get("MODEL").get("MODEL_DIR")
SAVED_MODEL = config.get("MODEL").get("WEIGHTS")
SAVED_FACE_MODEL = config.get("MODEL").get("SAVED_FACE_MODEL")
MODEL_LAST_LAYER_LOGITS = int(config.get(
    "MODEL").get("MODEL_LAST_LAYER_LOGITS"))
# Traning parameters
RETRAIN_BACKBONE = ast.literal_eval(
    config.get("TRAIN").get("RETRAIN_BACKBONE"))
BATCH_SIZE = config.get("TRAIN").get("BATCH_SIZE")
TRAIN_EPOCHS = config.get("TRAIN").get("TRAIN_EPOCHS")
LR = config.get("TRAIN").get("LR")
LR_FACE = config.get("TRAIN").get("LR_FACE")
GAMMA_FACE = config.get("TRAIN").get("GAMMA_FACE")
SAVE_ITER_FACE = config.get("TRAIN").get("SAVE_ITER_FACE")
DECAY_STEP = float(config.get("TRAIN").get("DECAY_STEP"))
DECAY_RATE = float(config.get("TRAIN").get("DECAY_RATE"))
SAMPLING_MARGIN = float(config.get("TRAIN").get("SAMPLING_MARGIN"))
BF_THRES = config.get("TRAIN").get("BF_THRES")
PLOT_DIR = config.get("TRAIN").get("PLOT_DIR")
SCHEDULER_FACE = config.get("TRAIN").get("SCHEDULER_FACE")
NUM_WORKERS = config.get("TRAIN").get("NUM_WORKERS")
NEG_SAMPLING_SIZE = config.get("TRAIN").get("NEG_SAMPLING_SIZE")

# Evaluation parameters
SINGLE_SHOW_PERF = ast.literal_eval(
    config.get("EVALUATION").get("SINGLE_SHOW_PERF"))
EVALUATE_FRAMES = ast.literal_eval(
    config.get("EVALUATION").get("EVALUATE_FRAMES"))
EVALUATE_FACE = ast.literal_eval(
    config.get("EVALUATION").get("EVALUATE_FACE")
)
EVALUATE_NAME = ast.literal_eval(config.get("EVALUATION").get("EVALUATE_NAME"))
CROP_VID = int(config.get("EVALUATION").get("CROP_VID"))

# Inference parameters
INF_DIR = config.get("INFERENCE").get("INF_DIR")
EXP_DIR_INF = config.get("INFERENCE").get("EXP_DIR_INF")
VIDEO_DIR_INF = config.get("INFERENCE").get("VIDEO_DIR_INF")
FRAME_DIR_INF = config.get("INFERENCE").get("FRAME_DIR_INF")
FACE_DIR_INF = config.get("INFERENCE").get("FACE_DIR_INF")
AUDIO_DIR_INF = config.get("INFERENCE").get("AUDIO_DIR_INF")
FRAMES_INF = ast.literal_eval(
    config.get("INFERENCE").get("FRAMES_INF")
)
TRANSCRIPTS_INF = ast.literal_eval(
    config.get("INFERENCE").get("TRANSCRIPTS_INF")
)
REFERENCE_FRAME_PATH_INF = config.get(
    "INFERENCE").get("REFERENCE_FRAME_PATH_INF")
REFERENCE_FRAME_EMBEDDING_PATH_INF = config.get("INFERENCE").get(
    "REFERENCE_FRAME_EMBEDDING_PATH_INF"
)
REFERENCE_FACE_PATH_INF = config.get(
    "INFERENCE").get("REFERENCE_FACE_PATH_INF")
REFERENCE_FACE_EMBEDDING_PATH_INF = config.get("INFERENCE").get(
    "REFERENCE_FACE_EMBEDDING_PATH_INF"
)
REFERENCE_NAME_DICT_PATH_INF = config.get("INFERENCE").get(
    "REFERENCE_NAME_DICT_PATH_INF"
)
REFERENCE_PROB_DICT_PATH_INF = config.get(
    "INFERENCE").get("REFERENCE_PROB_DICT_PATH_INF")
GEN_INF_EXPLANATION = ast.literal_eval(
    config.get("INFERENCE").get("GEN_INF_EXPLANATION"))

FRAME_THRES = float(config.get("INFERENCE").get("FRAME_THRES"))
FACE_THRES = float(config.get("INFERENCE").get("FACE_THRES"))
NAME_THRES = int(config.get("INFERENCE").get("NAME_THRES"))
CERTAINTY_THRES = float(config.get("INFERENCE").get("CERTAINTY_THRES"))
# seed dataloader
g = torch.Generator()
g.manual_seed(0)


def build_reference():
    """
    Training pipeline. With parameters set in config file, this function
    generates:
    1) a reference directory `reference` with the following contents:
        - `reference_frames` : a nested directory with refrerence frames for
                                each show.
        - `reference_faces` : a nested directory with refrerence faces for
                                each show.
        - `reference_emb` : a directory with refrerence frames
                                    enbeddings for all shows.
        - `reference_emb_face` : a directory with refrerence face
                                    enbeddings for all shows.
        - `reference_name_dict.json` : a json files with shownames as key and
                                        a list of names as value.
        - `reference_prob_dict.json` : a json files with reference frames and
                                        reference names as key and corresponding
                                        conditional probability as value.
    2) Model weights for frame model and face model, stored under `models` directory
    """
    # Preprocessing
    preprocess("Train", device, config)

    # Train siamese backbone for frame model
    if RETRAIN_BACKBONE:
        # Transformation parameters are defaulted how resnet is trained
        start = time.time()
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )
        train_set = ImageFolder(root=FRAME_DIR_TRAIN, transform=transform)
        eval_set = ImageFolder(root=FRAME_DIR_EVAL, transform=transform)
        num_classes = len(train_set.class_to_idx)
        train_batch_sampler = BalancedBatchSampler(
            train_set.targets, n_classes=num_classes, n_samples=NEG_SAMPLING_SIZE
        )
        test_batch_sampler = BalancedBatchSampler(
            eval_set.targets, n_classes=num_classes, n_samples=NEG_SAMPLING_SIZE
        )
        kwargs = {
            "num_workers": NUM_WORKERS,
            "pin_memory": True,
            "worker_init_fn": seed_worker,
            "generator": g,
        }
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_sampler=train_batch_sampler, **kwargs
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_set, batch_sampler=test_batch_sampler, **kwargs
        )

        # Training
        embedding_net = models.resnet50(pretrained=True)
        num_ftrs = embedding_net.fc.in_features
        embedding_net.fc = nn.Linear(num_ftrs, MODEL_LAST_LAYER_LOGITS)
        embedding_net.to(device)

        criterion = OnlineContrastiveLoss(
            SAMPLING_MARGIN, HardNegativePairSelector())

        optimizer = optim.Adam(embedding_net.parameters(), lr=LR)
        train_siamese_embedding(
            embedding_net,
            train_loader,
            eval_loader,
            device,
            criterion,
            optimizer,
            TRAIN_EPOCHS,
            model_name=MODEL_NAME,
            save_dir=MODEL_DIR,
            plot_save_dir=PLOT_DIR,
        )
        end = time.time()
        logger.info(f"Siamese training spent {end-start}s")


def inference(
    inference_clip,
    expected_show=None,
    explanation_save_dir=None,
    frame_thres=0.9,
    face_thres=0.8,
    previous_time_stamp=0.0
):
    """
    Inference pipeline. Main function for inference
    Args:
        inference_clip (str): path to video clip in .mxf format
        expected_show (str): show name, defined in show info, i.e. "Big Bang Theory", "Castle", ..
        explanation_save_dir (str): path to store explanation results
        frame_thres (float): threshold for frame matching
        face_thres (float): threshold for face matching
        previous_time_stamp (float): time stamp in seconds
    Returns:
        None
    """
    logger.info("Started Inference...")

    if FRAMES_INF:
        # Preprocessing
        start = time.time()
        preprocess(
            "Inference", device, config, inference_clip=inference_clip)
        end = time.time()
        logger.info(f"Processing total spent {end-start}s")

        # Frame predicton, Transformations are defaulted to how pretrained resnet is trained originally
        start = time.time()
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )

        inf_set = ImageFolderWithPaths(
            root=FRAME_DIR_INF, transform=transform)

        inf_loader = DataLoader(
            inf_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            generator=g,
        )

        model = torch.load(SAVED_MODEL, map_location=device)
        model.to(device)
        frame_result, len_predict = inference_siamese_network(
            model,
            device,
            inf_loader,
            REFERENCE_FRAME_EMBEDDING_PATH_INF,
            expected_show,
            frame_thres,
        )

        current_time_stamp = previous_time_stamp + len_predict*N_SECS_INF
        end = time.time()
        logger.info(f"Frame prediction spent {end-start}s")

        # Face predicton, Transformations are defaulted to how pretrained resnet is trained originally
        face_result = []
        if not len(glob.glob(FACE_DIR_INF)) == 0:
            start = time.time()
            transform = transforms.Compose(
                [
                    transforms.Resize((160, 160)),
                    transforms.ToTensor(),
                ]
            )
            inf_set = ImageFolderWithPaths(root=os.path.join(
                FACE_DIR_INF), transform=transform)
            inf_loader = DataLoader(
                inf_set,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                worker_init_fn=seed_worker,
                generator=g,
            )
            if SAVED_FACE_MODEL == "":
                model = InceptionResnetV1(pretrained='vggface2')
            else:
                model = torch.load(SAVED_FACE_MODEL, map_location=device)
            model.to(device)
            face_result, _ = inference_siamese_network(
                model,
                device,
                inf_loader,
                REFERENCE_FACE_EMBEDDING_PATH_INF,
                expected_show,
                face_thres,
                face_inf=True
            )
            end = time.time()
            logger.info(f"Face prediction spent {end-start}s")

        start = time.time()

        json_output = []
        with open(REFERENCE_PROB_DICT_PATH_INF, "r") as f:
            prob_dict = json.load(f)

        # generate frame/face results and explanations
        logger.info(
            f"Grabbing reference frames of expected show: {expected_show}")
        if not explanation_save_dir:
            explanation_save_dir = os.path.join(INF_DIR, "explanations")

        ref_frame_path_list = sorted(glob.glob(
            os.path.join(REFERENCE_FRAME_PATH_INF, f"{expected_show}/*.jpg")
        ))
        ext_frame_path_list = glob.glob(
            os.path.join(FRAME_DIR_INF, "test_clip", "*.jpg"))
        ref_face_path_list = sorted(glob.glob(
            os.path.join(REFERENCE_FACE_PATH_INF, f"{expected_show}/*.jpg")
        ))
        ext_face_path_list = glob.glob(
            os.path.join(FACE_DIR_INF, "test_clip", "*.jpg"))

        frame_exp_save_dir = os.path.join(
            explanation_save_dir,
            "frames",
        )
        face_exp_save_dir = os.path.join(
            explanation_save_dir,
            "faces",
        )
        os.makedirs(frame_exp_save_dir, exist_ok=True)
        os.makedirs(face_exp_save_dir, exist_ok=True)

        for ref_path, ext_id in frame_result:
            json_output.append(
                {"clip_name": inference_clip,
                 "match_type": "frame_match",
                 "match_ref": os.path.basename(ref_path),
                 "relative_time_stamp": f"{(ext_id+1)*N_SECS_INF+previous_time_stamp}s",
                 "match_certainty": str(prob_dict[expected_show]),
                 "current_certainty": str(0)})
            if GEN_INF_EXPLANATION:
                matched_ref_path = None
                for real_ref_path in ref_frame_path_list:
                    if os.path.basename(ref_path) == os.path.basename(real_ref_path):
                        matched_ref_path = real_ref_path
                        break
                if matched_ref_path == None:
                    ref_img = cv2.imread(real_ref_path)
                    logger.info(
                        "reference built path changed, updated reference build")
                else:
                    ref_img = cv2.imread(real_ref_path)

                ext_img = cv2.imread(ext_frame_path_list[ext_id])
                concat_img = cv2.hconcat([ref_img, ext_img])

                exp_save_name = os.path.join(
                    frame_exp_save_dir, os.path.basename(ext_frame_path_list[ext_id]))

                cv2.imwrite(exp_save_name, concat_img)

        for ref_path, ext_id in face_result:
            json_output.append(
                {"clip_name": inference_clip,
                 "match_type": "face_match",
                 "match_ref": os.path.basename(ref_path),
                 "relative_time_stamp": f"{(ext_id+1)*N_SECS_INF+previous_time_stamp}s",
                 "match_certainty": str(prob_dict[expected_show+"_face"]),
                 "current_certainty": str(0)})
            if GEN_INF_EXPLANATION:
                matched_ref_path = None
                for real_ref_path in ref_face_path_list:
                    if os.path.basename(ref_path) == os.path.basename(real_ref_path):
                        matched_ref_path = real_ref_path
                        break
                if matched_ref_path == None:
                    ref_img = cv2.imread(real_ref_path)
                    logger.info(
                        "reference built path changed, updated reference build")
                else:
                    ref_img = cv2.imread(real_ref_path)

                ext_img = cv2.imread(ext_face_path_list[ext_id])
                concat_img = cv2.hconcat([ref_img, ext_img])

                exp_save_name = os.path.join(
                    face_exp_save_dir, os.path.basename(ext_face_path_list[ext_id]))

                cv2.imwrite(exp_save_name, concat_img)
        end = time.time()
        logger.info(
            f"Frame result and explanation generation spent {end-start}s")

    if TRANSCRIPTS_INF:
        with open(REFERENCE_NAME_DICT_PATH_INF, "r") as f:
            ner_refs_dict = json.load(f)
        expected_ner_list = ner_refs_dict[expected_show]

        start = time.time()
        clip_match_list = []
        for channel_transcript_json in glob.glob(os.path.join(AUDIO_DIR_INF, "*.json")):

            with open(channel_transcript_json, "r") as f:
                trans_dict = json.load(f)
                sequence = trans_dict["transcription"]
                sequence_ts = trans_dict["trans_time_stamps"]

                for i, sentence in enumerate(sequence):
                    unique_words = set("".join(sentence).split())
                    matched_names_set = set(
                        expected_ner_list).intersection(unique_words)

                    if len(matched_names_set) > 0:
                        scaled_time = int(sequence_ts[i])
                        clip_match_list.append(
                            (matched_names_set, scaled_time))

        end = time.time()
        logger.info(
            f"Search entities from transcriptions spent {end-start}s")

        for (name_list, timestamp) in clip_match_list:
            for name in name_list:
                json_output.append(
                    {"clip_name": inference_clip,
                     "match_type": "name_match",
                     "match_ref": name,
                     "match_certainty": str(1),
                     "relative_time_stamp": f"{timestamp}",
                     "current_certainty": str(0)})

    logger.info(f"Generating Inference results")
    timestamp = str(datetime.datetime.now(tz)).replace(" ", "_")
    timestamp = timestamp.replace(":", "_")
    timestamp = timestamp.replace(".", "_")
    with open(
        os.path.join(INF_DIR, f"{timestamp}_tmp_results.json"), "w"
    ) as outfile:
        json.dump(json_output, outfile)
    return current_time_stamp


if __name__ == "__main__":
    if TASK == "Train":
        """
        Train and Build reference features from training dataset
        """
        build_reference()

    elif TASK == "Inference":
        """
        Perform inference on given clips from video stream when a "show check" is requested, 
        the event is triggered and will starts to analyse each downloaded clip. 

        Here the script analysis all downloaded clips in the demo folder
        Assume that duration of each stream is 30s and the saved clips from video streams are named as follows: 
        BBANGC051022-01_1.mxf, BBANGC051022-01_2.mxf, ....
        """

        if os.path.exists(os.path.dirname(FRAME_DIR_INF)):
            shutil.rmtree(os.path.dirname(FRAME_DIR_INF))
        if os.path.exists(EXP_DIR_INF):
            shutil.rmtree(EXP_DIR_INF)
        if os.path.exists(AUDIO_DIR_INF):
            shutil.rmtree(AUDIO_DIR_INF)
        if os.path.exists(AUDIO_DIR_INF + "_tmp"):
            shutil.rmtree(AUDIO_DIR_INF + "_tmp")
        for result_json_leftover in glob.glob(os.path.join(INF_DIR, "*_results.json")):
            os.remove(result_json_leftover)

        expected_show = EXPECTED_SHOW
        previous_time_stamp = 0.0
        for stream in sorted(glob.glob(os.path.join(INF_DIR, "*.mxf"))):
            logger.info("--------------------------------------")
            st = time.time()
            os.makedirs(VIDEO_DIR_INF, exist_ok=True)
            destination = os.path.join(VIDEO_DIR_INF, os.path.basename(stream))
            shutil.move(stream, destination)
            logger.info(f"doing inference on {stream}")
            current_time_stamp = inference(
                destination, expected_show, EXP_DIR_INF, FRAME_THRES, FACE_THRES, previous_time_stamp
            )
            previous_time_stamp = current_time_stamp

            shutil.move(os.path.join(VIDEO_DIR_INF,
                        os.path.basename(stream)), stream)
            end = time.time()
            logger.info(f"Single clip Inference spent {end-st}s")
            # remove temporary processing folder
            if os.path.exists(os.path.dirname(FRAME_DIR_INF)):
                shutil.rmtree(os.path.dirname(FRAME_DIR_INF))

        final_result_list = []
        sub_json_results = sorted(glob.glob(
            os.path.join(INF_DIR, "*_tmp_results.json")))
        frame_conf = 1
        for j_r in sub_json_results:
            with open(j_r, "r") as f:
                past_result_dict = json.load(f)
            for res in past_result_dict:
                frame_conf *= (1-float(res["match_certainty"]))
                res["current_certainty"] = str(1-frame_conf)
                final_result_list.append(res)
        result_generation_time = str(
            datetime.datetime.now(tz)).replace(" ", "-").replace(":", "-").replace(".", "-")
        with open(os.path.join(INF_DIR, f"final_result.json"), "w") as outfile:
            json.dump(final_result_list, outfile, indent=4)
