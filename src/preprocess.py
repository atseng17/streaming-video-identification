import os
import glob
import sys
import shutil
import logging
import ast
import pickle
import json
import subprocess
import random
from collections import Counter, defaultdict
import time
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pytz import timezone
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import scipy
from scipy.io import wavfile
from vosk import Model as vosk_model
from vosk import KaldiRecognizer, SetLogLevel
import wave
import spacy

from datasets import seed_worker, ImageFolderWithPaths
from train import evaluate_image_retrieval_SiameseNetwork, evaluate_all_frames

SetLogLevel(-1)
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

# seed dataloader
g = torch.Generator()
g.manual_seed(0)


def preprocess(phase, device, config, inference_clip=None):
    """extract frames from a video
    Args:
        phase (str): train, evaluation, or inference
        device (torch.device): cpu/gpu
        config (dict): config dictonary loaded from toml file
        inference_clip (str): path to video clip
    Returns:
        None
    """
    # ------------------------------------------------------------------------------#
    #                                 PARAMETERS                                   #
    # ------------------------------------------------------------------------------#
    # Data parameters
    VID_INFO = config.get("DATA").get("VID_INFO")
    RAW_VIDEO_DIR = config.get("DATA").get("RAW_VIDEO_DIR")
    VIDEO_DIR_TRAIN = config.get("DATA").get("VIDEO_DIR_TRAIN")
    VIDEO_DIR_EVAL = config.get("DATA").get("VIDEO_DIR_EVAL")
    FRAME_DIR_TRAIN = config.get("DATA").get("FRAME_DIR_TRAIN")
    FRAME_DIR_EVAL = config.get("DATA").get("FRAME_DIR_EVAL")
    FACE_DIR_TRAIN = config.get("DATA").get("FACE_DIR_TRAIN")
    FACE_DIR_EVAL = config.get("DATA").get("FACE_DIR_EVAL")
    AUDIO_DIR_TRAIN = config.get("DATA").get("AUDIO_DIR_TRAIN")
    # Preprocessing parameters
    RECREATE_VID_SPLIT = ast.literal_eval(
        config.get("PREPROCESS").get("RECREATE_VID_SPLIT")
    )
    RECREATE_FRAMES = ast.literal_eval(
        config.get("PREPROCESS").get("RECREATE_FRAMES"))
    RECREATE_REFERENCE_FRAME = ast.literal_eval(
        config.get("PREPROCESS").get("RECREATE_REFERENCE_FRAME")
    )
    RECREATE_REFERENCE_FACE = ast.literal_eval(
        config.get("PREPROCESS").get("RECREATE_REFERENCE_FACE")
    )
    RECREATE_REFERENCE_FRAME_EMBEDDINGS = ast.literal_eval(
        config.get("PREPROCESS").get("RECREATE_REFERENCE_FRAME_EMBEDDINGS")
    )
    RECREATE_REFERENCE_FACE_EMBEDDINGS = ast.literal_eval(
        config.get("PREPROCESS").get("RECREATE_REFERENCE_FACE_EMBEDDINGS")
    )
    RECREATE_REFERENCE_NAMES = ast.literal_eval(
        config.get("PREPROCESS").get("RECREATE_REFERENCE_NAMES")
    )
    RECREATE_PROBABILITY_TABLE = ast.literal_eval(
        config.get("PREPROCESS").get("RECREATE_PROBABILITY_TABLE")
    )
    RANDOM_REF_SELECTION = ast.literal_eval(
        config.get("PREPROCESS").get("RANDOM_REF_SELECTION")
    )
    FILTER_FACES = ast.literal_eval(
        config.get("PREPROCESS").get("FILTER_FACES")
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

    BLUR_THRES = config.get("PREPROCESS").get("BLUR_THRES")
    FPOSE_THRES = config.get("PREPROCESS").get("FPOSE_THRES")
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

    # Traning parameters
    RETRAIN_BACKBONE = ast.literal_eval(
        config.get("TRAIN").get("RETRAIN_BACKBONE"))
    BATCH_SIZE = config.get("TRAIN").get("BATCH_SIZE")
    BF_THRES = config.get("TRAIN").get("BF_THRES")
    MOST_COMMON_NAME_K = config.get("TRAIN").get("MOST_COMMON_NAME_K")
    NUM_WORKERS = config.get("TRAIN").get("NUM_WORKERS")
    # Evaluation parameters
    CROP_VID = int(config.get("EVALUATION").get("CROP_VID"))

    # Inference parameters
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
    FACE_THRES = config.get("INFERENCE").get("FACE_THRES")

    if phase == "Train":

        if RECREATE_VID_SPLIT == True and RETRAIN_BACKBONE == True:
            logger.info("RECREATE VIDS")
            if os.path.exists(VIDEO_DIR_TRAIN):
                shutil.rmtree(VIDEO_DIR_TRAIN)
            video_train_test_split(
                VID_INFO, RAW_VIDEO_DIR, VIDEO_DIR_TRAIN, VIDEO_DIR_EVAL
            )

        if RECREATE_FRAMES == True and RETRAIN_BACKBONE == True:
            logger.info("RECREATE FRAMES")
            if os.path.exists(FRAME_DIR_TRAIN):
                shutil.rmtree(FRAME_DIR_TRAIN)
                shutil.rmtree(FRAME_DIR_TRAIN + "_face_crop",
                              ignore_errors=True)
            if os.path.exists(FRAME_DIR_EVAL):
                shutil.rmtree(FRAME_DIR_EVAL)
                shutil.rmtree(FRAME_DIR_EVAL + "_face_crop",
                              ignore_errors=True)

            v_proc = VideoProcessor()
            v_proc.store_frames_from_video_clips_image_folder(
                VID_INFO,
                VIDEO_DIR_TRAIN,
                FRAME_DIR_TRAIN,
                device,
                n_secs=N_SECS_TRAIN,
                face_only=FACE_ONLY,
                black_frame_thres=BF_THRES,
                face_crop=FACE_CROP,
                face_crop_dir=FACE_DIR_TRAIN
            )
            v_proc.store_frames_from_video_clips_image_folder(
                VID_INFO,
                VIDEO_DIR_EVAL,
                FRAME_DIR_EVAL,
                device,
                n_secs=N_SECS_EVAL,
                face_only=FACE_ONLY,
                black_frame_thres=BF_THRES,
                face_crop=FACE_CROP,
                face_crop_dir=FACE_DIR_EVAL
            )

        if RECREATE_REFERENCE_FRAME == True and RETRAIN_BACKBONE == False:
            # The following section creates frame references
            logger.info("RECREATE REFERENCE FRAMES")
            st = time.time()
            if os.path.exists(REFERENCE_FRAME_PATH):
                shutil.rmtree(REFERENCE_FRAME_PATH)

            if RANDOM_REF_SELECTION == False:
                if SAVED_MODEL == "":
                    model = torch.load(
                        os.path.join(MODEL_DIR, MODEL_NAME,
                                     "checkpoint_latest.pt"),
                        map_location=device,
                    )
                else:
                    model = torch.load(SAVED_MODEL, map_location=device)

            make_frame_ref_folder(
                FRAME_DIR_TRAIN, REFERENCE_FRAME_PATH, NUM_REF, BATCH_SIZE, model, device, NUM_WORKERS, random=RANDOM_REF_SELECTION
            )
            end = time.time()
            logger.info(f"RECREATE REFERENCE FRAMES spent {end-st}s")

        if RECREATE_REFERENCE_FACE == True and RETRAIN_BACKBONE == False:
            logger.info("RECREATE REFERENCE FRAMES")
            st = time.time()
            if os.path.exists(REFERENCE_FACE_PATH):
                shutil.rmtree(REFERENCE_FACE_PATH)

            make_frame_ref_folder(
                FACE_DIR_TRAIN, REFERENCE_FACE_PATH, NUM_REF, BATCH_SIZE, None, device, NUM_WORKERS, random=True
            )

            if FILTER_FACES:
                # Currently faces are manually picked, no need to ass filters to faces
                pass

            end = time.time()
            logger.info(f"RECREATE REFERENCE FRAMES spent {end-st}s")

        if RECREATE_REFERENCE_FRAME_EMBEDDINGS == True and RETRAIN_BACKBONE == False:
            # The following section creates frame embeddings
            logger.info("RECREATE REFERENCE EMBEDDINGS")
            st = time.time()
            transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [
                                         0.229, 0.224, 0.225]),
                ]
            )

            ref_set = ImageFolderWithPaths(
                root=REFERENCE_FRAME_PATH, transform=transform)
            ref_loader = DataLoader(
                ref_set,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                worker_init_fn=seed_worker,
                generator=g,
            )
            if SAVED_MODEL == "":
                model = torch.load(
                    os.path.join(MODEL_DIR, MODEL_NAME,
                                 "checkpoint_latest.pt"),
                    map_location=device,
                )
            else:
                model = torch.load(SAVED_MODEL, map_location=device)
            model.to(device)
            model.eval()
            save_reference_image_embeddings(
                model, device, ref_loader, REFERENCE_FRAME_EMBEDDING_PATH
            )
            end = time.time()
            logger.info(f"RECREATE REFERENCE EMBEDDINGS spent {end-st}s")

        if RECREATE_REFERENCE_FACE_EMBEDDINGS == True and RETRAIN_BACKBONE == False:
            # The following section creates face embeddings
            logger.info("RECREATE REFERENCE FACES EMBEDDINGS")
            st = time.time()
            transform = transforms.Compose(
                [
                    transforms.Resize((160, 160)),
                    transforms.ToTensor(),
                ]
            )
            ref_face_set = ImageFolderWithPaths(
                root=REFERENCE_FACE_PATH, transform=transform)

            ref_face_loader = DataLoader(
                ref_face_set,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                worker_init_fn=seed_worker,
                generator=g,
            )

            if SAVED_FACE_MODEL == "":
                model = InceptionResnetV1(pretrained="vggface2")
                logger.info(
                    f"Using pretrained VGGface2 for embedding projection")
            else:
                model = torch.load(SAVED_FACE_MODEL, map_location=device)
            model.to(device)
            model.eval()
            save_reference_image_embeddings(
                model, device, ref_face_loader, REFERENCE_FACE_EMBEDDING_PATH
            )
            end = time.time()
            logger.info(f"RECREATE REFERENCE FACES EMBEDDINGS spent {end-st}s")

        if RECREATE_REFERENCE_NAMES and RETRAIN_BACKBONE == False:

            # The following section creates actor names
            logger.info("RECREATE REFERENCE NAMES")
            # Video to audio
            st = time.time()
            AudioProcessor.convert_video_to_wav_parallel(
                input_dir=VIDEO_DIR_TRAIN, out_path=AUDIO_DIR_TRAIN
            )
            end = time.time()
            logger.info(f"Audio extraction spent:{end-st}s")

            logger.info("Trascription extraction with Vosk model started")
            # Audio to transcription(The Assumption here is the Englich channel is in front center or left/right center)
            wav_paths = glob.glob(os.path.join(AUDIO_DIR_TRAIN, "*front*.wav"))
            st = time.time()
            vosk_weight = "models/vosk-model-en-us-0.22/"
            v_model = vosk_model(vosk_weight)
            for wave_file in wav_paths:
                AudioProcessor.vosk_speech_to_text_single_and_save(
                    v_model, wave_file)
            end = time.time()
            logger.info(f"Transcription extraction spent:{end-st}s")

            # Perform ner one transcripts and make reference
            transcriptions_list = glob.glob(
                os.path.join(AUDIO_DIR_TRAIN, "*.json"))
            st = time.time()
            _ = AudioProcessor.reference_transcript_ner(
                VID_INFO,
                transcriptions_list,
                save_ner_path=REFERENCE_NAME_DICT_PATH,
                most_common_k=MOST_COMMON_NAME_K,
                get_name_histogram=True,
            )
            ed = time.time()
            logger.info(f"Building ner dict spent {ed-st}s")

        if RECREATE_PROBABILITY_TABLE == True and RETRAIN_BACKBONE == False:
            # The following section creates conditional probabilities
            logger.info("RECREATE REFERENCE CONDITIONAL PROBABILITY")
            st = time.time()
            prob_dict = defaultdict(float)
            show_info_df = pd.read_csv(VID_INFO)
            # Assuming all shows (including unseen shows should be included in csv)
            show_list = list(set(show_info_df["Show Name"]))

            for expected_show in sorted(show_list):

                # Frame probs
                # Transforms
                transform = transforms.Compose(
                    [
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [
                            0.229, 0.224, 0.225]),
                    ]
                )

                eval_set = ImageFolderWithPaths(
                    root=FRAME_DIR_EVAL, transform=transform)
                eval_loader = DataLoader(
                    eval_set,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    worker_init_fn=seed_worker,
                    generator=g,
                )

                if SAVED_MODEL == "":
                    logger.info(f"using latest {MODEL_NAME} weights")
                    model = torch.load(
                        os.path.join(MODEL_DIR, MODEL_NAME,
                                     "checkpoint_latest.pt"),
                        map_location=device,
                    )
                else:
                    logger.info(f"using saved weights: {SAVED_MODEL}")
                    model = torch.load(SAVED_MODEL, map_location=device)
                model.to(device)
                model.eval()

                full_eval_metric = evaluate_image_retrieval_SiameseNetwork(
                    model,
                    device,
                    eval_loader,
                    REFERENCE_FRAME_EMBEDDING_PATH,
                    save_similarity_pair_histo=True,
                    expected_show=expected_show,
                    thres=0.9,
                    eval_as_classification=True,
                )

                precision = full_eval_metric[0]
                prob_dict[expected_show] = precision

                # Face probs
                transform = transforms.Compose(
                    [
                        transforms.Resize((160, 160)),
                        transforms.ToTensor(),
                    ]
                )

                eval_set = ImageFolderWithPaths(
                    root=FACE_DIR_EVAL, transform=transform)
                eval_loader = DataLoader(
                    eval_set,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    worker_init_fn=seed_worker,
                    generator=g,
                )

                if SAVED_FACE_MODEL == "":
                    logger.info(f"using latest pretrained vggface2 weights")
                    model = InceptionResnetV1(pretrained='vggface2')
                else:
                    logger.info(f"using saved weights: {SAVED_FACE_MODEL}")
                    model = torch.load(SAVED_FACE_MODEL, map_location=device,)

                model.to(device)
                model.eval()

                full_eval_metric = evaluate_image_retrieval_SiameseNetwork(
                    model,
                    device,
                    eval_loader,
                    REFERENCE_FACE_EMBEDDING_PATH,
                    save_similarity_pair_histo=True,
                    expected_show=expected_show,
                    thres=FACE_THRES,
                    eval_as_classification=True,
                )

                precision = full_eval_metric[0]
                prob_dict[expected_show+"_face"] = precision

                # Transcripton probs, based on number of direct matches, not precision
                prob_dict[expected_show+"_name"] = 1.0

            with open(REFERENCE_PROB_DICT_PATH, "w") as outfile:
                json.dump(prob_dict, outfile)
            end = time.time()
            logger.info(
                f"RECREATE REFERENCE CONDITIONAL PROBABILITY spent {end-st}s")

    elif phase == "Inference":
        if FRAMES_INF:
            start = time.time()
            logger.info("CREATE FRAMES")
            v_proc = VideoProcessor()
            v_proc.store_frames_from_single_video_clip_image_folder(
                inference_clip,
                FRAME_DIR_INF,
                device,
                n_secs=N_SECS_INF,
                face_only=FACE_ONLY,
                black_frame_thres=BF_THRES,
                crop_length=CROP_VID,
                face_crop=FACE_CROP,
                face_crop_dir=FACE_DIR_INF
            )

            if FILTER_FACES:
                face_img_list = glob.glob(
                    os.path.join(FACE_DIR_INF, "test_clip", "*.jpg"))
                filtered_list = VideoProcessor.blur_filter(
                    face_img_list, BLUR_THRES)
                filtered_list = VideoProcessor.pose_filter(
                    filtered_list, FPOSE_THRES, device)
                if len(filtered_list) != 0:
                    for img_path in filtered_list:
                        save_path = img_path.replace(
                            "crop", "crop_filtered")
                        image = cv2.imread(img_path)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, image)
            end = time.time()
            logger.info(f"frame+face processing spent {end-start}s")

        if TRANSCRIPTS_INF:
            # Video to audio
            st = time.time()
            AudioProcessor.convert_video_to_wav_parallel(
                input_dir=VIDEO_DIR_INF, out_path=AUDIO_DIR_INF
            )
            end = time.time()
            logger.info(f"Audio extraction spent:{end-st}s")

            # Audio to transcription
            wav_paths = glob.glob(os.path.join(AUDIO_DIR_INF, "*front*.wav"))
            st = time.time()
            vosk_weight = "models/vosk-model-en-us-0.22/"
            v_model = vosk_model(vosk_weight)
            for wave_file in wav_paths:
                logger.info(f"Extracting transcripts from {wave_file}")
                if os.path.exists(wave_file.replace(".wav", ".json")):
                    continue
                AudioProcessor.vosk_speech_to_text_single_and_save(
                    v_model, wave_file)
            end = time.time()
            logger.info(f"Transcription extraction spent:{end-st}s")


class AudioProcessor:
    # The default constructor for a video processor.
    def __init__(self, inf_vid_path, inf_audio_path):
        pass

    @staticmethod
    def convert_video_to_wav_single(input_path, out_path=None):
        """extract audio channels a given video clip
        Args:
            input_dir (str): path to video folder with .mxf clips
            out_path (str): path to save audio files
        Returns:
            response (str): path to saved audio files
        """
        os.makedirs(out_path, exist_ok=True)
        temp_out_path = out_path.replace("audio", "audio_tmp")
        os.makedirs(temp_out_path, exist_ok=True)

        response = ""
        file_name = os.path.basename(input_path).split(".")[
            0].replace(" ", "_")

        # Extract audio stream
        query = (
            f"ffmpeg -y -i {input_path} -c:a pcm_s16le {temp_out_path}/{file_name}.wav"
        )
        p1 = subprocess.Popen(
            query, shell=True, stdout=subprocess.PIPE
        )
        response = f"{temp_out_path}/{file_name}.wav"
        p1.wait()

        # Extract seperate channels from audio stream
        query = f'ffmpeg -y -i {response} -filter_complex \
                        "channelsplit=channel_layout=7.1[FL][FR][FC][LFE][BL][BR][SL][SR]" \
                        -map "[FL]" {out_path}/{file_name}_front_left.wav -map "[FR]" {out_path}/{file_name}_front_right.wav \
                        -map "[FC]" {out_path}/{file_name}_front_center.wav -map "[LFE]" {out_path}/{file_name}_low_frequency_effects.wav \
                        -map "[BL]" {out_path}/{file_name}_back_left.wav -map "[BR]" {out_path}/{file_name}_back_right.wav \
                        -map "[SL]" {out_path}/{file_name}_side_left.wav -map "[SR]" {out_path}/{file_name}_side_right.wav'

        p2 = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE)
        p2.wait()

        return response

    @staticmethod
    def convert_video_to_wav_parallel(input_dir, out_path="data/processed/audio"):
        """extract audio channels from all video clips in a folder
        Args:
            input_dir (str): path to video folder with .mxf clips
            out_path (str): path to save audio files
        Returns:
            None
        """
        os.makedirs(out_path, exist_ok=True)
        temp_out_path = out_path.replace("audio", "audio_tmp")
        os.makedirs(temp_out_path, exist_ok=True)
        process_list = []
        for input_path in glob.glob(os.path.join(input_dir, "*.mxf")):
            file_name = os.path.basename(input_path).split(".")[
                0].replace(" ", "_")

            # Extract audio stream
            query = f"ffmpeg -y -i {input_path} -c:a pcm_s16le {temp_out_path}/{file_name}.wav"
            process_list.append(
                subprocess.Popen(query, shell=True, stdout=subprocess.PIPE)
            )
        for p in process_list:
            p.wait()

        process_list_2 = []
        for input_path in glob.glob(os.path.join(temp_out_path, "*.wav")):
            file_name = os.path.basename(input_path).split(".")[
                0].replace(" ", "_")
            # Extract seperate channels from audio stream
            query = f'ffmpeg -y -i {input_path} -filter_complex \
                        "channelsplit=channel_layout=7.1[FL][FR][FC][LFE][BL][BR][SL][SR]" \
                        -map "[FL]" {out_path}/{file_name}__front_left.wav -map "[FR]" {out_path}/{file_name}__front_right.wav \
                        -map "[FC]" {out_path}/{file_name}__front_center.wav -map "[LFE]" {out_path}/{file_name}__low_frequency_effects.wav \
                        -map "[BL]" {out_path}/{file_name}__back_left.wav -map "[BR]" {out_path}/{file_name}__back_right.wav \
                        -map "[SL]" {out_path}/{file_name}__side_left.wav -map "[SR]" {out_path}/{file_name}__side_right.wav'

            p2 = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE)
            p2.wait()
        for p2 in process_list_2:
            p2.wait()

    @staticmethod
    def vosk_word_extraction(audio_data, audio_info, model):
        """Vosk speech to text extraction
        Args: 
            audio_data (numpy.ndarray): amplitude values
            audio_info (float): sampling rate
            model (vosk model object): pretrained vosk model

        Returns: 
            results (list): list of entity dictionaries
        """

        temp_file = io.BytesIO()
        scipy.io.wavfile.write(temp_file, audio_info,
                               audio_data.astype(np.int16))

        wf = wave.open(temp_file, "rb")
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getcomptype() != "NONE"
        ):
            logger.info("Audio file must be WAV format mono PCM.")
            exit(1)
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        # get the list of JSON dictionaries
        results = []
        trans_time_stamp = []
        # recognize speech using vosk model
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                json_results = json.loads(rec.Result())
                part_result = json_results["text"]
                results.append(part_result)
                try:
                    part_time_stamp = json_results["result"][-1]["end"]
                except:
                    part_time_stamp = ""
                trans_time_stamp.append(part_time_stamp)
        json_final_results = json.loads(rec.FinalResult())
        part_result = json_final_results["text"]
        try:
            part_time_stamp = json_final_results["result"][-1]["end"]
        except:
            pass

        results.append(part_result)
        trans_time_stamp.append(part_time_stamp)

        return results, trans_time_stamp

    @staticmethod
    def vosk_speech_to_text_single_and_save(model, audio_path):
        """Extract transcription from audio file and save as json
        Args:
            model (str): path to vosk model weight
            audio_path (str): path to a single audio file
        Returns:
            None
        """
        if os.path.exists(audio_path.replace(".wav", ".json")):
            logger.info("transcriptions already exist")
        else:
            samplerate, wav_data = wavfile.read(audio_path)
            audio_words, trans_time_stamps = AudioProcessor.vosk_word_extraction(
                wav_data, samplerate, model
            )
            file_name = os.path.basename(audio_path).split(".")[0]
            dict_ = {"file_name": file_name, "transcription": audio_words,
                     "trans_time_stamps": trans_time_stamps}
            out_dir = os.path.dirname(audio_path)
            with open(f"{os.path.join(out_dir, file_name)}.json", "w") as outfile:
                json.dump(dict_, outfile)

    @staticmethod
    def reference_transcript_ner(
        vid_info,
        transcriptions_list,
        save_ner_path,
        most_common_k=5,
        get_name_histogram=False,
    ):
        """Perform Named Entity Recognition(NER) on given transcritpions
        Args:
            vid_info (.csv): path to video file
            transcriptions_list(int): list fof transcription files in json format
            save_ner_path (str): path to store reference file in json format
            most_common_k (int): pick the top n names as reference
            get_name_histogram (bool): True if plot hostograms of names in a show
        Returns:
            frames(list): list of frames with shape (H,W,3)
        """
        os.makedirs(os.path.dirname(save_ner_path), exist_ok=True)
        show_info_df = pd.read_csv(vid_info)
        show_ner_set = defaultdict(list)
        # Load ner model weights
        sp_lg = spacy.load("en_core_web_lg")

        # Collect all entities of transcriptions for each show
        for transcript_json in transcriptions_list:
            logger.info(f"processing {transcript_json}")
            with open(transcript_json, "r") as f:
                trans_dict = json.load(f)
            clip_name = trans_dict["file_name"].split("__")[0] + ".mxf"
            show_name = show_info_df[show_info_df["File Name"] == clip_name][
                "Show Name"
            ].item()
            sequence = trans_dict["transcription"]
            sequence = ". ".join(sequence)

            ner_dict = {(ent.text.strip(), ent.label_)
                        for ent in sp_lg(sequence).ents}

            name_list = []
            for (word, entity_type) in ner_dict:
                if entity_type == "PERSON":
                    name_list.append(word)

            show_ner_set[show_name].extend(name_list)
        # Get most common names in each show as reference names of the show
        for key, val in show_ner_set.items():
            cnt = Counter(val).most_common(most_common_k)
            show_ner_set[key] = [w for w, wcnt in cnt]
            if get_name_histogram:
                top_10_list_names = [w for w, wcnt in cnt]
                top_10_list_counts = [wcnt for w, wcnt in cnt]
                color_list = ["steelblue", "royalblue"]
                fig, axs = plt.subplots(figsize=(12, 5))
                axs.bar(
                    top_10_list_names,
                    np.array(top_10_list_counts),
                    label="Train",
                    color=color_list[0],
                    edgecolor="none",
                )
                axs.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
                axs.set_ylabel("Counts", fontsize=20)
                axs.set_xlabel("Names", fontsize=20)
                axs.set_title(f"{key}", fontsize=20)
                plt.xticks(rotation="vertical")
                fig.tight_layout()
                plt.savefig(f"name_histo_{key}.png")

        # Save reference names of the shows
        with open(save_ner_path, "w") as outfile:
            json.dump(show_ner_set, outfile)
        return show_ner_set

    @staticmethod
    def get_evaluation_ner_dict(vid_info, transcriptions_path):
        """Evaluate names references on eval dataset
        Args:
            vid_info (.csv): path to video file
            transcriptions_list(int): list fof transcription files in json format
        Returns:
            frames(list): list of frames with shape (H,W,3)
        """
        show_info_df = pd.read_csv(vid_info)
        eval_dict = defaultdict(lambda: defaultdict(list))
        sp_lg = spacy.load("en_core_web_lg")
        for channel_transcript_json in transcriptions_path:

            with open(channel_transcript_json, "r") as f:
                trans_dict = json.load(f)

            clip_name = trans_dict["file_name"].split("_")[0] + ".mxf"
            show_name = show_info_df[show_info_df["File Name"] == clip_name][
                "Show Name"
            ].item()

            sequence = trans_dict["transcription"]
            sequence = ". ".join(sequence)
            ner_dict = {(ent.text.strip(), ent.label_)
                        for ent in sp_lg(sequence).ents}

            for (word, entity_type) in ner_dict:
                if entity_type == "PERSON":
                    eval_dict[show_name][clip_name].append(word)
        return eval_dict

    @staticmethod
    def get_inference_ner_list(transcriptions_path):
        """Evaluate names references on eval dataset
        Args:
            vid_info (.csv): path to video file
            transcriptions_list(int): list fof transcription files in json format
        Returns:
            frames(list): list of frames with shape (H,W,3)
        """
        inf_list = []
        sp_lg = spacy.load("en_core_web_lg")
        for channel_transcript_json in transcriptions_path:

            with open(channel_transcript_json, "r") as f:
                trans_dict = json.load(f)

            sequence = trans_dict["transcription"]
            sequence = ". ".join(sequence)
            ner_dict = {(ent.text.strip(), ent.label_)
                        for ent in sp_lg(sequence).ents}

            for (word, entity_type) in ner_dict:
                if entity_type == "PERSON":
                    inf_list.append(word)
        return inf_list


class VideoProcessor:
    # The default constructor for a video processor.
    def __init__(self):
        pass

    def get_frames_per_n_seconds(
        self, filename, n_secs, black_frame_thres=10, crop_length=0, first_frame=False
    ):
        """extract frames from a video
        Args:
            filename(str): path to video file
            n_secs(int): extract frame per n seconds
            black_frame_thres (int): sum of pixel values as threshold for black frame skipping
            crop_length(int): only get frames for the first n seconds
        Returns:
            frames(list): list of frames with shape (H,W,3)
        """

        frames = []
        v_cap = cv2.VideoCapture(filename)
        frameRate = v_cap.get(5)  # frame rate
        while v_cap.isOpened():
            frame_id = v_cap.get(1)  # current frame number
            current_sec = frame_id // frameRate
            if crop_length != 0:
                if current_sec // crop_length == 1:
                    logger.info("extraction ended, reached crop length")
                    break
            success, frame = v_cap.read()

            if success != True:
                break
            if frame_id % np.floor(frameRate * n_secs) == 0:
                if np.mean(frame) < black_frame_thres:
                    logger.info("skipping black screen")
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frames.append(frame)
                if first_frame:
                    v_cap.release()
                    return frames
        v_cap.release()

        return frames

    def store_frames(self, frames, path2store):
        """save frames to local directory
        Args:
            frames(list): list of np.ndarray arrays
            path2store(str): path to video frames
        Returns:
            img_path_list(list): list of frame paths
        """
        img_path_list = []
        for i, frame in enumerate(frames):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            path2img = path2store + "_frame" + str(i) + ".jpg"
            cv2.imwrite(path2img, frame)
            img_path_list.append(path2img)
        return img_path_list

    def store_frames_from_single_video_clip_image_folder(
        self,
        video_clip_path,
        frame_save_dir,
        device,
        n_secs=None,
        face_only=False,
        black_frame_thres=10,
        crop_length=0,
        face_crop=False,
        face_crop_dir=None
    ):
        """wrapper for extracting and save frames to local directory, frames will be stored as 
        {clipname}_frame_{i}.jpg, faces will be stored as face_{clipname}_{i}.jpg, 
        relative time for obtaining these frames/faces wii be n_secs*i

        Args:
            video_dir (str): path to video dir
            frame_save_dir (str): path to store frames
            frame_csv_save_path (str): path to save frame info
            n_secs (int): extract frames per n seconds
            face_only (boolean): if true, then only save frames with faces
            device (torch device object)
            black_frame_thres (int): sum of pixel values as threshold for black frame skipping
            crop_length(int): only get frames for the first n seconds
            inference (boolean): if not true, function assumes pipeline is in evaluation phase
            face_crop (boolean): option to crop faces from frames and save to disk
        Returns:
            None
        """

        logger.info(f"creating inference frames at {frame_save_dir}")
        os.makedirs(frame_save_dir, exist_ok=True)

        if face_only:
            # define face detection model with suggested parameters in the MTCNN paper
            face_model = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                keep_all=True,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=device,
            )

        skip_frame = False
        video_name = video_clip_path.split("/")[-1].split(".")[0]

        frame_sub_dir = os.path.join(frame_save_dir, "test_clip")
        os.makedirs(frame_sub_dir, exist_ok=True)
        logger.info(
            f"extracting frames from {video_clip_path} every {n_secs} secs")

        frames = self.get_frames_per_n_seconds(
            video_clip_path, n_secs, black_frame_thres, crop_length=crop_length
        )
        logger.info(
            f"frame capture completed, {len(frames)} frames in current clip.")

        iter_ = 0
        while len(frames) == 0:
            logger.info(
                f"no frames captured for {video_clip_path}, try using higher fps{n_secs/(iter_+1)}"
            )
            frames = self.get_frames_per_n_seconds(
                video_clip_path, n_secs, black_frame_thres, crop_length=crop_length
            )
            if iter_ == 2:
                skip_frame = True
                break
            iter_ += 1
        if skip_frame:
            logger.info(
                f"still no frames captured for {video_clip_path}, skipping this clip"
            )

        clip_name = video_clip_path.split("/")[-1]
        # face detection via MTCNN model
        if face_only:
            title_frame_list = self.get_frames_per_n_seconds(
                video_clip_path, 1, black_frame_thres, crop_length, first_frame=True)
            title_frame = cv2.cvtColor(title_frame_list[0], cv2.COLOR_RGB2BGR)
            path2store = os.path.join(frame_sub_dir, clip_name.split(".")[0])

            path2img = path2store + "_frame" + "0000" + ".jpg"
            cv2.imwrite(path2img, title_frame)

            self.get_face_frames(
                clip_name, frame_sub_dir, frames, face_model, face_crop, face_crop_dir
            )

        else:
            _ = self.store_frames(
                frames, os.path.join(frame_sub_dir, clip_name.split(".")[0])
            )

    def store_frames_from_video_clips_image_folder(
        self,
        video_info,
        video_dir,
        frame_save_dir,
        device,
        n_secs=None,
        face_only=False,
        black_frame_thres=10,
        crop_length=0,
        inference=False,
        face_crop=False,
        face_crop_dir=None
    ):
        """wrapper for extracting and save frames to local directory
        Args:
            video_dir (str): path to video dir
            frame_save_dir (str): path to store frames
            frame_csv_save_path (str): path to save frame info
            n_secs (int): extract frames per n seconds
            face_only (boolean): if true, then only save frames with faces
            device (torch device object)
            black_frame_thres (int): sum of pixel values as threshold for black frame skipping
            crop_length(int): only get frames for the first n seconds
            inference (boolean): if not true, function assumes pipeline is in evaluation phase
            face_crop (boolean): option to crop faces from frames and save to disk
        Returns:
            None
        """

        video_list = sorted(glob.glob(video_dir + "/*.mxf"))

        logger.info(f"creating inference frames at {frame_save_dir}")
        os.makedirs(frame_save_dir, exist_ok=True)

        if face_only:
            # define face detection model
            face_model = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                keep_all=True,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=device,
            )
        if not inference:
            show_info_df = pd.read_csv(video_info)

        for i, video_clip_path in enumerate(video_list):
            skip_frame = False
            if not inference:
                logger.info(
                    f"video_clip_pathvideo_clip_path: {video_clip_path}")
                video_name = show_info_df[
                    show_info_df["File Name"] == os.path.basename(
                        video_clip_path)
                ]["Show Name"].values[0]
            else:
                video_name = video_clip_path.split("/")[-1].split(".")[0]

            frame_sub_dir = os.path.join(frame_save_dir, video_name)
            os.makedirs(frame_sub_dir, exist_ok=True)
            logger.info(
                f"extracting frames from {video_clip_path} every {n_secs} secs")

            current_clipname = os.path.basename(video_clip_path).split(".")[0]
            current_frames = glob.glob(os.path.join(
                frame_sub_dir, f"{current_clipname}*"))
            if len(current_frames) != 0:
                logger.info("frames already exists")
                continue

            frames = self.get_frames_per_n_seconds(
                video_clip_path, n_secs, black_frame_thres, crop_length=crop_length
            )
            logger.info(
                f"frame capture completed, {len(frames)} frames in current clip."
            )

            iter_ = 0
            while len(frames) == 0:
                logger.info(
                    f"no frames captured for {video_clip_path}, try using higher fps{n_secs/(iter_+1)}"
                )
                frames = self.get_frames_per_n_seconds(
                    video_clip_path, n_secs, black_frame_thres, crop_length=crop_length
                )
                if iter_ == 2:
                    skip_frame = True
                    break
                iter_ += 1
            if skip_frame:
                logger.info(
                    f"still no frames captured for {video_clip_path}, skipping this frame"
                )
                continue

            clip_name = video_clip_path.split("/")[-1]
            # face detection via MTCNN model
            if face_only:
                title_frame_list = self.get_frames_per_n_seconds(
                    video_clip_path, 1, black_frame_thres, crop_length, first_frame=True)
                title_frame = cv2.cvtColor(
                    title_frame_list[0], cv2.COLOR_RGB2BGR)
                path2store = os.path.join(
                    frame_sub_dir, clip_name.split(".")[0])

                path2img = path2store + "_frame" + "0000" + ".jpg"
                cv2.imwrite(path2img, title_frame)

                self.get_face_frames(
                    clip_name, frame_sub_dir, frames, face_model, face_crop, face_crop_dir
                )

            else:
                _ = self.store_frames(
                    frames, os.path.join(
                        frame_sub_dir, clip_name.split(".")[0])
                )

    def get_face_frames(self, clip_name, frame_sub_dir, frames, model, face_crop=False, face_crop_dir=None):
        """Only store frames with faces
        Args:
            clip_name (str): clip basename
            frame_sub_dir (str): path to store frames for each show
            frames (list): list of np.ndarray
            model (torch.nn object): face detection model
        Returns:
            none, saves video frames to folders
        """
        pil_frame_list = [
            Image.fromarray(frame.astype("uint8"), "RGB") for frame in frames
        ]
        try:
            if face_crop:
                train_val_inf_dir = os.path.dirname(frame_sub_dir)
                video_name = os.path.basename(frame_sub_dir)
                face_crop_dir = os.path.join(
                    face_crop_dir, video_name
                )
                clip_origin = clip_name.split(".")[0]

                save_paths = [
                    os.path.join(
                        face_crop_dir, f"{clip_origin}_face{i}.jpg")
                    for i in range(len(frames))
                ]

            else:
                save_paths = None

            _, prob = model(pil_frame_list, return_prob=True,
                            save_path=save_paths)

        except ValueError as e:
            logger.info(
                f"frame list empty? {len(pil_frame_list)==0}, error message: {e}"
            )

        # get all face frames
        face_frame = []
        for i, p in enumerate(prob):
            if p[0] is not None:
                # if there are no faces in frame, a None will show up in the
                # first entry of the tuple
                face_frame.append(frames[i])

        # if no faces at all, save all selected frames
        if len(face_frame) == 0:
            logger.info(f"{clip_name} does not have face frames")
            _ = self.store_frames(
                frames, os.path.join(frame_sub_dir, clip_name.split(".")[0])
            )
        else:
            _ = self.store_frames(
                face_frame, os.path.join(
                    frame_sub_dir, clip_name.split(".")[0])
            )

    @staticmethod
    def variance_of_laplacian(image):
        """Apply Laplacian on given image
        Args:
            image (opencv image object): image object
        Returns:
            (float) Laplacian of the source image
        """
        return cv2.Laplacian(image, cv2.CV_64F).var()

    @staticmethod
    def blur_filter(img_list, thres):
        """Filter blurry images
        Args:
            img_list (listt): list image paths
            thres (float): threshold for blurriness
        Returns:
            whitelist (list): list of acceptable faces(paths)
        """
        if len(img_list) == 0:
            return []
        fm_list = []
        whitelist = []
        for face_path in img_list:
            image = cv2.imread(face_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = VideoProcessor.variance_of_laplacian(gray)
            fm_list.append(fm)
            if fm > thres:
                whitelist.append(face_path)
        return whitelist

    @staticmethod
    def pose_filter(img_list, thres, device):
        """Pose Filter(left eye, right eye, nose and two corners of the mouth)
        Args:
            img_list (listt): list image paths
            thres (float): threshold for blurriness
            device (torch.device): cpu/gpu
        Returns:
            whitelist (list): list of acceptable faces(paths)
        """
        white_list = set()
        if len(img_list) == 0:
            return []

        pil_list = []
        dist_list = []
        black_list = []
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )

        for img_path in img_list:
            pil_list.append(Image.open(img_path))
        if len(pil_list) == 0:
            return white_list

        boxes, probs, points = mtcnn.detect(pil_list, landmarks=True)

        for i, pose_set in enumerate(points):
            if pose_set is None:
                black_list.append(img_list[i])
            else:
                left_eye_x = float(pose_set[:, 0, 0][0])
                right_eye_x = float(pose_set[:, 1, 0][0])
                eye_dist = abs(left_eye_x-right_eye_x)
                dist_list.append(eye_dist)
                if eye_dist < thres:
                    black_list.append(img_list[i])
        white_list = set(img_list)-set(black_list)

        return white_list

    @staticmethod
    def array_to_pil(image_array):
        """converts np.ndarray to PIL onject
        Args:
            image_array(np.ndarray): image array
        Returns:
            pil_image (PIL object): PIL object
        """
        pil_image = Image.fromarray(np.uint8(image_array)).convert("RGB")
        return pil_image

    @staticmethod
    def show_torch_image(img, text=None):
        """converts np.ndarray to PIL onject
        Args:
            img(torch.tensor): image tensor
        Returns:
            None, display image
        """
        npimg = img.numpy()
        plt.axis("off")
        if text:
            plt.text(
                75,
                8,
                text,
                style="italic",
                fontweight="bold",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 10},
            )

        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


def video_train_test_split(video_info, raw_video_dir, video_dir_train, video_dir_val):
    """Train test split based on video clips
    Args:
        video_info (.csv): Siamese Model
        raw_video_dir (str): path to all clips
        video_dir_train (str): path to store train clips
        video_dir_val (str): path to store validation clips
    Returns:
        None, stores clips in different dirs based on train/test split
    """

    os.makedirs(video_dir_train, exist_ok=True)
    os.makedirs(video_dir_val, exist_ok=True)

    # Currently needs manually reformat given raw xlsx file to a csv file
    show_info_df = pd.read_csv(video_info)
    vid_dict = dict(Counter(list(show_info_df["Show Name"])))
    for vid_name, cts in vid_dict.items():
        vid_df = show_info_df[show_info_df["Show Name"] == vid_name]
        for it, vid_base_name in enumerate(vid_df["File Name"]):
            source = os.path.join(raw_video_dir, vid_base_name)
            if it < int(cts * 0.8):
                dest = os.path.join(video_dir_train, os.path.basename(source))
                logger.info(f"{vid_base_name} moved to train folder")
            else:
                dest = os.path.join(video_dir_val, os.path.basename(source))
                logger.info(f"{vid_base_name} moved to val folder")
            shutil.copy(source, dest)
    logger.info("Train/val video split completed")


def make_frame_ref_folder(
    train_folder_path, reference_folder_path, num_ref, batch_size, saved_model, device, num_workers, random=True, new_show=None
):
    """Pick reference frames from training set for siamese comparison,
       Currently number of ref for each show should be the same
    Args:
        train_folder_path(str): path to training frames
        reference_folder_path(str): path to store reference frames
        num_ref(int): number of references frames for comparison
        batch_size(int): number of samples in a batch
        saved_model(torch.nn object): Siamese Model
        device(torch.device): cpu/gpu
        num_workers(int): number of workers for toch dataloader
        random(bool): random selection num_ref training frames as refrence frames if True
        new_show(str): name of unseen/new show, only used when updating dataset
    Returns:
        None
    """

    if random:
        if new_show:
            show_loc = new_show
        else:
            show_loc = "*"

        for class_dir in glob.glob(os.path.join(train_folder_path, show_loc)):

            class_frames_path_list = glob.glob(
                os.path.join(class_dir, "*.jpg"))
            # min(num_ref, len(class_frames_path_list)) deals with edge cases where number of frames are less then user defined
            random_index_set = np.random.choice(
                range(len(class_frames_path_list)), min(num_ref, len(class_frames_path_list)), replace=False
            )

            for rand_i in random_index_set:

                source = class_frames_path_list[rand_i]
                class_dir = source.split("/")[-2]
                frame_name = source.split("/")[-1]
                dest = os.path.join(reference_folder_path,
                                    class_dir, frame_name)

                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(source, dest)
    else:
        # use intelligent selection
        ref_path_dict = evaluate_all_frames(batch_size, frame_path=train_folder_path,
                                            saved_model=saved_model, device=device,
                                            num_workers=num_workers, seed_worker=seed_worker,
                                            generator=g, num_ref=num_ref)

        if new_show:
            for source in ref_path_dict[new_show]:
                dest = os.path.join(reference_folder_path,
                                    new_show, os.path.basename(source))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(source, dest)
        else:
            for class_dir, frame_name_list in ref_path_dict.items():
                for source in frame_name_list:
                    dest = os.path.join(reference_folder_path,
                                        class_dir, os.path.basename(source))
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy(source, dest)


def save_reference_image_embeddings(model, device, ref_loader, reference_emb_path):
    """Save embeddings from reference frames for siamese comparison
    Args:
        model (torch.nn object): Siamese Model
        device (torch.device object): cpu/GPU
        ref_loader (torch dataloader object): reference frame dataloader
        reference_emb_path (str): path to save reference embeddings
    Returns:
        None
    """

    label_list = []
    ebd_list = []
    path_list = []
    for target_inputs, target_labels, target_paths in ref_loader:
        target_inputs = target_inputs.to(device)
        target_labels = target_labels.to(device)
        label_list.extend(target_labels.cpu().numpy())
        with torch.set_grad_enabled(False):
            target_output1 = model(target_inputs)
            ebd_list.extend(target_output1.cpu().numpy())
            path_list.extend(target_paths)

    label_emb_tup = (label_list, ebd_list, path_list)
    os.makedirs(os.path.dirname(reference_emb_path), exist_ok=True)
    with open(reference_emb_path, "wb") as f:
        pickle.dump(label_emb_tup, f)
