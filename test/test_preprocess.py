import os
import shutil
import torch
from facenet_pytorch import MTCNN
from src.preprocess import video_processor

def test_get_face_frames():
    video_clip_path = "test/compressed_ff.mov"
    clip_name = video_clip_path.split("/")[-1]
    frame_sub_dir = "test/ff_frame"
    n_secs = 10
    device =  torch.device("cpu")

    if os.path.exists(frame_sub_dir):
        shutil.rmtree(frame_sub_dir)
    if not os.path.exists(frame_sub_dir):
        os.makedirs(frame_sub_dir)  
    
    face_model = MTCNN(image_size=160, margin=0, min_face_size=20, keep_all=False,
                          thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                          device=device)

    frames = video_processor().get_frames_per_n_seconds(video_clip_path, n_secs, 
                                                        black_frame_thres = 10, crop = None)
    
    video_processor().get_face_frames(clip_name, frame_sub_dir, frames, face_model)
    
    results_dir = sorted(os.listdir(frame_sub_dir))

    assert len(results_dir)==1
    assert results_dir[0]=="compressed_ff_frame0.jpg"
    shutil.rmtree(frame_sub_dir)



def test_get_frames_per_n_seconds():
    video_clip_path = "test/compressed_ff.mov"
    clip_name = video_clip_path.split("/")[-1]
    frame_sub_dir = "test/ff_frame"
    n_secs = 10
    device =  torch.device("cpu")

    if os.path.exists(frame_sub_dir):
        shutil.rmtree(frame_sub_dir)
    if not os.path.exists(frame_sub_dir):
        os.makedirs(frame_sub_dir)  

    frames = video_processor().get_frames_per_n_seconds(video_clip_path, n_secs, 
                                                        black_frame_thres = 10, crop = None)

    _ = video_processor().store_frames(frames, os.path.join(frame_sub_dir, clip_name.split(".")[0]))

    results_dir = sorted(os.listdir(frame_sub_dir))

    assert len(results_dir)==2
    assert results_dir[0]=="compressed_ff_frame0.jpg"
    assert results_dir[1]=="compressed_ff_frame1.jpg"
    shutil.rmtree(frame_sub_dir)
