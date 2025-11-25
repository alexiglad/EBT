import argparse
import os
import shutil
import getpass
import json
import subprocess
import cv2

VIDEO_EXTENSIONS = [".mov", ".mp4", ".webm", ".avi"]
def get_video_files(dataset_path):
    video_files = []
    for root, dirs, files in os.walk(dataset_path):
        path = root.split('/')
        for file in files:
            full_path = '/'.join(path+[file])
            is_video = any(ext in full_path.lower() for ext in VIDEO_EXTENSIONS)
            if is_video: video_files.append(full_path)
    return video_files

def log_corrupt_files(video_files, log_path):
    corrupt_files = []
    for fp in video_files:
        try:
            length = get_length(fp)
            assert length > 0

            cap = cv2.VideoCapture(fp)
            assert cap.isOpened()
            cap.release()
        except Exception as e:
            #print(e)
            fp_components = os.path.normpath(fp).split('/')
            #e.g. train/vid_1.mp4, val/vid_1.mp4
            new_fp = '/'.join(fp_components[-2:]) 
            print(new_fp)

            corrupt_files.append(new_fp)
        
    with open(log_path, 'a') as f:
        f.write('\n'.join(corrupt_files))

def get_length(filename):
    global ffprobe
    command = ffprobe + ' -v quiet -print_format json -show_format "{}"'.format(filename)
    data = json.loads(subprocess.check_output(command, shell=True))
    return float(data['format']['duration'])

def main():
    global ffprobe

    current_user = getpass.getuser()
    # FFPROBE SETUP
    # First, check if ffprobe is in path
    if shutil.which("ffprobe") is not None:
        ffprobe = shutil.which("ffprobe")
    # If not, check if environment variable was set
    elif os.getenv("FFPROBE_PATH") is not None:
        ffprobe = os.getenv("FFPROBE_PATH")
    else:
        raise ValueError("no ffprobe found")
    assert os.path.exists(ffprobe)

    parser = argparse.ArgumentParser(
                        prog='find_corrupt_files',
                        description='Given a dataset directory, finds all corrupt video files and appends them to log_path')
    parser.add_argument("dataset_path", help="path to dataset")
    parser.add_argument("log_path", help="text file to log the corrupt files to")

    args = parser.parse_args()
    dataset_path = args.dataset_path
    log_path = args.log_path

    video_files = get_video_files(dataset_path)
    log_corrupt_files(video_files, log_path)



# traverse root directory, and list directories as dirs and files as files


    
if __name__ == '__main__':
    main()