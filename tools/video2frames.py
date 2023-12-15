import os
import sys
import cv2
import moviepy.editor as moviepy
from tqdm import tqdm

if len(sys.argv) < 1:
    print ("Please provide arkit data folder")
    sys.exit(1)
# Reading the data
data_path = sys.argv[1]
video_file_name = sys.argv[2]

video_path = data_path + "/" + video_file_name
output_folder = data_path + "/images"

ext = video_file_name.split('.')[1]
if ext.lower() != 'mp4':
    clip = moviepy.VideoFileClip(video_path)
    clip.write_videofile(video_path.split('.')[0] + '.mp4')


vidcap = cv2.VideoCapture(video_path.split('.')[0] + '.mp4')

if not os.path.isdir(output_folder):
    os.makedirs(output_folder) 

length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#success,image = vidcap.read()

for count in tqdm(range(length)):
  success,image = vidcap.read()
  if image is None or not success:
    print("Failed reading: ", "frame%d.jpg" % count, "\n")
    break
  image_path = os.path.join(output_folder, "frame%d.jpg" % count)
  cv2.imwrite(image_path, image)     # save frame as JPEG file
  