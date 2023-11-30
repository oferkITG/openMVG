import os
import cv2
video_path = r"C:\work_itg\data\2023-11027 Aaron iphone open src app\2023-11-27T09-37-23\Frames.m4v"
output_folder = r"C:\work_itg\data\2023-11027 Aaron iphone open src app\2023-11-27T09-37-23\images"
vidcap = cv2.VideoCapture(video_path)

if not os.path.isdir(output_folder):
    os.makedirs(output_folder) 

success,image = vidcap.read()
count = 0
success = True

while success:
  success,image = vidcap.read()
  if image is None:
     break
  image_path = os.path.join(output_folder, "frame%d.jpg" % count)
  cv2.imwrite(image_path, image)     # save frame as JPEG file
  count += 1