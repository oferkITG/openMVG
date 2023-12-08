import os
import cv2
data_path = "/home/gabi/Work/october23/tunnels/DATA/2023-11-27-20231207T082758Z-001/2023-11-27T90-37-23"
video_path = data_path + "/Frames.avi"
output_folder = data_path + "/images"
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
  print("frame: ", count, "Saved image: ", image_path, "success: ", success, "\n")
  count += 1