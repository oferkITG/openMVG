import os
import cv2
data_path = "/october23/tunnels/DATA/BLUR"
video_path = data_path + "/test_video_with_blur.mp4"
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