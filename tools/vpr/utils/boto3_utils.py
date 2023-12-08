import boto3
import os
import torch
import sys


s3_resource = boto3.resource('s3')
bucket_name = 'artifacts-9900'
bucket = s3_resource.Bucket(bucket_name)
'''
for obj in bucket.objects.filter(Prefix='assets/weights'):
    if obj.key == "assets/weights/pretrained_vgg16_pitts30k_netvlad_from_matlab.pth":
        bucket.download_file(obj.key, os.path.join("../weights/",os.path.basename(obj.key)))
    

for obj in bucket.objects.filter(Prefix='assets/vpr'):
    s3_dirname = os.path.dirname(obj.key)
    print(s3_dirname)
    print(obj.key)
    local_dir_path = os.path.join("../data",s3_dirname)
    os.makedirs(local_dir_path, exist_ok=True)

    s3_filename = obj.key
    if obj.key != s3_dirname+"/":
        filepath = os.path.join(local_dir_path, os.path.basename(s3_filename))
        print("{}->{}".format(obj.key, filepath))
        bucket.download_file(obj.key, filepath)

s3 = boto3.client('s3')
with open("../temp.jpg", "rb") as f:
    s3.upload_fileobj(f, bucket_name, "assets/vpr/temp.jpg")
'''
# copy isd folder for a specific set of images
dest = "../isd/"
for obj in bucket.objects.filter(Prefix='assets/vpr/isd'):
    if os.path.basename(obj.key).startswith("AAE"):
        bucket.download_file(obj.key, os.path.join(dest,os.path.basename(obj.key)))


'''
import os
l = [f for f in os.listdir("/mnt/rawdata/")]
print(l)
'''





