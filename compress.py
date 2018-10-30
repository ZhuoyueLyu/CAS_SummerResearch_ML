# 导入pillow
from PIL import Image
import os
path = "/Users/zhuoyuelyu/Documents/Courses/ML_Material/Data/Road_Sign/Original"
path_des = "/Users/zhuoyuelyu/Documents/Courses/ML_Material/Data/Road_Sign/Original/compressed_30*30"
# 加载原始图片
for f in os.popen('ls ' + path):
    if "Original" in f:
        new_path = (path + "/" + f).replace("\n", "")
        img = Image.open(new_path)
        img.thumbnail((33, 33))
        new_save_path = (path_des + "/" + f).replace("\n", "")
        img.save(new_save_path)

