import tifffile
import os
import imageio
from tqdm import tqdm

d = r'C:\Users\pywang7\Desktop\DATA\M7\2021-02-09'
FOLDER_ANALYZED = r'ANALYZED'
FOLDER_MOVIES = os.path.join(FOLDER_ANALYZED, 'MOVIE')
FOLDER_TIFS = os.path.join(FOLDER_ANALYZED, 'TIFFS')
ORIGINAL_FPS = 200
NEW_FPS = 100
QUALITY = 8 ## 10 is highest

tifs = []
for root, dirs, files in os.walk(d):
    for file in files:
        if file.endswith('.tif'):
            tifs.append(os.path.join(root, file))

for tif in tqdm(tifs, position=0, leave=True):
    path, name = os.path.split(tif)
    arr = tifffile.imread(tif)
    os.makedirs(os.path.join(path, FOLDER_MOVIES), exist_ok=True)
    new_pathname = os.path.join(path, FOLDER_MOVIES, name[:-4] + '.avi')
    imageio.mimwrite(new_pathname,
                     arr,
                     fps=NEW_FPS,
                     codec='mjpeg',
                     quality=QUALITY)

    # new_pathname = os.path.join(path, FOLDER_MOVIES, name[:-4] + '.mp4')
    # imageio.mimwrite(new_pathname,
    #                  arr,
    #                  fps=NEW_FPS,
    #                  codec='libx264',
    #                  quality=QUALITY)
