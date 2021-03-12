import os
import glob
import imageio

d = r'C:\Users\Peter\Desktop\DLC\new_position-pw-2021-03-01\videos_pred_dlc'

def mp4_to_avi(d, delete_old=True):
    mp4_files = sorted(glob.glob(d + '/*.mp4'))

    for mp4 in mp4_files:
        reader = imageio.get_reader(mp4)
        fps = reader.get_meta_data()['fps']
        writer = imageio.get_writer(os.path.splitext(mp4)[0] + '.avi',
                                    fps=fps,
                                    codec='mjpeg',
                                    pixelformat='yuvj420p',
                                   quality=10)

        for im in reader:
            writer.append_data(im[:, :, :])
        writer.close()

    if delete_old:
        for mp4 in mp4_files:
            os.remove(mp4)