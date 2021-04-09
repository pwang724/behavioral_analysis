import os
import shutil

def batch_rename(d, from_to_dict):
    for k, v in from_to_dict.items():
        for root, dirs, files in os.walk(d):
            for file in files:
                if k in file:
                    new_filename = file.replace(k, v)
                    os.rename(os.path.join(root, file),
                              os.path.join(root, new_filename))


def batch_delete(d, delete_list):
    for root, dirs, files in os.walk(d):
        for file in files:
            for k in delete_list:
                if k in file:
                    os.remove(os.path.join(root, file))


def batch_swap(d):
    for root, dirs, files in os.walk(d):
        for file in files:
            if 'FRONT.avi' in file:
                new_filename = file[6:-9] + file[:6] + file[-9:]
                print(new_filename)
            elif 'SIDE.avi' in file:
                new_filename = file[6:-8] + file[:6] + file[-8:]
                print(new_filename)
            elif '.npy' in file:
                new_filename = file[6:-4] + file[5] + file[:5] + file[-4:]
                print(new_filename)
            os.rename(os.path.join(root, file),
                      os.path.join(root, new_filename))


if __name__ == "__main__":
    d = r'C:\Users\Peter\Desktop\DATA\M9\2021.03.11\ANALYZED\POSE_2D'
    from_to_dict = {
        # 'CAM0': 'FRONT',
        # 'CAM1': 'SIDE'
        # '2921':'2021',
        # 'SIDE':'CAM_1'
    }
    # batch_rename(d, from_to_dict)
    # batch_delete(d, ['CAM0'])
    # batch_swap(d)
    # batch_rename(d, from_to_dict)

    # batch_delete(d, ['FRONT.tif', 'SIDE.tif'])


    d = r'C:\Users\Peter\Desktop\DATA'
    dout = r'C:\Users\Peter\Desktop\ANALYZED'
    for root, dirs, files in os.walk(dout):
        for dir in dirs:
            if dir == 'ANALYZED':
                # path, mouse = os.path.split(root)
                # path, date = os.path.split(path)
                # newroot = os.path.join(dout, date, mouse)
                # os.makedirs(newroot, exist_ok=True)
                # shutil.move(os.path.join(root, dir),
                #             os.path.join(newroot, dir))
                # print(f'from: {os.path.join(root, dir)} '
                #       f'to: {os.path.join(newroot, dir)}')
                for f in os.listdir(os.path.join(root, dir)):
                    old = os.path.join(root, dir, f)
                    new = os.path.join(root, f)
                    shutil.move(old, new)
                    print(old, new)
                os.rmdir(os.path.join(root, dir))



