# import deeplabcut as dlc
import deeplabcutcore as dlccore
import os
import tools

if __name__ == '__main__':
    orientation = 'FRONT'
    parent_directory = r'C:\Users\Peter\Desktop\DLC'
    project_path = os.path.join(parent_directory,
                                f'M2_M4_M5_M9_FRONT-pw-2021-03-08')
    config_path = os.path.join(project_path, 'config.yaml')
    video_path = os.path.join(project_path, 'videos')

    # # Get images
    # selection_dict = {
    #     r'C:\Users\Peter\Desktop\DLC\DATA\M9\2021.03.08':
    #         list(range(10, 20)) + list(range(80, 90)),
    # }
    # image_master_dir = os.path.join(parent_directory, 'DATA', 'M9')
    # image_dirs = [os.path.join(image_master_dir, x) for x in os.listdir(image_master_dir)]
    # image_dirs = [x for x in image_dirs if os.path.isdir(x)]
    # dict_of_avis = {x: get_files(x, (orientation + '.avi')) for x in image_dirs}
    # for k, v in dict_of_avis.items():
    #     v = np.array(v)
    #     if k in selection_dict.keys():
    #         dict_of_avis[k] = list(v[selection_dict[k]])
    #     else:
    #         dict_of_avis[k] = []
    # print_dict(dict_of_avis, length=True)
    # list_of_avis = list(itertools.chain(*dict_of_avis.values()))
    # pprint.pprint(list_of_avis)

    # video_dir = r'C:\Users\Peter\Desktop\anipose\reach-unfilled\date' \
    #             r'\calibration'
    # videos = ['DATA_CALIBRATION_00018_CAM_0.avi',
    #           'DATA_CALIBRATION_00018_CAM_1.avi']
    # list_of_avis = [os.path.join(video_dir, x) for x in videos]
    #
    # # Create project
    # config_path = dlc.create_new_project(
    #     'calibration',
    #     'pw',
    #     videos=list_of_avis,
    #     copy_videos=True,
    #     working_directory=parent_directory)

    # Label frames
    # dlc.extract_frames(config_path,
    #                   mode='manual',
    #                   algo='kmeans',
    #                   crop=False,
    #                   userfeedback=False)
    # dlc.label_frames(config_path)
    # dlc.SkeletonBuilder(config_path)
    # dlc.check_labels(config_path)

    # Train network
    # dlccore.create_training_dataset(config_path, augmenter_type='imgaug')
    dlccore.train_network(config_path,
                          displayiters=1000,
                          saveiters=10000,
                          )

    # Plot
    # dlc.analyze_videos(config_path,
    #                    videos=[video_path],
    #                    videotype='avi',
    #                    )
    # dlc.filterpredictions(config_path, [video_path], videotype='avi')
    # dlc.create_labeled_video(config_path,
    #                          [video_path],
    #                          videotype='avi',
    #                          filtered=False)

    ## Refine
    # new_videos = [
        # r'M9_2021.03.11_00010_CAM0.avi',
        # r'M9_2021.03.11_00045_CAM0.avi',
    #     r'M9_2021.03.11_00033_CAM0.avi',
    # ]
    # new_video_path = r'C:\Users\Peter\Desktop\DATA\M9\2021.03.11'
    # new_videos = [os.path.join(new_video_path, x) for x in new_videos]
    # dlc.add_new_videos(config_path,
    #                    new_videos,
    #                    copy_videos=True
    #                    )

    # dlc.extract_outlier_frames(config_path,
    #                            new_videos,
    #                            outlieralgorithm='manual')
    # dlc.refine_labels(config_path)
    # dlc.merge_datasets(config_path)
