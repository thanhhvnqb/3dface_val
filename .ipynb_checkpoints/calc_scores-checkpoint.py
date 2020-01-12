import time
import sys
import os
from os import listdir, makedirs
from os.path import isfile, join, isdir
import pickle
import gzip

import numpy as np
import torchfile
import matplotlib.pylab as plt


def getBBsize(points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    bbsize = maxs-mins
    normby = np.sqrt(bbsize[0] * bbsize[1])
    return np.sqrt(bbsize[0] * bbsize[1])
#     return np.linalg.norm(bbsize)

def calcDistance(preds, gt_pts):
    bbsize = getBBsize(gt_pts)
    final_dist = None
    min_dist = None
    for pred in preds:
        if min_dist is None or min_dist > np.linalg.norm(pred - gt_pts) / bbsize:
            min_dist = np.linalg.norm(pred - gt_pts) / bbsize
            final_dist = np.linalg.norm(pred - gt_pts, axis=1) / bbsize
    return final_dist


def calculateMetrics(dists):
    print(dists.shape)
    errors = np.mean(dists,axis=0)
    print(np.mean(errors))
    print(errors.shape)
    axes1 = np.linspace(0,1,1000)
    axes2 = np.zeros(1000)
    print(errors.shape[0])
    for i in range(1000):
        axes2[i] = (errors<axes1[i]).sum()/float(errors.shape[0])
    # plt.xlim(0,7)
    # plt.ylim(0,100)
    # plt.yticks(np.arange(0,110,10))
    # plt.xticks(np.arange(0,8,1))
    # plt.grid()
    # plt.title('NME (%)', fontsize=20)
    # plt.xlabel('NME (%)', fontsize=16)
    # plt.ylabel('Test Images (%)', fontsize=16)
    # plt.plot(axes1*100,axes2*100,'b-',label='FAN (Ours)',lw=3)
    # plt.legend(loc=4, fontsize=16)
    # plt.show()
    auc = np.sum(axes2[:70])/70
    print('AUC: ', auc)
    return auc

out_pred_folder = '/home/thanh/DATA/workspace/face2d/out/evaluation_pts/aug_blur_jscl/'
db_eval = {
        "Menpo-3D": '/home/thanh/DATA/Dataset/face pose/face 3d landmarks/LS3D-W/Menpo-3D/',
        '300W': "/home/thanh/DATA/Dataset/face pose/face 3d landmarks/LS3D-W/300W-Testset-3D/",
        '300VW-A': "/home/thanh/DATA/Dataset/face pose/face 3d landmarks/LS3D-W/300VW-3D/CatA/",
        '300VW-B': "/home/thanh/DATA/Dataset/face pose/face 3d landmarks/LS3D-W/300VW-3D/CatB/",
        '300VW-C': "/home/thanh/DATA/Dataset/face pose/face 3d landmarks/LS3D-W/300VW-3D/CatC/",
        '300VW-Trainset': "/home/thanh/DATA/Dataset/face pose/face 3d landmarks/LS3D-W/300VW-3D/Trainset/"}
key = "Menpo-3D"
gt_folder = db_eval[key]
lepoch = [60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]
lepoch = [60, 65]
auc_ep = dict()
for epoch in lepoch:
    pred_folder = join(out_pred_folder, key) + "/" + str(epoch) + "/"
    suffix = ['.jpg', '.png']
    files = [os.path.join(dirpath, filename) for dirpath, dirnames, filenames in os.walk(gt_folder)
                                    for filename in filenames if filename[-4:] in suffix]
    files.sort()
    print(pred_folder + "all_pred.zip")
    try:
        with gzip.open(pred_folder + "all_pred.zip", 'rb') as f:
            buffer = f.read()
        full_landmarks = pickle.loads(buffer)
    except Exception:
        print("Do not have file.")

    dists = None
    for nframe, imfile in enumerate(files):
        print("Process %d/%d. " % (nframe + 1, len(files)), end='')
        pts_file = imfile
        for file_type in suffix:
            pts_file = pts_file.replace(file_type, '.t7')
        pts_file = join(gt_folder, pts_file)
        landmarks = full_landmarks[imfile]
        try:
            groundtruth = torchfile.load(pts_file)
        except KeyError:
            print("Do not have groundtruth.")
            continue
    #     try:
    #         with gzip.open(pred_folder + imfile + ".zip", 'rb') as f:
    #             buffer = f.read()
    #         map_data = pickle.loads(buffer)
    #     except Exception:
    #         print("Do not have file.")
    #         continue
        # if not landmarks:
        #     print("List is empty.")
        else:
            dist = calcDistance(landmarks, groundtruth)
            dist = dist.reshape((68,1))
            if dists is None:
                dists = dist
            else:
                dists = np.append(dists, dist, axis=1)
            print("Done.")

    auc_ep[epoch] = calculateMetrics(dists)

header = ''
value = ''

for key, val in auc_ep.items():
    header += '%5d' % key
    val += '%1.3f' % val
