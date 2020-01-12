"""Run Face 2D System."""
import math
import time
import os
import sys
from os import listdir, makedirs
from os.path import isfile, join, isdir
import pickle
import gzip

import mxnet as mx
import numpy as np
import scipy.io as sio
import cv2 as cv
import pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
import matplotlib.patches as patches
import torchfile

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(cur_path, '../lib'))
from processing import bbox_pred, clip_boxes, nms

from visualization import put_2dface
from core_module.mutablemod import MutableModule

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = "0"


class Predictor(object):
    def __init__(self,
                 symbol,
                 data_names,
                 label_names,
                 context=mx.cpu(),
                 max_data_shapes=None,
                 provide_data=None,
                 provide_label=None,
                 arg_params=None,
                 aux_params=None):
        self._mod = MutableModule(
            symbol, data_names, label_names, context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)
        # print(self._mod.output_shapes)

    def forward_names(self, data_batch):
        self._mod.forward(data_batch)
        return [
            dict(zip(self._mod.output_names, _))
            for _ in zip(*self._mod.get_outputs(merge_multi_context=False))
        ]

    def forward(self, data_batch):
        self._mod.forward(data_batch)
        return self._mod.get_outputs(merge_multi_context=False)


def get_image_face(image, points, box_size=256):
    max_xy = np.around(np.max(points, axis=0)).astype(np.int)
    min_xy = np.around(np.min(points, axis=0)).astype(np.int)
    d = max(1, 0.1 * max(max_xy - min_xy))
    face = np.zeros((4, ), dtype=np.int)
    pad = np.zeros((4, ), dtype=np.int)
    face[0] = min_xy[0] - d
    face[1] = min_xy[1] - d
    face[2] = max_xy[0] + d
    face[3] = max_xy[1] + d
    min0 = int(max(0, face[0]))
    min1 = int(max(0, face[1]))
    max0 = int(min(image.shape[1] - 1, face[2]))
    max1 = int(min(image.shape[0] - 1, face[3]))
    pad[0] = min1 - face[1]
    pad[1] = min0 - face[0]
    pad[2] = face[3] - max1
    pad[3] = face[2] - max0
    d = max(face[2] - face[0], face[3] - face[1])
    tmp_d = (d - face[3] + face[1])
    pad[0] = pad[0] + tmp_d // 2
    pad[2] = pad[2] + tmp_d - tmp_d // 2
    face[1] = face[1] - tmp_d // 2
    face[3] = face[3] + tmp_d - tmp_d // 2
    tmp_d = (d - face[2] + face[0])
    pad[1] = pad[1] + tmp_d // 2
    pad[3] = pad[3] + tmp_d - tmp_d // 2
    face[0] = face[0] - tmp_d // 2
    face[2] = face[2] + tmp_d - tmp_d // 2

    new_img = image[min1:max1 + 1, min0:max0 + 1, :]
    new_points = points - face[:2]
    # print('img_shape', new_img.shape)
    # print('pad', pad)
    # print(new_img.shape[0] + pad[0] + pad[2], new_img.shape[1] + pad[1] + pad[3])
    # print('face', face, face[2] - face[0] + 1, face[3] - face[1] + 1)
    face_width = face[2] - face[0] + 1

    try:
        scale = box_size / face_width
        new_img = cv.resize(new_img, (0, 0), fx=scale, fy=scale)
        new_points = new_points * scale
    except Exception as e:
        print("Exception:", e)
    pad = (pad * scale).astype(np.int)
    pad[2] = int(max(0, box_size - pad[0] - new_img.shape[0]))
    pad[3] = int(max(0, box_size - pad[1] - new_img.shape[1]))
    new_img = np.pad(
        new_img, ((pad[0], pad[2]), (pad[1], pad[3]), (0, 0)), 'constant', constant_values=(0))
    return (new_img, face, scale, pad, new_points)


def get_landmarks(cmodel, params, image, gt_pts):
    stride = params['stride']
    box_size = params['imgsize']
    imageToTest, face, scale, pad, new_gt_pts = get_image_face(image, gt_pts, box_size)
#     face[0] = face[0] - pad[1] / scale
#     face[1] = face[1] - pad[0] / scale
    testimage = np.moveaxis(imageToTest, 2, 0)

    onedata = mx.io.DataBatch(
        data=[[mx.nd.array([testimage[:, :, :]])]],
        label=[],
        pad=0,
        index=0,
        provide_data=[[('data', (1, 3, box_size, box_size))]],
        provide_label=[None])
    tic = time.time()
    try:
        result = cmodel.forward(onedata)
        # print(len(result))
        heatmap = np.moveaxis(result[-1][0].asnumpy()[0], 0, -1)

        heatmap = cv.resize(
            heatmap, (0, 0), fx=stride / scale, fy=stride / scale, interpolation=cv.INTER_CUBIC)
    except Exception as e:
        print("Exception:", e)
        return None, None
    run_time = time.time() - tic
    points = np.zeros((68, 2))
    for i in range(68):
        x = heatmap[:, :, i]
        point = np.unravel_index(np.argmax(x), x.shape)
        points[i, :] = [face[0] + point[1], face[1] + point[0]]
    return points, face, run_time


is_running = True
def get_params():
    """Get params to run system."""
    params = dict()
    params['imgsize'] = 256
    params['stride'] = 4
    params['context'] = mx.gpu(0)
    params['model'] = 'model/HGFPN'
    params['out_pred'] = 'out/'
    params['save_fig'] = False
    return params


def run_evaluation(input_folder, output_folder, params, is_mat=False):
    """Run CPM with input as an image."""
    suffix = ['.jpg', '.png']
    # files = [f for f in listdir(input_folder) if isfile(join(input_folder, f)) and f[-4:] in suffix]
    files = [
        os.path.join(dirpath, filename) for dirpath, dirnames, filenames in os.walk(input_folder)
        for filename in filenames if filename[-4:] in suffix
    ]
    files.sort()
    save_fig = params['save_fig']
    if save_fig:
        fig = None
    try:
        sym, arg_params, aux_params = mx.model.load_checkpoint(params['model'], 0)
    except Exception as e:
        print(e)
        return
    data_names = ['data']
    label_names = []
    max_data_shape = [[('data', (1, 3, 256, 256))]]
    provide_data = [[('data', (1, 3, 256, 256))]]
    provide_label = [None]
    predictor = Predictor(
        sym,
        data_names,
        label_names,
        context=[params['context']],
        max_data_shapes=max_data_shape,
        provide_data=provide_data,
        provide_label=provide_label,
        arg_params=arg_params,
        aux_params=aux_params)
    full_landmarks = dict()
    for nframe, imfile in enumerate(files):
        nframe += 1
        if not is_running:
            break
        tic = time.time()
        print("Frame: %5d/%5d" % (nframe, len(files)), end='')
        output_file = imfile.replace(input_folder, output_folder)
        out_dir = os.path.dirname(output_file)
        if not isdir(out_dir):
            makedirs(out_dir)
        if isfile(output_file + ".zip"):
            try:
                with gzip.open(output_file + ".zip", 'rb') as f:
                    buffer = f.read()
                map_data = pickle.loads(buffer)
                full_landmarks[imfile] = map_data
                print(". Already done.")
                continue
            except Exception as e:
                pass
        image = cv.imread(imfile)
        try:
            if is_mat:
                pts_file = imfile[:-4] + '.mat'
                gt = sio.loadmat(pts_file)['pt3d_68']
                gt_pts = gt.T[:,:2]
            else:
                pts_file = imfile[:-4] + '.t7'
                gt_pts = torchfile.load(pts_file)
        except Exception as e:
            print("\nException", e)
            print("Could not find groundtruth file of %s" % imfile)
            continue
        
        print(", load GT: %.3fs" % (time.time() - tic), end='')

        points, face, run_time = get_landmarks(cmodel=predictor, params=params, image=image, gt_pts=gt_pts)
        if points is None:
            continue
        print(", landmarks: %.3fs" % run_time, end='')
        full_landmarks[imfile] = points
        if save_fig:
            img2 = image[:, :, ::-1]
            if fig is None:
                # fig = plt.figure(figsize=(19.2, 10.8))
                def handle_close(evt):
                    global is_running
                    is_running = False
                    # plt.close(evt.canvas.figure)
                    # sys.exit('Closed Figure!')

                fig = plt.figure(figsize=(19.2, 10.8))
                fig.canvas.mpl_connect('close_event', handle_close)
                ax = fig.add_subplot(111)
                ax.axis('off')
            else:
                ax.clear()
                ax.axis('off')
            ax.imshow(img2)
            put_2dface(ax=ax, face=None, points=points, color='b')
            put_2dface(ax=ax, face=None, points=gt_pts, color='r')
            plt.pause(.0001)
            plt.draw()
            output_file = imfile.replace(input_folder, join(output_folder,"figs"))
            fig.savefig(fname=output_file + ".pdf", transparent=True, bbox_inches='tight', pad_inches=0)
        zip_file = gzip.GzipFile(output_file + ".zip", 'wb')
        zip_file.write(pickle.dumps(points, pickle.HIGHEST_PROTOCOL))
        zip_file.close()
        print(", Time: %.3fs" % (time.time() - tic))
    zip_file = gzip.GzipFile(join(output_folder, "all_pred") + ".zip", 'wb')
    zip_file.write(pickle.dumps(full_landmarks, pickle.HIGHEST_PROTOCOL))
    zip_file.close()


if __name__ == "__main__":
    params = get_params()
    db_eval = {
        "Menpo-3D": 'data/Menpo-3D/',
        '300W': "data/300W-Testset-3D/",
        'AFLW2000-3D-Reannotated': "data/AFLW2000-3D-Reannotated/",
        'AFLW2000-3D': "data/AFLW2000/",
        '300VW-A': "data/CatA/",
        '300VW-B': "data/CatB/",
        '300VW-C': "data/CatC/",
    }
    for key, folder in db_eval.items():
        print("Dataset:", key)
        run_evaluation(folder, join(params['out_pred'], key) + "/", params, is_mat=True if key=='AFLW2000-3D' else False)
        # run_eval_detect(folder, params)
