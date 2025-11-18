#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.getcwd()))

import pickle
import onnxruntime as ort
from onnx_infer.utils import letterbox_image, get_table_line, adjust_lines, line_to_line
import onnx
import numpy as np

class WrapInferenceSession:
    def __init__(self, onnx_bytes):
        self.sess = ort.InferenceSession(onnx_bytes.SerializeToString())
        self.onnx_bytes = onnx_bytes
        print(self.sess.get_outputs()[0].name)
        print(self.sess.get_inputs()[0].name)

    def run(self, *args):
        return self.sess.run(*args)

    def __getstate__(self):
        return {'onnx_bytes': self.onnx_bytes}

    def __setstate__(self, values):
        self.onnx_bytes = values['onnx_bytes']
        self.sess = ort.InferenceSession(self.onnx_bytes.SerializeToString())

class TableLineModelLoad:
    def __init__(self, model_path):
        self.model_path = model_path
    def test_model(self):
        onnx_bytes = onnx.load_model(self.model_path)
        wrap = WrapInferenceSession(onnx_bytes=onnx_bytes)
        return pickle.dumps(wrap)

def load_table_wire_line_model(onnx_model_path, cuda=False):
    print('load table wire line model ...')
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider'] # OpenVINOExecutionProvider, CPUExecutionProvider
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    return session

def table_line(table_line_model, img, size=(640, 640), hprob=0.5, vprob=0.5, row=50, col=30, alph=15):
    sizew, sizeh = size
    inputBlob, fx, fy = letterbox_image(img[..., ::-1], (sizew, sizeh))
    pred = table_line_model.run([table_line_model.get_outputs()[0].name], {table_line_model.get_inputs()[0].name: np.array([np.array(inputBlob) / 255.0], dtype = np.float32)})[0]

    pred = pred[0]
    vpred = pred[..., 1] > vprob
    hpred = pred[..., 0] > hprob

    vpred = vpred.astype(int)
    hpred = hpred.astype(int)
    colboxes = get_table_line(vpred, axis=1, lineW=col)
    rowboxes = get_table_line(hpred, axis=0, lineW=row)
    ccolbox = []
    crowlbox = []
    if len(rowboxes) > 0:
        rowboxes = np.array(rowboxes)
        rowboxes[:, [0, 2]] = rowboxes[:, [0, 2]] / fx
        rowboxes[:, [1, 3]] = rowboxes[:, [1, 3]] / fy
        xmin = rowboxes[:, [0, 2]].min()
        xmax = rowboxes[:, [0, 2]].max()
        ymin = rowboxes[:, [1, 3]].min()
        ymax = rowboxes[:, [1, 3]].max()
        ccolbox = [[xmin, ymin, xmin, ymax], [xmax, ymin, xmax, ymax]]
        rowboxes = rowboxes.tolist()

    if len(colboxes) > 0:
        colboxes = np.array(colboxes)
        colboxes[:, [0, 2]] = colboxes[:, [0, 2]] / fx
        colboxes[:, [1, 3]] = colboxes[:, [1, 3]] / fy

        xmin = colboxes[:, [0, 2]].min()
        xmax = colboxes[:, [0, 2]].max()
        ymin = colboxes[:, [1, 3]].min()
        ymax = colboxes[:, [1, 3]].max()
        colboxes = colboxes.tolist()
        crowlbox = [[xmin, ymin, xmax, ymin], [xmin, ymax, xmax, ymax]]

    rowboxes += crowlbox
    colboxes += ccolbox

    rboxes_row_, rboxes_col_ = adjust_lines(rowboxes, colboxes, alph=alph)
    rowboxes += rboxes_row_
    colboxes += rboxes_col_
    nrow = len(rowboxes)
    ncol = len(colboxes)
    for i in range(nrow):
        for j in range(ncol):
            rowboxes[i] = line_to_line(rowboxes[i], colboxes[j], 10)
            colboxes[j] = line_to_line(colboxes[j], rowboxes[i], 10)

    return rowboxes, colboxes

