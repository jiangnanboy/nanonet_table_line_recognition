#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from onnx_infer.table_build import tableBuild, to_excel

sys.path.insert(0, os.path.dirname(os.getcwd()))

import cv2
import numpy as np
from PIL import Image
import argparse
from onnx_infer.table_line import table_line

from onnx_infer.utils import minAreaRectbox, measure, eval_angle, draw_lines
from onnx_infer.image import rotate_cut_img
import concurrent.futures

def re_scale(x_1, y_1, x_2, y_2, scale_ratio=0.05):
    table_box = [0, 0, 0, 0, 0, 0, 0, 0]
    wid = x_2 - x_1
    hei = y_2 - y_1
    table_box[0] = x_1 - scale_ratio * wid
    table_box[1] = y_1 - scale_ratio * hei
    table_box[2] = x_2 + scale_ratio * wid
    table_box[3] = y_1 - scale_ratio * hei
    table_box[4] = x_2 + scale_ratio * wid
    table_box[5] = y_2 + scale_ratio * hei
    table_box[6] = x_1 - scale_ratio * wid
    table_box[7] = y_2 + scale_ratio * hei
    return table_box, x_1, y_1

def re_scale_v2(x_1, y_1, x_2, y_2, scale_ratio=0.05):
    table_box = [0, 0, 0, 0, 0, 0, 0, 0]
    wid = x_2 - x_1
    hei = y_2 - y_1
    table_box[0] = x_1 - scale_ratio * wid
    table_box[1] = y_1
    table_box[2] = x_2 + scale_ratio * wid
    table_box[3] = y_1
    table_box[4] = x_2 + scale_ratio * wid
    table_box[5] = y_2
    table_box[6] = x_1 - scale_ratio * wid
    table_box[7] = y_2
    return table_box, x_1, y_1

def cut_img(img, table_box):
    im = Image.fromarray(img)
    tmpImg, boxes = rotate_cut_img(im, table_box, leftAdjustAlph=0.0, rightAdjustAlph=0.0)
    tmpImg = cv2.cvtColor(np.array(tmpImg), cv2.COLOR_RGB2BGR)

    return tmpImg, boxes

def rank_boxes(box):
    box = sorted(box, key=lambda x: sum([x[1], x[3]]))
    return list(box)

def data_augment(image):
    factor = 0.5499
    table = np.array([(i / 255.0) * factor * 255 for i in np.arange(0, 256)]).clip(0, 255).astype(np.uint8)
    image = cv2.LUT(image, table)
    return image

def sub_img_table(img, boxes):
    scale_ratio = 0.02
    sub_table = []
    if len(boxes) == 0:
        return sub_table
    elif len(boxes) == 1:
        x_1, y_1, x_2, y_2 = boxes[0]
        table_box, xmin, ymin = re_scale(x_1, y_1, x_2, y_2, scale_ratio=scale_ratio)
        sub_img, sub_boxes = cut_img(img, table_box)
        sub_table.append([sub_img])
    else:
        adBoxes = rank_boxes(boxes)
        for box_id, _box in enumerate(adBoxes):
            if box_id == 0:
                x_1 = _box[0]
                y_1 = _box[1]
                x_2 = _box[2]
                y_2 = _box[3]
                table_box, xmin, ymin = re_scale(x_1, y_1, x_2, y_2, scale_ratio=scale_ratio)
                sub_img, sub_boxes = cut_img(img, table_box)
                sub_table.append([sub_img])
            elif box_id < len(adBoxes) and box_id != len(adBoxes) - 1:
                x_1 = _box[0]
                y_1 = _box[1]
                x_2 = _box[2]
                y_2 = _box[3]
                table_box, xmin, ymin = re_scale(x_1, y_1, x_2, y_2, scale_ratio=scale_ratio)
                sub_img, sub_boxes = cut_img(img, table_box)
                sub_table.append([sub_img])
            else:
                x_1 = _box[0]
                y_1 = _box[1]
                x_2 = _box[2]
                y_2 = _box[3]
                table_box, xmin, ymin = re_scale(x_1, y_1, x_2, y_2, scale_ratio=scale_ratio)
                sub_img, sub_boxes = cut_img(img, table_box)
                sub_table.append([sub_img])
    return sub_table

def thread_table_line(table_line_model, sub_table, table_line_size):
    # table_n = len(self.boxes)
    table_n = len(sub_table)
    print('total table: {}'.format(table_n))

    table_line_result = []
    if table_n == 0:
        return table_line_result
    with concurrent.futures.ThreadPoolExecutor(max_workers=table_n) as executor:
        to_do = []
        for i in range(table_n):
            childImg = sub_table[i][0]
            future = executor.submit(table_line, table_line_model, data_augment(childImg)[..., ::-1], table_line_size, 0.5, 0.5, 50, 30, 15)
            to_do.append(future)
        for future in concurrent.futures.as_completed(to_do):
            table_line_result.append(future.result())
    return table_line_result

def thread_table_ceil(table_line_model, sub_table, table_line_size):
    table_line_result_list = thread_table_line(table_line_model, sub_table, table_line_size)
    table_n = len(table_line_result_list)
    table_ceil_boxes = []
    if table_n == 0:
        return table_ceil_boxes
    sub_img = sub_table[0][0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=table_n) as executor:
        to_do = []
        for i in range(table_n):
            table_line_result = table_line_result_list[i]
            rowboxes, colboxes = table_line_result[0], table_line_result[1]
            future = executor.submit(table_ceil, sub_img, rowboxes, colboxes)
            to_do.append(future)
        for future in concurrent.futures.as_completed(to_do):
            if future.result() is not None:
                table_ceil_boxes.extend(future.result())
    return table_ceil_boxes

def table_ceil(img, rowboxes, colboxes):
    tmp = np.zeros(img.shape[:2], dtype='uint8')
    tmp = draw_lines(tmp, rowboxes + colboxes, color=255, lineW=2)
    labels = measure.label(tmp < 255, connectivity=2)
    regions = measure.regionprops(labels)
    ceilboxes = minAreaRectbox(regions, False, tmp.shape[1], tmp.shape[0], True, True)
    ceilboxes = np.array(ceilboxes) # [[x1, y1, x2, y2, x3, y3, x4, y4], [...], [...]]
    if len(ceilboxes.shape) != 2:
        return None
    return ceilboxes

def table_structure_recognition(table_line_model, sub_table, table_line_size = (640, 640)):
    # [array([x1, y1, x2, y2, x3, y3, x4, y4]), array([...]), array([...]), ...]
    table_ceil_boxes = thread_table_ceil(table_line_model, sub_table, table_line_size)
    return table_ceil_boxes

def table_build(table_ceil_boxes, workbook=None):
    res = None
    if len(table_ceil_boxes) != 0:
        tablebuild = tableBuild(table_ceil_boxes)
        cor = tablebuild.cor
        for line in cor:
            line['text'] = 'cell-test' ##ocr
        workbook = to_excel(cor, workbook=workbook)
        res = cor
    return res, workbook

def parse_table_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelPath', type=str, default='./onnx/table_line.onnx', help='model path(s)') #
    parser.add_argument('--isTableDetect', default=True, type=bool, help="是否先进行表格检测")
    parser.add_argument('--tableSize', default='640,640', type=str, help="表格检测输入size")
    parser.add_argument('--tableLineSize', default='1024,1024', type=str, help="表格直线输入size")
    parser.add_argument('--isToExcel', default=True, type=bool, help="是否输出到excel")
    parser.add_argument('--tableSavePath', default='../test_imgs', type=str, help="切分表保存路径")
    parser.add_argument('--saveToExcelPath', type=str, help='save excel path(s)')
    parser.add_argument('--jpgPath',
                        default='../test_imgs/1.jpg',
                        type=str, help="测试图像地址")
    opt = parser.parse_args()
    return opt

