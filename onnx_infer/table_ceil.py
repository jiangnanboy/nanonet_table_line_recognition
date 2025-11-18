#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.getcwd()))

import cv2
import numpy as np
from PIL import Image
import argparse
from onnx_infer.table_line import table_line

from onnx_infer.table_build import tableBuild,to_excel
from onnx_infer.utils import minAreaRectbox, measure, eval_angle, draw_lines
from onnx_infer.image import rotate_cut_img
import concurrent.futures

class table:
    def __init__(self, img, layout_opt, layout_model, table_line_model, tableSavePath, tableSize=(416, 416), tableLineSize=(1024, 1024), isToExcel=False):
        self.img = img
        self.layout_opt= layout_opt
        self.layout_model = layout_model
        self.table_line_model = table_line_model
        self.tableSavePath = tableSavePath
        self.tableSize = tableSize
        self.tableLineSize = tableLineSize
        self.isToExcel = isToExcel
        self.img_degree()
        self.table_boxes_detect()
        self.sub_img_table()
        self.thread_table_ceil()

    def img_degree(self):
        img, degree = eval_angle(self.img, angleRange=[-15, 15])
        self.img = img
        self.degree = degree

    def rank_boxes(self, box):
        box = sorted(box, key=lambda x: sum([x[1], x[3]]))
        return list(box)

    def re_scale(self, x_1, y_1, x_2, y_2, scale_ratio=0.02):
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

    def cut_img(self, table_box):
        im = Image.fromarray(self.img)
        tmpImg, boxes = rotate_cut_img(im, table_box, leftAdjustAlph=0.0, rightAdjustAlph=0.0)
        tmpImg = cv2.cvtColor(np.array(tmpImg), cv2.COLOR_RGB2BGR)
        # height, width = tmpImg.shape[:2]
        # img = cv2.resize(tmpImg, (int(0.25 * width), int(0.25 * height)))
        return tmpImg, boxes

    def sub_img_table(self):
        self.sub_table = []
        if len(self.boxes) == 0:
            self.sub_table = []
        elif len(self.boxes) == 1:
            x_1, y_1, x_2, y_2 = self.boxes[0]
            # x_1, y_1, x_2, y_2 = self.adBoxes[0]
            table_box, xmin, ymin = self.re_scale(x_1, y_1, x_2, y_2)
            img, boxes = self.cut_img(table_box)
            cv2.imwrite(self.tableSavePath + "0_tmpImg.jpg", img)
            self.sub_table.append((img, boxes, xmin, ymin))
        else:
            X_1, Y_1, X_2, Y_2 = 0, 0, 0, 0
            x_1, y_1, x_2, y_2 = 0, 0, 0, 0
            table_box = [0, 0, 0, 0, 0, 0, 0, 0]
            adBoxes = self.rank_boxes(self.boxes)
            for box_id, _box in enumerate(adBoxes):
                if box_id == 0:
                    x_1 = _box[0]
                    y_1 = _box[1]
                    x_2 = _box[2]
                    y_2 = _box[3]
                    table_box, xmin, ymin = self.re_scale(x_1, y_1, x_2, y_2)
                    img, boxes = self.cut_img(table_box)
                    cv2.imwrite(self.tableSavePath + str(box_id) + "_tmpImg.jpg", img)
                    self.sub_table.append((img, boxes, xmin, ymin))
                elif box_id < len(adBoxes) and box_id != len(adBoxes) - 1:
                    x_1 = _box[0]
                    y_1 = _box[1]
                    x_2 = _box[2]
                    y_2 = _box[3]
                    table_box, xmin, ymin = self.re_scale(x_1, y_1, x_2, y_2)
                    img, boxes = self.cut_img(table_box)
                    cv2.imwrite(self.tableSavePath + str(box_id) + "_tmpImg.jpg", img)
                    self.sub_table.append((img, boxes, xmin, ymin))
                else:
                    x_1 = _box[0]
                    y_1 = _box[1]
                    x_2 = _box[2]
                    y_2 = _box[3]
                    table_box, xmin, ymin = self.re_scale(x_1, y_1, x_2, y_2)
                    img, boxes = self.cut_img(table_box)
                    cv2.imwrite(self.tableSavePath + str(box_id) + "_tmpImg.jpg", img)
                    self.sub_table.append((img, boxes, xmin, ymin))

    def table_boxes_detect(self):
        h, w = self.img.shape[:2]
        boxes = [[0, 0, w, h]]
        adBoxes = [[0, 0, w, h]]
        scores = [0]

        print('boxes: {}'.format(boxes))
        print('adboxes: {}'.format(adBoxes))
        print('scores: {}'.format(scores))

        self.boxes = boxes
        self.adBoxes = adBoxes
        self.scores = scores

    def no_thread_table_line(self):
        table_n = len(self.sub_table)
        print('total table: {}'.format(table_n))
        table_line_result = []
        t1 = time.time()
        for i in range(table_n):
            childImg = self.sub_table[i]
            rowboxes, colboxes, child_img_xmin, child_img_ymin = table_line(self.table_line_model, childImg[..., ::-1], 0, 0,  self.tableLineSize, 0.5, 0.5, 50, 30, 15)
            table_line_result.append((rowboxes, colboxes, child_img_xmin, child_img_ymin))
        t2 = time.time()
        print('no thread table line : {}'.format((t2 - t1)))
        return table_line_result

    def no_thread_table_ceil(self):
        table_line_result_list = self.no_thread_table_line()
        table_n = len(table_line_result_list)
        self.tableCeilBoxes = []
        t1 = time.time()
        for i in range(table_n):
            table_line_result = table_line_result_list[i]
            rowboxes, colboxes, xmin, ymin = table_line_result[0], table_line_result[1], table_line_result[2], \
                                             table_line_result[3]
            ceilboxes = self._table_ceil(rowboxes, colboxes, xmin, ymin)
            self.tableCeilBoxes.extend(ceilboxes)
        t2 = time.time()
        print('no_thread_table ceil : {}'.format((t2 - t1)))

    def thread_table_line(self):
        # table_n = len(self.boxes)
        table_n = len(self.sub_table)
        print('total table: {}'.format(table_n))

        table_line_result = []
        if table_n == 0:
            return table_line_result
        t1 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=table_n) as executor:
            to_do = []
            for i in range(table_n):
                xmin, ymin, xmax, ymax = [int(x) for x in self.adBoxes[i]]
                # childImg = self.img[ymin:ymax, xmin:xmax]
                childImg = self.sub_table[i][0]
                # childImg = data_augment(childImg)
                boxes = self.sub_table[i][1]
                xmin = self.sub_table[i][2]
                ymin = self.sub_table[i][3]
                # xmin = boxes['cx']
                # ymin = boxes['cy']
                # print('xmin: {}\n ymin:{}'.format(xmin, ymin))
                future = executor.submit(table_line, self.table_line_model, childImg[..., ::-1], xmin, ymin, self.tableLineSize, 0.5, 0.5, 50, 30, 15)
                to_do.append(future)
            for future in concurrent.futures.as_completed(to_do):
                table_line_result.append(future.result())
        t2 = time.time()
        print('table line : {}'.format((t2 - t1)))
        return table_line_result

    def thread_table_ceil(self):
        table_line_result_list = self.thread_table_line()
        table_n = len(table_line_result_list)

        self.tableCeilBoxes = []
        if table_n != 0:
            t1 = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=table_n) as executor:
                to_do = []
                for i in range(table_n):
                    table_line_result = table_line_result_list[i]
                    rowboxes, colboxes, xmin, ymin = table_line_result[0], table_line_result[1], table_line_result[2], \
                                                     table_line_result[3]
                    future = executor.submit(self._table_ceil, rowboxes, colboxes, xmin, ymin)
                    to_do.append(future)
                for future in concurrent.futures.as_completed(to_do):
                    if future.result() is not None:
                        self.tableCeilBoxes.extend(future.result())
            t2 = time.time()
            print('table ceil boxes : {}'.format(self.tableCeilBoxes))
            print('thread_table ceil : {}'.format((t2 - t1)))

    def _table_ceil(self, rowboxes, colboxes, xmin, ymin):
        # height, width = self.img.shape[:2]
        # img = cv2.resize(self.img, (int(0.25 * width), int(0.25*height)))
        # self.img = img
        # tmp = np.zeros(self.img.shape[:2], dtype='uint8')
        img = self.img
        tmp = np.zeros(img.shape[:2], dtype='uint8')
        tmp = draw_lines(tmp, rowboxes + colboxes, color=255, lineW=2)

        measure_time1 = time.time()
        labels = measure.label(tmp < 255, connectivity=2)
        measure_time2 = time.time()
        print('measure : {}'.format((measure_time2 - measure_time1)))

        regionprops_time1 = time.time()
        regions = measure.regionprops(labels)
        regionprops_time2 = time.time()
        print('regionprops : {}'.format((regionprops_time2 - regionprops_time1)))

        minAreaRectbox_time1 = time.time()
        ceilboxes = minAreaRectbox(regions, False, tmp.shape[1], tmp.shape[0], True, True)
        minAreaRectbox_time2 = time.time()
        print('minAreaRectbox : {}'.format((minAreaRectbox_time2 - minAreaRectbox_time1)))

        ceilboxes = np.array(ceilboxes)
        if len(ceilboxes.shape) != 2:
            return None
        ceilboxes[:, [0, 2, 4, 6]] += xmin
        ceilboxes[:, [1, 3, 5, 7]] += ymin

        return ceilboxes

    def table_ceil(self):
        n = len(self.boxes)
        print('total table: {}'.format(n))

        self.tableCeilBoxes = []
        self.childImgs = []
        t1 = time.time()
        for i in range(n):

            xmin, ymin, xmax, ymax = [int(x) for x in self.boxes[i]]
            childImg = self.img[ymin:ymax, xmin:xmax]
            rowboxes, colboxes, _, _ = table_line(self.table_line_model, childImg[..., ::-1], None, None, size=self.tableLineSize, hprob=0.5, vprob=0.5)

            print(rowboxes)
            print(colboxes)

            tmp = np.zeros(self.img.shape[:2], dtype='uint8')
            tmp = draw_lines(tmp, rowboxes + colboxes, color=255, lineW=2)

            measure_time1 = time.time()
            labels = measure.label(tmp < 255, connectivity=2)
            measure_time2 = time.time()
            print('measure : {}'.format((measure_time2 - measure_time1)))

            regionprops_time1 = time.time()
            regions = measure.regionprops(labels)
            regionprops_time2 = time.time()
            print('regionprops : {}'.format((regionprops_time2 - regionprops_time1)))

            minAreaRectbox_time1 = time.time()
            ceilboxes = minAreaRectbox(regions, False, tmp.shape[1], tmp.shape[0], True, True)
            minAreaRectbox_time2 = time.time()
            print('minAreaRectbox : {}'.format((minAreaRectbox_time2 - minAreaRectbox_time1)))

            ceilboxes = np.array(ceilboxes)
            if len(ceilboxes.shape) != 2:
                break
            ceilboxes[:, [0, 2, 4, 6]] += xmin
            ceilboxes[:, [1, 3, 5, 7]] += ymin

            self.tableCeilBoxes.extend(ceilboxes)
            self.childImgs.append(childImg)

        t2 = time.time()
        print('table ceil : {}'.format((t2 - t1)))

    def table_build(self):
        '''
        table rebuild
        :return:
        '''
        self.res = None
        self.workbook = None
        if len(self.tableCeilBoxes) != 0:
            tablebuild = tableBuild(self.tableCeilBoxes)
            cor = tablebuild.cor
            for line in cor:
                line['text'] = 'table-test' ##ocr
            if self.isToExcel:
                workbook = to_excel(cor, workbook=None)
            self.res = cor
            self.workbook = workbook
            self.workbook.save('../test_imgs/result.xls')

    def table_ocr(self):
        """use ocr and match ceil"""
        pass

def parse_table_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelPath', type=str, default='./onnx/table_line.onnx', help='model path(s)') #
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

