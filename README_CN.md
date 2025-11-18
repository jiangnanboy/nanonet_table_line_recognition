[English](README.md) | 简体中文
# NanoNet-Table:一种超轻量级的有线表格识别方法 
在无意间看到一篇CBMS2021的一篇论文《NanoNet: Real-Time Polyp Segmentation in Video Capsule Endoscopy and Colonoscopy》，

该论文是专为视频胶囊内窥镜和结肠镜图像的息肉分割设计的架构，是一种医学图像分割的方法。

故打算基于以上模型的架构去实现一超轻量级的有线表格识别方法，模型训练后的大小为 __1MB__ 左右（是不是贼轻量），用于手动移动部署毫无压力。

训练数据自己标注的，共1000张左右，要想效果好，可增加数据量，模型利用tensorflow训练。

为了用户轻松自由毫无压力使用，我这里已经将tensorflow训练的模型转为onnx格式了，识别推理只需要用onnxruntime即可。

![](test_imgs/architecture.png) 
## 实现功能   
- [x] 识别表格中的线条
- [x] 结果转为excel

## weights模型文件   
模型文件: 

1. models/model.h5 

2. models/table_light_line.onnx

## 训练（tensorflow2.5版本训练）     
见本项目中的train/train.py    

## onnx识别
见本项目中的onnx_infer/onnx_inference.py

```python
from onnx_infer.table_build import table_xlsx
from onnx_infer.table_line import load_table_wire_line_model, table_line
from onnx_infer.table_structure_reg import table_ceil
from onnx_infer.utils import draw_lines

table_wire_model_path = '../models/table_light_line.onnx'

# load model
table_wire_model = load_table_wire_line_model(table_wire_model_path)

import cv2

table_img = '../test_imgs/6.jpg'

table_img = cv2.imread(table_img)

# get rows and columns
rowboxes, colboxes = table_line(table_wire_model, table_img)

# draw lines
img = draw_lines(table_img, rowboxes + colboxes, color=(255, 0, 0), lineW=2)
cv2.imwrite('../test_imgs/6_line.jpg', img)

# get ceil boxes
ceilboxes = table_ceil(table_img, rowboxes, colboxes)

print(ceilboxes)

# convert to an excel table
workbook = table_xlsx(ceilboxes)
workbook.save('table_ceil.xlsx')
```

## 部分识别结果展示
转为excel结构中，表格里的每个单元格的文字“cell-test”是人为加入的，后面可以结合ocr将文字填入单元格中。

![](test_imgs/1.jpg) 

![](test_imgs/1_line.jpg) 

![](test_imgs/1_excel.png) 

---

![](test_imgs/3.png) 

![](test_imgs/3_line.jpg) 

![](test_imgs/3_excel.png) 

## contact
1. github：https://jiangnanboy.github.io
2. blog：https://blog.csdn.net/qq_20182781
3. email:2229029156@qq.com
   
