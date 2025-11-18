from onnx_infer.table_build import table_xlsx
from onnx_infer.table_line import load_table_wire_line_model, table_line
from onnx_infer.table_structure_reg import table_ceil
from onnx_infer.utils import draw_lines

table_wire_model_path = '../models/table_light_line.onnx'

# load model
table_wire_model = load_table_wire_line_model(table_wire_model_path)


import cv2

table_img = '../test_imgs/1.jpg'

table_img = cv2.imread(table_img)

# get rows and columns
rowboxes, colboxes = table_line(table_wire_model, table_img)

# draw lines
img = draw_lines(table_img, rowboxes + colboxes, color=(255, 0, 0), lineW=2)
cv2.imwrite('../test_imgs/1_line.jpg', img)

# get ceil boxes
ceilboxes = table_ceil(table_img, rowboxes, colboxes)

print(ceilboxes)

# convert to an excel table
workbook = table_xlsx(ceilboxes)
workbook.save('table_ceil.xlsx')