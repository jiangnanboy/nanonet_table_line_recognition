import tensorflow as tf

# from config import vgg19_h5_path
# from train.model import build_model
from model import ModelNet

def keras2tf():
    '''
    加载h5或hdf5格式模型文件，保存为tf模型，pb格式的模型文件
    :return:
    '''
    # model_path = r'E:\git_project\table-detect-master\models\nanonet_model\model.h5'
    # model = tf.keras.models.load_model(model_path)

    # 1.h5转为tf的pb格式
    model = ModelNet((640, 640, 3), 2)
    model.load_weights('../models/model.h5')
    model.save('\pb', save_format='tf')

    # 将pb格式转为onnx格式

    # 2.转为Onnx
    # 利用tf2onnx下的convert将pb格式转为onnx
    # 命令：>python -m tf2onnx.convert --saved-model ./tfmodel/ --output ./onnx/table_line.onnx --opset 11 --verbose

    # 3.将fp32转为fp16精度
    # 利用onnxmltools进行精度转换
    # import onnxmltool
    #
    # from onnxmltools.utils.float16_converter import convert_float_to_float16
    # onnx_model = onnxmltools.load_model('./onnx/table_line.onnx')
    # onnx_model_fp16 = convert_float_to_float16(onnx_model)
    # onnxmltools.utils.save_model(onnx_model_fp16, './onnx/table_line_fp16.onnx')


if __name__ == '__main__':
    keras2tf()










