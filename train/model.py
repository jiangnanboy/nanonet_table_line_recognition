from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D , Input
from tensorflow.keras.layers import  Cropping2D
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Large, MobileNetV3Small
from nanonet.se import squeeze_excite_block

def residual_block(x, num_filters):
    x_init = x
    x = Conv2D(num_filters//4, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters//4, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    s = Conv2D(num_filters, (1, 1), padding="same")(x_init)
    s = BatchNormalization()(x)

    x = Add()([x, s])
    x = Activation("relu")(x)
    x = squeeze_excite_block(x)
    return x

def ModelNet(input_shape, num_classes):

    f = [32, 64, 128]
    inputs = Input(shape=input_shape, name="input_image")

    ## Encoder
    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.50)
    # 获取模型每一层的参数名
    # for i, layer in enumerate(encoder.layers):
    #     print(f"Layer {i}: {layer.name}")
    #     for j, variable in enumerate(layer.variables):
    #         print(f"\tVariable {j}: {variable.name}")

    encoder_output = encoder.get_layer(name="block_6_expand_relu").output # block_6_expand_relu

    print('encoder_output shape:{}'.format(encoder_output.shape))
    skip_connections_name = ["input_image", "block_1_expand_relu", "block_3_expand_relu"]
    x = residual_block(encoder_output, 192)
    print('x shape:{}'.format(x.shape))

    ## Decoder
    for i in range(1, len(skip_connections_name)+1, 1):
        print(' index ------- {}'.format(-i))
        x_skip = encoder.get_layer(skip_connections_name[-i]).output
        print('x_skip shape:{}'.format(x_skip.shape))
        x_skip = Conv2D(f[-i], (1, 1), padding="same")(x_skip)
        x_skip = BatchNormalization()(x_skip)
        x_skip = Activation("relu")(x_skip)

        x = UpSampling2D((2, 2), interpolation='bilinear')(x)

        try:
            print('x shape:{}'.format(x.shape))
            print('x_skip shape:{}'.format(x_skip.shape))
            x = Concatenate()([x, x_skip])
        except Exception as e:
            x = Cropping2D(cropping=((1, 0), (0, 0)))(x)
            x = Concatenate()([x, x_skip])

        x = residual_block(x, f[-i])

    # Output section
    OUT = Conv2D(num_classes, kernel_size=1, use_bias=False,padding='same', activation='sigmoid')(x)
    # Model configuration
    model = Model(inputs=[inputs, ], outputs=[OUT, ])

    return model

if __name__ == '__main__':

    model = ModelNet((640, 640, 3), 2)
