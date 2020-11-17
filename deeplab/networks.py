import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from deeplab.modules import BasicBlock, SeparableConvBlock, XceptionBackbone


class ASPP(layers.Layer):
    def __init__(self, output_stride=8):
        super(ASPP, self).__init__()
        d_rates = (12,24,36) if output_stride==8 else (6,12,18)

        self.block_1x1 = BasicBlock(256, 1, 1)
        self.block_3x3_rate_S = BasicBlock(256, d_rate=d_rates[0], activation=True)
        self.block_3x3_rate_M = BasicBlock(256, d_rate=d_rates[1], activation=True)
        self.block_3x3_rate_L = BasicBlock(256, d_rate=d_rates[2], activation=True)
        self.blocks_pooling = [
            layers.GlobalAveragePooling2D(),
            layers.Lambda(lambda x: x[:, tf.newaxis, tf.newaxis, :]),
            BasicBlock(256, 1, 1)
        ]

        self.concat = layers.Concatenate()
        self.concat_1x1 = BasicBlock(256, 1, 1)

    def call(self, x, training=None):
        x1 = self.block_1x1(x)
        x2 = self.block_3x3_rate_S(x)
        x3 = self.block_3x3_rate_M(x)
        x4 = self.block_3x3_rate_L(x)
        x5 = x

        for block in self.blocks_pooling:
            x5 = block(x5)

        f_shape = x._shape_tuple()[1:3]
        x5 = tf.image.resize(x5, f_shape, "bilinear")
        x = self.concat([x1, x2, x3, x4, x5])
        return self.concat_1x1(x)


class DeepLabV3Decoder(layers.Layer):
    def __init__(self, n_classes=1):
        super(DeepLabV3Decoder, self).__init__()
        self.skip_block = BasicBlock(48, 1, 1)
        self.concat = layers.Concatenate()

        self.sepconv1 = SeparableConvBlock(256, activation=True)
        self.sepconv2 = SeparableConvBlock(256, activation=True)
        self.conv = layers.Conv2D(n_classes, 1, 1, "same")

        self.sigmoid = layers.Activation("sigmoid")

    def call(self, x, skip, input_shape, training=None):
        skip_shape = skip._shape_tuple()[1:3]
        x = tf.image.resize(x, skip_shape, "bilinear")

        skip = self.skip_block(skip)

        x = self.concat([x, skip])
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = self.conv(x)

        x = tf.image.resize(x, input_shape, "bilinear")
        return self.sigmoid(x)


class DeepLabV3pXc(Model):
    def __init__(self, output_stride=8, n_classes=1):
        super(DeepLabV3pXc, self).__init__()
        self.backbone = XceptionBackbone(output_stride)
        self.aspp = ASPP(output_stride)
        self.decoder = DeepLabV3Decoder(n_classes)

    def call(self, x, training=None):
        input_shape = x._shape_tuple()[1:3]

        x, skip = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, skip, input_shape)
        return x

    def get_summary(self, input_shape=(256,256,3)):
        inputs = Input(input_shape)
        model = Model(inputs, self.call(inputs, False))
        print(model.summary())
