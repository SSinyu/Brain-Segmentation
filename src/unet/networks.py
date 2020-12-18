import tensorflow as tf
from tensorflow.keras import layers, Input, Model, Sequential
from src.unet.modules import (
    get_filters,
    ConvBlock,
    UpConvConcatBlock,
    AttConcatBlock,
    CustomAttConcatBlock
)


class UNet(Model):
    def __init__(self, n_filter=64, n_blocks=5, tconv=False):
        super(UNet, self).__init__()
        enc_filters, dec_filters, _ = get_filters(n_filter, n_blocks)

        self.enc_blocks = []
        for f in enc_filters:
            self.enc_blocks.append(ConvBlock(f))
        self.maxpool = layers.MaxPool2D()

        self.upsample_blocks = []
        for i in range(n_blocks-1):
            self.upsample_blocks.append(
                UpConvConcatBlock(enc_filters[-i], tconv))

        self.dec_blocks = []
        for f in dec_filters:
            self.dec_blocks.append(ConvBlock(f))

        self.conv = layers.Conv2D(1, 1, 1, "same")

    def call(self, x):
        residuals = []
        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            if i != (len(self.enc_blocks)-1):
                residuals.append(x)
                x = self.maxpool(x)

        residuals = residuals[::-1]
        for residual, up_block, dec_block in zip(residuals,
                                                self.upsample_blocks,
                                                self.dec_blocks):
            x = up_block(x, residual)
            x = dec_block(x)
        return self.conv(x)

    def get_summary(self, input_shape=(256,256,1)):
        inputs = Input(input_shape)
        return Model(inputs, self.call(inputs)).summary()


class AttentionUNet(Model):
    def __init__(self, n_filter=64, n_blocks=5, att_type=None, tconv=False):
        super(AttentionUNet, self).__init__()
        enc_filters, dec_filters, _ = get_filters(n_filter, n_blocks)

        self.enc_blocks = []
        for f in enc_filters:
            self.enc_blocks.append(ConvBlock(f))
        self.maxpool = layers.MaxPool2D()

        self.att_blocks = []
        for i in range(n_blocks-1):
            self.att_blocks.append(
                AttConcatBlock(enc_filters[-i], tconv) if att_type is None else \
                CustomAttConcatBlock(enc_filters[-i], dec_filters[i], att_type, tconv)
                )

        self.dec_blocks = []
        for f in dec_filters:
            self.dec_blocks.append(ConvBlock(f))

        self.conv = layers.Conv2D(1, 1, 1, "same")

    def call(self, x):
        residuals = []
        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            if i != (len(self.enc_blocks)-1):
                residuals.append(x)
                x = self.maxpool(x)

        residuals = residuals[::-1]
        for residual, att_block, dec_block in zip(residuals,
                                                self.att_blocks,
                                                self.dec_blocks):
            x = att_block(x, residual)
            x = dec_block(x)
        return self.conv(x)

    def get_summary(self, input_shape=(256,256,1)):
        inputs = Input(input_shape)
        return Model(inputs, self.call(inputs)).summary(line_length=110)
