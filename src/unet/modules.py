import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Sequential, Model


d_params = {
    "kernel_size": 3,
    "strides": 1,
    "padding": "same",
    "use_bias": False
}

u_params = d_params.copy()
u_params["strides"] = 2

a_params = d_params.copy()
a_params["kernel_size"] = 1


def get_filters(n_filter=64, n_blocks=5):
    mp = [1]
    for i in range(2, n_blocks+1):
        mp.append(mp[-1]*2)

    enc_filters = [n_filter*i for i in mp]
    if n_blocks > 5:
        for i in range(n_blocks -5):
            enc_filters.append(enc_filters[-1]*1)
    dec_filters = enc_filters[:-1][::-1]
    return enc_filters, dec_filters, mp


class ConvBlock(layers.Layer):
    def __init__(self, n_filter):
        super(ConvBlock, self).__init__()
        self.blocks = Sequential()
        for _ in range(2):
            self.blocks.add(layers.Conv2D(n_filter, **d_params))
            self.blocks.add(layers.BatchNormalization())
            self.blocks.add(layers.ReLU())

    def call(self, x):
        return self.blocks(x)


class UpConvConcatBlock(layers.Layer):
    def __init__(self, n_filter, tconv=False):
        super(UpConvConcatBlock, self).__init__()
        if tconv is True:
            self.blocks = [layers.Conv2DTranspose(n_filter, **u_params)]
        else:
            self.blocks = [
               layers.UpSampling2D(interpolation='bilinear'),
               layers.Conv2D(n_filter, **d_params)
            ]
        self.concat = layers.Concatenate()

    def call(self, x, residual):
        for block in self.blocks:
            x = block(x)
        return self.concat([x, residual])


class AttentionBlock(layers.Layer):
    def __init__(self, n_filter):
        super(AttentionBlock, self).__init__()
        self.g_blocks = Sequential([
            layers.Conv2D(n_filter, **a_params),
            layers.BatchNormalization()
        ])
        self.x_blocks = Sequential([
            layers.Conv2D(n_filter, **a_params),
            layers.BatchNormalization()
        ])
        self.out_blocks = Sequential([
            layers.Conv2D(1, **a_params),
            layers.BatchNormalization(),
            layers.Activation("sigmoid")
        ])
        self.add = layers.Add()
        self.relu = layers.ReLU()
        self.multiply = layers.Multiply()

    def call(self, g, x):
        _g = self.g_blocks(g)
        _x = self.x_blocks(x)
        out = self.relu(self.add([_g, _x]))
        out = self.out_blocks(out)
        return self.multiply([x, out])


class NonLocalAttModule(layers.Layer):
    def __init__(self, n_filter):
        super(NonLocalAttModule, self).__init__()
        self.theta_blocks = Sequential([
            layers.Conv2D(n_filter, **a_params),
            layers.Reshape((-1, n_filter))
        ])
        self.phi_blocks = Sequential([
            layers.Conv2D(n_filter, **a_params),
            layers.Reshape((n_filter, -1))
        ])
        self.matmul = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))
        self.softmax = layers.Softmax()
        self.g_blocks = Sequential([
            layers.Conv2D(n_filter, **a_params),
            layers.Reshape((-1, n_filter))
        ])
        self.conv = layers.Conv2D(n_filter*2, **a_params)
        self.add = layers.Add()

    def call(self, x):
        _, d1, d2, d3 = x.shape
        theta_out = self.theta_blocks(x)
        phi_out = self.phi_blocks(x)
        out = self.softmax(self.matmul([theta_out, phi_out]))

        g_out = self.g_blocks(x)
        out = self.matmul([out, g_out])
        out = layers.Reshape((d1, d2, d3//2))(out)
        out = self.conv(out)
        return self.add([x, out])


class ConvBlockAttModule(layers.Layer):
    def __init__(self, n_filter, r=8):
        super(ConvBlockAttModule, self).__init__()
        self.ch_att_blocks_1 = Sequential([
            layers.GlobalMaxPooling2D(),
            layers.Reshape((1, 1, n_filter)),
            layers.Dense(n_filter//r, activation="relu"),
            layers.Dense(n_filter)
        ])
        self.ch_att_blocks_2 = Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Reshape((1, 1, n_filter)),
            layers.Dense(n_filter//r, activation="relu"),
            layers.Dense(n_filter)
        ])
        self.add = layers.Add()
        self.sigmoid = layers.Activation("sigmoid")
        self.multiply = layers.Multiply()

        self.sp_att_block_1 = layers.Lambda(
            lambda x: K.max(x, axis=3, keepdims=True))
        self.sp_att_block_2 = layers.Lambda(
            lambda x: K.max(x, axis=3, keepdims=True))

        self.concat = layers.Concatenate()
        self.conv = layers.Conv2D(
            1, 7, 1, "same", activation="sigmoid", use_bias=False)

    def call(self, x):
        ch_out_1 = self.ch_att_blocks_1(x)
        ch_out_2 = self.ch_att_blocks_2(x)
        ch_out = self.sigmoid(self.add([ch_out_1, ch_out_2]))
        ch_out = self.multiply([x, ch_out])

        sp_out_1 = self.sp_att_block_1(ch_out)
        sp_out_2 = self.sp_att_block_2(ch_out)
        sp_out = self.concat([sp_out_1, sp_out_2])
        sp_out = self.conv(sp_out)
        return self.multiply([ch_out, sp_out])


class AttConcatBlock(layers.Layer):
    def __init__(self, n_filter, tconv=False):
        super(AttConcatBlock, self).__init__()
        self.up_blocks = Sequential()
        if tconv is True:
            self.up_blocks.add(layers.Conv2DTranspose(n_filter, **u_params))
        else:
            self.up_blocks.add(layers.UpSampling2D(interpolation="bilinear"))
            self.up_blocks.add(layers.Conv2D(n_filter, **d_params))

        self.att_block = AttentionBlock(n_filter)
        self.concat = layers.Concatenate()

    def call(self, x, residual):
        x = self.up_blocks(x)
        att_out = self.att_block(x, residual)
        return self.concat([x, att_out])


class CustomAttConcatBlock(layers.Layer):
    def __init__(self, n_filter, att_filter, att_type="cbam", tconv=False):
        super(CustomAttConcatBlock, self).__init__()
        self.up_blocks = Sequential()
        if tconv is True:
            self.up_blocks.add(layers.Conv2DTranspose(n_filter, **u_params))
        else:
            self.up_blocks.add(layers.UpSampling2D(interpolation="bilinear"))
            self.up_blocks.add(layers.Conv2D(n_filter, **d_params))

        if att_type == "cbam":
            self.att_block = ConvBlockAttModule(att_filter)
        elif att_type == "nln":
            self.att_block = NonLocalAttModule(att_filter//2)
        self.concat = layers.Concatenate()

    def call(self, x, residual):
        x = self.up_blocks(x)
        att_out = self.att_block(residual)
        return self.concat([x, att_out])
