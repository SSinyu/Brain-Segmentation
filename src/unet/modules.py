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


class UpConvAttConcatBlock(layers.Layer):
    def __init__(self, n_filter, tconv=False):
        super(UpConvAttConcatBlock, self).__init__()
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
