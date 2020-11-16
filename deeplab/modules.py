from tensorflow.keras import layers


def custom_pad(k_size, d_rate):
    s = k_size + (k_size-1) * (d_rate-1)
    pad_s = (s-1)//2
    pad_e = (s-1) - pad_s
    pad_layer = layers.ZeroPadding2D((pad_s, pad_e))
    return pad_layer


class ConvBlock(layers.Layer):
    def __init__(self, n_filters, k_size=3, stride=1, d_rate=1):
        super(ConvBlock, self).__init__()
        pad_type = "same" if stride == 1 else "valid"

        self.blocks = []
        if stride > 1:
            self.blocks.append(custom_pad(k_size, d_rate))
        self.blocks.append(
            layers.Conv2D(n_filters, k_size, stride, pad_type, use_bias=False, dilation_rate=d_rate)
        )

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class BasicBlock(layers.Layer):
    def __init__(self, n_filters, k_size=3, stride=1, d_rate=1):
        super(BasicBlock, self).__init__()
        self.blocks = [
            # ConvBlock(n_filters, k_size, stride, d_rate) if pad_block else
            layers.Conv2D(n_filters, k_size, stride, "same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ]

    def call(self, x, training=None):
        for block in self.blocks:
            x = block(x)
        return x


class SeparableConvBlock(layers.Layer):
    def __init__(self, n_filters, k_size=3, stride=1, d_rate=1, activation=False):
        super(SeparableConvBlock, self).__init__()
        pad_type = "same" if stride == 1 else "valid"

        self.blocks = [
            custom_pad(k_size, d_rate) if stride > 1 else None,
            layers.ReLU() if activation is False else None,
            layers.DepthwiseConv2D(k_size, stride, pad_type, use_bias=False, dilation_rate=d_rate),
            layers.BatchNormalization(),
            layers.ReLU() if activation is True else None,
            layers.Conv2D(n_filters, 1, 1, "same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU() if activation is True else None
        ]

    def call(self, x, training=None):
        for block in self.blocks:
            if block is not None:
                x = block(x)
        return x


class XceptionBlock(layers.Layer):
    def __init__(self, n_filters, skip_type=None, stride=1, d_rate=1, activation=False, return_skip=False):
        super(XceptionBlock, self).__init__()
        # self.sepconv_1 = SeparableConvBlock(n_filters[0], 3, 1, d_rate, activation)
        # self.sepconv_2 = SeparableConvBlock(n_filters[1], 3, 1, d_rate, activation)
        # self.sepconv_3 = SeparableConvBlock(n_filters[2], 3, stride, d_rate, activation)
        self.sepconv_blocks = [
            SeparableConvBlock(n_filters[i], 3, stride if i==2 else 1, d_rate, activation) for i in range(3)
        ]

        self.skip_type = skip_type
        self.return_skip = return_skip
        self.add = layers.Add()
        if skip_type == "conv":
            self.skipconv = ConvBlock(n_filters[-1], 1, stride)
            self.skipbn = layers.BatchNormalization()

    def call(self, x, training=None):
        residual = x

        for i, block in enumerate(self.sepconv_blocks):
            residual = block(residual)
            if i == 1:
                skip = residual

        if self.skip_type == "conv":
            out = self.skipconv(x)
            out = self.skipbn(out)
            out = self.add([residual, out])
        elif self.skip_type == "sum":
            out = self.add([residual, x])
        elif self.skip_type == None:
            out = residual

        if self.return_skip:
            return out, skip
        else:
            return out


class XceptionBackbone(layers.Layer):
    def __init__(self, output_stride=8):
        super(XceptionBackbone, self).__init__()
        x_stride = 1 if output_stride==8 else 2
        d_rates = (2,2,4) if output_stride==8 else (1,1,2)

        self.entry_flow = [
            BasicBlock(32, 3, 2),
            BasicBlock(64, 3, 1),
            XceptionBlock([128]*3, "conv", 2),
            XceptionBlock([256]*3, "conv", 2, return_skip=True),
            XceptionBlock([728]*3, "conv", x_stride)
        ]
        self.middle_flow = [
            XceptionBlock([728]*3, "sum", 1, d_rates[0]) for _ in range(16)
        ]
        self.exit_flow = [
            XceptionBlock([728,1024,1024], "conv", 1, d_rates[1]),
            XceptionBlock([1536,1536,2048], None, 1, d_rates[2], True)
        ]

    def call(self, x, training=None):
        for i, block in enumerate(self.entry_flow):
            if i == 3:
                x, skip = block(x)
            else:
                x = block(x)

        for block in self.middle_flow:
            x = block(x)

        for block in self.exit_flow:
            x = block(x)
        return x, skip
