from tensorflow.keras import Model, layers, Input, Sequential


def conv3d(f, k=3, s=1, p="same", d=1, transpose=False):
    if transpose is True:
        return layers.Conv3DTranspose(f, k, s, p, use_bias=False)
    else:
        return layers.Conv3D(f, k, s, p, dilation_rate=d, use_bias=False)


def upsample_type(ch, mode="upsample"):
    if mode == "upsample":
        upsample = Sequential([
            layers.UpSampling3D(2),
            BasicBlock(ch, bn=True, act=True)
        ])
    elif mode == "transpose":
        upsample = BasicBlock(ch, transpose=True, bn=True, act=True)
    else:
        upsample = None
    return upsample


class BasicBlock(layers.Layer):
    def __init__(self, ch, k=3, s=1, d=1, transpose=False, bn=None, act=None):
        super(BasicBlock, self).__init__()
        self.conv = conv3d(ch, k, s, d=d, transpose=transpose)
        self.bn = layers.BatchNormalization() if bn else None
        self.act = layers.PReLU() if act else None

    def call(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ContractionBlock(layers.Layer):
    def __init__(self, ch, n, downsample=True, connect="add"):
        super(ContractionBlock, self).__init__()
        assert connect in ["add", "concat"]

        self.layers = Sequential()
        for i in range(n):
            if (i+1) == n:
                self.layers.add(BasicBlock(ch, bn=True))
            else:
                self.layers.add(BasicBlock(ch, bn=True, act=True))

        self.connect = layers.Add() if connect == "add" else layers.Concatenate()
        self.connect_conv = BasicBlock(ch, bn=True) if connect == "add" else None
        self.act = layers.PReLU()
        self.downsample = BasicBlock(ch, s=2, bn=True, act=True) if downsample else None

    def call(self, x):
        residual = x
        if self.connect_conv is not None:
            residual = self.connect_conv(residual)
        x = self.layers(x)
        x = self.connect([residual, x])
        x = self.act(x)
        fmap = x
        if self.downsample is not None:
            return self.downsample(x), fmap
        else:
            return x


class ExpansionBlock(layers.Layer):
    def __init__(self, ch, n, upsample="upsample", connect="add"):
        super(ExpansionBlock, self).__init__()
        assert connect in ["add", "concat"]

        self.layers = Sequential()
        for i in range(n):
            if (i+1) == n:
                self.layers.add(BasicBlock(ch, bn=True))
            else:
                self.layers.add(BasicBlock(ch, bn=True, act=True))

        self.connect = layers.Add() if connect == "add" else layers.Concatenate()
        self.connect_conv = BasicBlock(ch, bn=True) if connect == "add" else None
        self.act = layers.PReLU()
        self.upsample = upsample_type(ch, upsample)

    def call(self, x, fmap):
        residual = x
        if self.connect_conv is not None:
            residual = self.connect_conv(residual)
        x = layers.Concatenate()([x, fmap])
        x = self.layers(x)
        x = self.connect([residual, x])
        x = self.act(x)
        if self.upsample is not None:
            return self.upsample(x)
        else:
            return x


class VNet(Model):
    def __init__(self, ch, n_classes, upsample="upsample", name="VNet", **kwargs):
        super(VNet, self).__init__(name=name, **kwargs)
        mc, mn = [1,2,4,8,16], [1,2,3,3,3]
        self.contracts = [
            ContractionBlock(ch*c, n, False if (i+1)==len(mc) else True)
            for i, (c, n) in enumerate(zip(mc, mn))]
        self.upsample = upsample_type(ch*mc[-1], upsample)

        mc.reverse()
        mn.reverse()
        self.expands = [
            ExpansionBlock(ch*c, n, False if (i+1)==len(mc) else upsample)
            for i, (c, n) in enumerate(zip(mc[:-1], mn[1:]))]
        self.outputs = Sequential([
            BasicBlock(n_classes, k=1),
            layers.Softmax()])

    def call(self, x):
        x, c1 = self.contracts[0](x)
        x, c2 = self.contracts[1](x)
        x, c3 = self.contracts[2](x)
        x, c4 = self.contracts[3](x)
        x = self.contracts[4](x)
        x = self.upsample(x)

        x = self.expands[0](x, c4)
        x = self.expands[1](x, c3)
        x = self.expands[2](x, c2)
        x = self.expands[3](x, c1)
        return self.outputs(x)


inputs = Input((64,64,64,1))
outputs = VNet(16, 10)(inputs)
models = Model(inputs=inputs, outputs=outputs)
print(models.summary())

