import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class CosineAnnealingScheduler(LearningRateSchedule):
    def __init__(self, init_lr, n_epochs, n_reset=1):
        super(CosineAnnealingScheduler, self).__init__()
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.n_reset = n_reset
        self.pi = 3.14159265

    def __call__(self, step):
        _cos = self.pi * (step % (self.n_epochs // self.n_reset))
        _cos /= self.n_epochs // self.n_reset
        _cos = tf.cos(_cos) + 1
        return tf.cast(self.init_lr / 2*_cos, tf.float32)


class TransformerScheduler(LearningRateSchedule):
    def __init__(self, dim=512, warmup_steps=1000):
        super(TransformerScheduler, self).__init__()
        self.dim = tf.cast(dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        c0 = tf.math.rsqrt(step)
        c1 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dim) * tf.mat.minimum(c0, c1)
