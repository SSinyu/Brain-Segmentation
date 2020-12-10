import tensorflow as tf
from tensorflow.keras import losses


class BinaryCrossEntropyLoss(losses.Loss):
    def __init__(
        self, from_logits:bool=True, label_smooth:float=0, name:str="bce_loss"
    ):
        super().__init__(name=name)
        self.from_logits = from_logits
        self.label_smooth = label_smooth

    def call(self, y_true, y_pred):
        return losses.binary_crossentropy(
            y_true, y_pred, self.from_logits, self.label_smooth
        )


class WeightedBinaryCrossEntropyLoss(losses.Loss):
    def __init__(
        self, from_logits:bool=True, w0=.2, w1=.8, label_smooth:float=0, name:str="wbce_loss"
    ):
        super().__init__(name=name)
        self.bce = BinaryCrossEntropyLoss(from_logits, label_smooth)
        self.w0, self.w1 = w0, w1

    def call(self, y_true, y_pred):
        w = y_true * self.w1 + (1.- y_true) * self.w0
        loss = w * self.bce(y_true, y_pred)
        return tf.reduce_men(loss)


class DiceLoss(losses.Loss):
    def __init__(
        self, from_logits:bool=True, smooth:float=1., name:str="dice_loss"
    ):
        super().__init__(name=name)
        self.from_logits = from_logits
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.reshape(
            tf.sigmoid(y_pred) if self.from_logits else y_pred,
            (-1,)
        )
        intersection = tf.reduce_sum(y_true * y_pred)
        dice = (2. * intersection + self.smooth) / \
                (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + self.smooth)
        return 1 - dice


class JaccardLoss(losses.Loss):
    def __init__(
        self, from_logits:bool=True, smooth:float=1., name:str="jaccard_loss"
    ):
        super().__init__(name=name)
        self.from_logits = from_logits
        self.smooth = smooth

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred))
        _dn = tf.reduce_sum(tf.abs(y_true) + tf.abs(y_pred))
        jaccard = (intersection + self.smooth) / \
                    (_dn - intersection + self.smooth)
        return 1 - jaccard


class FocalLoss(losses.Loss):
    """
    # https://github.com/umbertogriffo/focal-loss-keras

    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0
    """
    def __init__(
        self, from_logits:bool=True, gamma:float=2., alpha:float=.25, name:str="focal_loss"
    ):
        super().__init__(name=name)
        self.from_logits = from_logits
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(
            tf.sigmoid(y_pred) if self.from_logits else y_pred,
            1e-7, 1-1e-7  # epsilon = 1e-7
        )
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1-y_pred)
        alphat = tf.where(
            tf.equal(y_true, 1),
            tf.ones_like(y_true)*self.alpha,
            1 - (tf.ones_like(y_true)*self.alpha)
        )
        loss = alphat * tf.pow((1-pt), self.gamma) * -tf.math.log(pt)
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))


class TverskyLoss(losses.Loss):
    """
    # https://github.com/nabsabraham/focal-tversky-unet
    """
    def __init__(
        self, from_logits:bool=True, alpha:float=.7, smooth:float=1, name:str="tversky_loss"
    ):
        super().__init__(name=name)
        self.from_logits = from_logits
        self.alpha = alpha
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.reshape(
            tf.sigmoid(y_pred) if self.from_logits else y_pred,
            (-1,)
        )
        tp = tf.reduce_sum(y_true * y_pred)
        fn = tf.reduce_sum(y_true * (1-y_pred))
        fp = tf.reduce_sum((1-y_true) * y_pred)
        _dn = (tp + self.alpha * fn + (1-self.alpha) * fp + self.smooth)
        return (tp + self.smooth) / _dn


class BinaryCrossEntropyDiceLoss(losses.Loss):
    def __init__(
        self, from_logits:bool=True, bce_w:float=1., dice_w:float=1.,
        smooth:float=1., name:str="bce_dice_loss"
    ):
        super().__init__(name=name)
        self.from_logits = from_logits
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.bce_loss = BinaryCrossEntropyLoss(from_logits)
        self.dice_loss = DiceLoss(from_logits, smooth)

    def call(self, y_true, y_pred):
        bce = self.bce_loss(y_true, y_pred)
        dice = self.dice_loss(y_true, y_pred)
        return self.bce_w * bce + self.dice_w * dice


class FocalTverskyLoss(losses.Loss):
    """
    # https://github.com/nabsabraham/focal-tversky-unet
    """
    def __init__(
        self, from_logits:bool=True, gamma:float=.75, alpha:float=.7,
        smooth:float=1, name:str="focal_tversky_loss"
    ):
        super().__init__(name=name)
        self.gamma = gamma
        self.tversky = TverskyLoss(from_logits, alpha, smooth)

    def call(self, y_true, y_pred):
        tv = self.tversky(y_true, y_pred)
        return tf.pow((1-tv), self.gamma)
