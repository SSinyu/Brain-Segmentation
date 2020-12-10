import tensorflow as tf
from tensorflow.keras import optimizers, metrics

from src import scheduler, losses
from src.models import get_model
from src.loader import DataLoader


class Solver:
    def __init__(self, config, datasets):
        self.config = config
        self.datasets = datasets

        self.mode = config["mode"]
        if config["mode"] == "train":
            self.val_datasets = DataLoader(
                "valid", **config["dataset"]
            )

        training_config = config["training"]
        self.epochs = training_config["epochs"]
        self.print_iter = training_config["print_iter"]
        self.save_epoch = training_config["save_epoch"]

        model_func = get_model(config["model"]["type"])
        self.model = model_func(
            **config[config["model"]["type"]]
        )

        lr = scheduler.CosineAnnealingScheduler(
            training_config["init_learning_rate"],
            training_config["epochs"]
        )
        self.optimizer = optimizers.Adam(lr, **config["optimizer"])

        self.bce_w = training_config["bce_loss_weight"]
        self.dice_w = training_config["dice_loss_weight"]
        self.bce = losses.BinaryCrossEntropyLoss()
        self.dice = losses.DiceLoss()

        self.get_ckpt_manager(training_config["save_path"])

    def get_ckpt_manager(self, ckpt_path, keep=5):
        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_path, max_to_keep=keep)

    def train(self):
        n_iters = len(self.datasets)
        val_loss = 1e6
        for epoch in range(self.epochs):
            for (i, (image, mask)) in enumerate(self.datasets):
                bce_loss, dice_loss = self.train_batch(image, mask)

                if (i+1) % self.print_iter == 0:
                    print(f"[{epoch+1}/{self.epochs}] Epoch, [{i+1}/{n_iters}] Iter")
                    print(f"BCE Loss : {bce_loss:.5f}, DICE Loss : {dice_loss:.5f}")

            v_bce, v_dice = self.valid_steps()
            v_total = v_bce + v_dice
            print(f"===== [{epoch+1}/{self.epochs}] Epoch")
            print(f"===== val BCE : {v_bce:.5f}, val DICE : {v_dice:.5f}, val Total : {v_total:.5f}")

            if v_total < val_loss:
                val_loss = v_total
                ckpt_save_path = self.ckpt_manager.save()

            if (epoch+1) % self.save_epoch == 0:
                f = self.config["save_path"] + f"/epoch_{epoch+1}.h5"
                self.model.save_weights(f)
                print("save ", f)

    def test(self):
        raise NotImplementedError

    @tf.function
    def train_batch(self, image, mask):
        with tf.GradientTape() as tape:
            pred = self.model(image)
            bce_loss = self.bce(mask, pred)
            dice_loss = self.dice(mask, pred)
            total_loss = self.bce_w * bce_loss + self.dice_w * dice_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        return bce_loss, dice_loss

    def valid_steps(self):
        bce_loss, dice_loss = 0., 0.
        for v_image, v_mask in self.val_datasets:
            _bce, _dice = self.valid_batch(v_image, v_mask)
            bce_loss += _bce
            dice_loss += _dice
        bce_loss /= len(self.val_datasets)
        dice_loss /= len(self.val_datasets)
        return bce_loss, dice_loss

    @tf.function
    def valid_batch(self, image, mask):
        with tf.GradientTape() as tape:
            pred = self.model(image)
            bce_loss = self.bce(mask, pred)
            dice_loss = self.dice(mask, pred)
        return bce_loss, dice_loss
