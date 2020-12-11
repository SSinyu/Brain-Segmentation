from tqdm import tqdm
from pathlib import Path
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

        self.train_bce = metrics.Mean()
        self.train_dice = metrics.Mean()
        self.train_iou = metrics.MeanIoU(num_classes=2)

        if config["mode"] == "train":
            self.valid_datasets = DataLoader(
                "valid", **config["dataset"]
            )
        self.test_bce = metrics.Mean()
        self.test_dice = metrics.Mean()
        self.test_iou = metrics.MeanIoU(num_classes=2)

        self.get_ckpt_manager(config["save_path"])

    def get_ckpt_manager(self, ckpt_path, keep=5):
        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_path, max_to_keep=keep)

    def train(self):
        n_iters = len(self.datasets)
        valid_loss = 1e6
        for epoch in range(self.epochs):
            self.n_reset()
            for (i, (image, mask)) in enumerate(self.datasets):
                self.train_batch(image, mask)

                if (i+1) % self.print_iter == 0:
                    print(f"[{epoch+1}/{self.epochs}] Epoch, [{i+1}/{n_iters}] Iter ==> ", end=" ")
                    print(f"BCE Loss: {self.train_bce.result():.5f}", end="  ")
                    print(f"Dice Loss: {self.train_dice.result():.5f}", end="  ")
                    print(f"IoU: {self.train_iou.result():.5f}")

            self.test_steps(True)
            print(f"===== [{epoch+1}/{self.epochs}] Epoch")
            print(f"===== valid BCE: {self.test_bce.result():.5f}", end="  ")
            print(f"valid DICE: {self.test_dice.result():.5f}", end="  ")
            print(f"valid IoU: {self.test_iou.result():.5f}")

            valid_total = self.bce_w * self.test_bce.result().numpy() + \
                        self.dice_w * self.test_dice.result().numpy()
            if valid_total < valid_loss:
                print(f"Validation loss reduced from {valid_loss:.5f} to {valid_total:.5f}")
                valid_loss = valid_total
                ckpt_save_path = self.ckpt_manager.save()

            if (epoch+1) % self.save_epoch == 0:
                f = self.config["save_path"] / Path(f"epoch_{epoch+1}.h5")
                self.model.save_weights(str(f))
                print("save ", str(f))

    def test(self, ep=None):
        self.load_weight(ep)
        self.test_steps(False)
        print(f"==> test BCE: {self.test_bce.result():.5f}")
        print(f"==> test DICE: {self.test_dice.result():.5f}")
        print(f"==> test IoU: {self.test_iou.result():.5f}")

    def n_reset(self):
        self.train_bce.reset_states()
        self.train_dice.reset_states()
        self.train_iou.reset_states()
        self.test_bce.reset_states()
        self.test_dice.reset_states()
        self.test_iou.reset_states()

    def load_weight(self, ep=None):
        f = str(self.config["save_path"])
        if ep is None:
            self.ckpt.restore(tf.train.latest_checkpoint(f)).expect_partial()
        else:
            f = f / f"/epoch_{ep}.h5"
            self.model.load_weights(f)

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
        self.train_bce(bce_loss)
        self.train_dice(dice_loss)
        self.train_iou(mask, tf.sigmoid(pred))

    @tf.function
    def test_batch(self, image, mask):
        pred = self.model(image)
        bce_loss = self.bce(mask, pred)
        dice_loss = self.dice(mask, pred)
        self.test_bce(bce_loss)
        self.test_dice(dice_loss)
        self.test_iou(mask, tf.sigmoid(pred))

    def test_steps(self, valid=True):
        if valid:
            datasets = self.valid_datasets
        else:
            datasets = self.datasets
        for image, mask in tqdm(datasets):
            self.test_batch(image, mask)
