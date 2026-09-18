from pytorch_lightning.callbacks import Callback, ModelCheckpoint

import subprocess

def get_callbacks(run_name, out_dir):
    loss_callback = LossCallback()

    #### Making a directory to store the intermediate model weights:
    temp_dir = f"{out_dir}/_temp/"
    subprocess.run(['mkdir', temp_dir])

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=temp_dir,
        filename=run_name + "--{epoch:02d}--{val_loss:.6f}",
        save_weights_only=False,  # Will reload the model!
    )

    return temp_dir, [loss_callback, checkpoint_callback]

class LossCallback(Callback):
    """ Largely for storing training QC metrics during training with pytorch lightning.
    """
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.lrs = []

    def on_train_epoch_end(self, trainer, pl_module):
        avg_train_loss = trainer.callback_metrics.get('train_loss')
        if avg_train_loss is not None:
            self.train_losses.append(avg_train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        avg_val_loss = trainer.callback_metrics.get('val_loss')
        if avg_val_loss is not None:
            avg_val_loss = avg_val_loss.item()
            self.val_losses.append(avg_val_loss)

        opt = trainer.optimizers[0]
        lr = opt.param_groups[0]['lr']
        if lr is not None:
            self.lrs.append( lr )
