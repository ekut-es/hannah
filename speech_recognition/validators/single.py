import shutil
import logging
import torch
from hydra.utils import instantiate

from .base import BaseValidator


class SingleValidator(BaseValidator):
    def training_core(self, lit_module, profiler,
                      callbacks, checkpoint_callback, opt_callback,
                      logger):
        # INIT PYTORCH-LIGHTNING
        lit_trainer = instantiate(
            self.config.trainer,
            profiler=profiler,
            callbacks=callbacks,
            checkpoint_callback=checkpoint_callback,
            logger=logger,
        )

        if self.config["auto_lr"]:
            # run lr finder (counts as one epoch)
            lr_finder = lit_trainer.lr_find(lit_module)

            # inspect results
            fig = lr_finder.plot()
            fig.savefig("./learning_rate.png")

            # recreate module with updated config
            suggested_lr = lr_finder.suggestion()
            self.config["lr"] = suggested_lr

        # PL TRAIN
        lit_trainer.fit(lit_module)
        ckpt_path = "best"

        if lit_trainer.fast_dev_run:
            logging.warning(
                "Trainer is in fast dev run mode, switching off loading of best model for test"
            )
            ckpt_path = None

        # PL TEST
        lit_trainer.test(ckpt_path=ckpt_path, verbose=False)
        if not lit_trainer.fast_dev_run:
            lit_module.save()
            if checkpoint_callback and checkpoint_callback.best_model_path:
                shutil.copy(checkpoint_callback.best_model_path, "best.ckpt")

        return opt_callback.result()

    def eval(self, model_name):
        lit_trainer, lit_module, profiler = build_trainer(model_name,
                                                          self.config)
        test_loader = lit_module.test_dataloader()

        lit_module.eval()
        lit_module.freeze()

        results = None
        for batch in test_loader:
            result = lit_module.forward(batch[0])
            if results is None:
                results = result
            else:
                results = torch.cat([results, result])
        return results
