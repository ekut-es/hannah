import shutil
import logging
import torch
from hydra.utils import instantiate

from .base import BaseValidator


class CrossValidator(BaseValidator):
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

        results = list()

        for i, (train, val, test) in enumerate(zip(lit_module.train_dataloader(),
                                                   lit_module.val_dataloader(),
                                                   lit_module.test_dataloader())):

            lit_module_cross_val = instantiate(
                self.config.module,
                dataset=self.config.dataset,
                model=self.config.model,
                optimizer=self.config.optimizer,
                features=self.config.features,
                scheduler=self.config.get("scheduler", None),
                normalizer=self.config.get("normalizer", None),
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
            lit_trainer.fit(lit_module_cross_val,
                            train_dataloader=train, val_dataloaders=val)
            ckpt_path = "best"

            if lit_trainer.fast_dev_run:
                logging.warning(
                    "Trainer is in fast dev run mode, switching off loading of best model for test"
                )
                ckpt_path = None

            # PL TEST
            lit_trainer.test(test_dataloaders=test,
                             ckpt_path=ckpt_path, verbose=False)
            if not lit_trainer.fast_dev_run:
                lit_module_cross_val.save()
                if checkpoint_callback and checkpoint_callback.best_model_path:
                    shutil.copy(checkpoint_callback.best_model_path, "best.ckpt")

            results += [opt_callback.result()]

        return results

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
