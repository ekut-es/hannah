import logging
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
import omegaconf
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from hydra.utils import instantiate
import torch
from hannah.callbacks.optimization import HydraOptCallback
from hannah.utils.utils import common_callbacks
from pytorch_lightning import Trainer


msglogger = logging.getLogger(__name__)


class ProgressiveShrinkingModelTrainer:
    def __init__(self,
                 parent_config=None,
                 epochs_warmup=10,
                 epochs_kernel_step=10,
                 epochs_depth_step=10,
                 epochs_width_step=10,
                 epochs_dilation_step=10,
                 epochs_grouping_step=10,
                 epochs_dsc_step=10,
                 epochs_tuning_step=0,
                 elastic_kernels_allowed=False,
                 elastic_depth_allowed=False,
                 elastic_width_allowed=False,
                 elastic_dilation_allowed=False,
                 elastic_grouping_allowed=False,
                 elastic_dsc_allowed=False,
                 evaluate=True,
                 random_evaluate=True,
                 random_eval_number=100,
                 extract_model_config=False,
                 warmup_model_path="",) -> None:
        self.config = parent_config
        self.epochs_warmup = epochs_warmup
        self.epochs_kernel_step = epochs_kernel_step
        self.epochs_depth_step = epochs_depth_step
        self.epochs_width_step = epochs_width_step
        self.epochs_dilation_step = epochs_dilation_step
        self.epochs_grouping_step = epochs_grouping_step
        self.epochs_dsc_step = epochs_dsc_step
        self.epochs_tuning_step = epochs_tuning_step
        self.elastic_kernels_allowed = elastic_kernels_allowed
        self.elastic_depth_allowed = elastic_depth_allowed
        self.elastic_width_allowed = elastic_width_allowed
        self.elastic_dilation_allowed = elastic_dilation_allowed
        self.elastic_grouping_allowed = elastic_grouping_allowed
        self.elastic_dsc_allowed = elastic_dsc_allowed

        self.evaluate = evaluate
        self.random_evaluate = random_evaluate
        self.random_eval_number = random_eval_number
        self.warmup_model_path = warmup_model_path
        self.extract_model_config = extract_model_config

    def build_model(self):
        config = OmegaConf.create(self.config)
        # logger = TensorBoardLogger(".")

        seed = config.get("seed", 1234)
        if isinstance(seed, list) or isinstance(seed, omegaconf.ListConfig):
            seed = seed[0]
        seed_everything(seed, workers=True)

        if not torch.cuda.is_available():
            config.trainer.gpus = None

        callbacks = common_callbacks(config)
        opt_monitor = config.get("monitor", ["val_error"])
        opt_callback = HydraOptCallback(monitor=opt_monitor)
        callbacks.append(opt_callback)
        checkpoint_callback = instantiate(config.checkpoint)
        callbacks.append(checkpoint_callback)
        self.config = config
        # trainer will be initialized by rebuild_trainer
        self.trainer = None
        model = instantiate(
            config.module,
            dataset=config.dataset,
            model=config.model,
            optimizer=config.optimizer,
            features=config.features,
            scheduler=config.get("scheduler", None),
            normalizer=config.get("normalizer", None),
            _recursive_=False,
        )
        model.setup("fit")
        return model

    def run_training(self, model):
        ofa_model = model.model

        self.kernel_step_count = ofa_model.ofa_steps_kernel
        self.depth_step_count = ofa_model.ofa_steps_depth
        self.width_step_count = ofa_model.ofa_steps_width
        self.dilation_step_count = ofa_model.ofa_steps_dilation
        self.grouping_step_count = ofa_model.ofa_steps_grouping
        self.dsc_step_count = ofa_model.ofa_steps_dsc
        ofa_model.elastic_kernels_allowed = self.elastic_kernels_allowed
        ofa_model.elastic_depth_allowed = self.elastic_depth_allowed
        ofa_model.elastic_width_allowed = self.elastic_width_allowed
        ofa_model.elastic_dilation_allowed = self.elastic_dilation_allowed
        ofa_model.elastic_grouping_allowed = self.elastic_grouping_allowed
        ofa_model.elastic_dsc_allowed = self.elastic_dsc_allowed
        ofa_model.full_config = self.config["model"]

        logging.info("Kernel Steps: %d", self.kernel_step_count)
        logging.info("Depth Steps: %d", self.depth_step_count)
        logging.info("Width Steps: %d", self.width_step_count)
        logging.info("Grouping Steps: %d", self.grouping_step_count)
        logging.info("DSC Steps: %d", self.dsc_step_count)
        # logging.info("dsc: %d", self.grouping_step_count)

        self.submodel_metrics_csv = ""
        self.random_metrics_csv = ""

        if self.elastic_width_allowed:
            self.submodel_metrics_csv += "width, "
            self.random_metrics_csv += "width_steps, "

        if self.elastic_kernels_allowed:
            self.submodel_metrics_csv += "kernel, "
            self.random_metrics_csv += "kernel_steps, "

        if self.elastic_dilation_allowed:
            self.submodel_metrics_csv += "dilation, "
            self.random_metrics_csv += "dilation_steps, "

        if self.elastic_depth_allowed:
            self.submodel_metrics_csv += "depth, "
            self.random_metrics_csv += "depth, "

        if self.elastic_grouping_allowed:
            self.submodel_metrics_csv += "grouping, "
            self.random_metrics_csv += "group_steps, "

        if self.elastic_dsc_allowed:
            self.submodel_metrics_csv += "dsc, "
            self.random_metrics_csv += "dsc, "

        if (
            self.elastic_width_allowed
            | self.elastic_kernels_allowed
            | self.elastic_dilation_allowed
            | self.elastic_depth_allowed
            | self.elastic_grouping_allowed
            | self.elastic_dsc_allowed
        ):
            self.submodel_metrics_csv += (
                "acc, total_macs, total_weights, torch_params\n"
            )
            self.random_metrics_csv += "acc, total_macs, total_weights, torch_params\n"

        # self.random_metrics_csv = "width_steps, depth, kernel_steps, acc, total_macs, total_weights, torch_params\n"

        logging.info("Once for all Model:\n %s", str(ofa_model))
        # TODO Warmup DSC on or off?
        self.warmup(model, ofa_model)
        ofa_model.reset_shrinking()

        self.train_elastic_kernel(model, ofa_model)
        ofa_model.reset_shrinking()
        self.train_elastic_dilation(model, ofa_model)
        ofa_model.reset_shrinking()
        self.train_elastic_depth(model, ofa_model)
        ofa_model.reset_shrinking()
        self.train_elastic_width(model, ofa_model)
        ofa_model.reset_shrinking()
        self.train_elastic_grouping(model, ofa_model)
        ofa_model.reset_shrinking()
        self.train_elastic_dsc(model, ofa_model)
        ofa_model.reset_shrinking()

        if self.evaluate:
            self.eval_model(model, ofa_model)

            if self.random_evaluate:
                # save random metrics
                msglogger.info("\n%s", self.random_metrics_csv)
                with open("OFA_random_sample_metrics.csv", "w") as f:
                    f.write(self.random_metrics_csv)
            # save self.submodel_metrics_csv
            msglogger.info("\n%s", str(self.submodel_metrics_csv))
            with open("OFA_elastic_metrics.csv", "w") as f:
                f.write(self.submodel_metrics_csv)



    def warmup(self, model, ofa_model):
        """
        > The function rebuilds the trainer with the warmup epochs, fits the model,
        validates the model, and then calls the on_warmup_end() function to
        change some internal variables

        :param model: the model to be trained
        :param ofa_model: the model that we want to train
        """
        # warm-up.
        self.rebuild_trainer("warmup", self.epochs_warmup)
        if self.epochs_warmup > 0 and self.warmup_model_path == "":
            self.trainer.fit(model)
            ckpt_path = "best"
        elif self.warmup_model_path != "":
            ckpt_path = self.warmup_model_path
        self.trainer.validate(ckpt_path=ckpt_path, model=model, verbose=True)
        ofa_model.on_warmup_end()
        ofa_model.reset_validation_model()
        msglogger.info("OFA completed warm-up.")

    def train_elastic_width(self, model, ofa_model):
        """
        > The function trains the model for a number of epochs, then adds a width
        step, then trains the model for a number of epochs, then adds a width step,
        and so on

        :param model: the model to train
        :param ofa_model: the model that will be trained
        """
        if self.elastic_width_allowed:
            # train elastic width
            # first, run channel priority computation
            ofa_model.progressive_shrinking_compute_channel_priorities()
            for current_width_step in range(1, self.width_step_count):
                # add a width step
                ofa_model.progressive_shrinking_add_width()
                if self.epochs_width_step > 0:
                    self.rebuild_trainer(
                        f"width_{current_width_step}", self.epochs_width_step
                    )
                    self.trainer.fit(model)
                    ckpt_path = "best"
                    self.trainer.validate(ckpt_path=ckpt_path, verbose=True)
            msglogger.info("OFA completed width steps.")

    def train_elastic_depth(self, model, ofa_model):
        """
        > The function trains the model for a number of epochs, then progressively
        shrinks the depth of the model, and trains the model for a number of epochs
        again

        :param model: the model to train
        :param ofa_model: the model to be trained
        """
        if self.elastic_depth_allowed:
            # train elastic depth
            for current_depth_step in range(1, self.depth_step_count):
                # add a depth reduction step
                ofa_model.progressive_shrinking_add_depth()
                if self.epochs_depth_step > 0:
                    self.rebuild_trainer(
                        f"depth_{current_depth_step}", self.epochs_depth_step
                    )
                    self.trainer.fit(model)
                    ckpt_path = "best"
                    self.trainer.validate(ckpt_path=ckpt_path, verbose=True)
            msglogger.info("OFA completed depth steps.")

    def train_elastic_kernel(self, model, ofa_model):
        """
        > The function trains the elastic kernels by progressively shrinking the
        model and training the model for a number of epochs and repeats this process
        until the number of kernel steps is reached

        :param model: the model to train
        :param ofa_model: the model that will be trained
        """
        if self.elastic_kernels_allowed:
            # train elastic kernels
            for current_kernel_step in range(1, self.kernel_step_count):
                # add a kernel step
                ofa_model.progressive_shrinking_add_kernel()
                if self.epochs_kernel_step > 0:
                    self.rebuild_trainer(
                        f"kernel_{current_kernel_step}", self.epochs_kernel_step
                    )
                    self.trainer.fit(model)
                    ckpt_path = "best"
                    self.trainer.validate(ckpt_path=ckpt_path, verbose=True)
            msglogger.info("OFA completed kernel matrices.")

    def train_elastic_dilation(self, model, ofa_model):
        """
        > The function trains the model for a number of epochs, then adds a dilation
        step, and trains the model for a number of epochs, and repeats this process
        until the number of dilation steps is reached

        :param model: the model to be trained
        :param ofa_model: the model that will be trained
        """
        if self.elastic_dilation_allowed:
            # train elastic kernels
            for current_dilation_step in range(1, self.dilation_step_count):
                # add a kernel step
                ofa_model.progressive_shrinking_add_dilation()
                if self.epochs_dilation_step > 0:
                    self.rebuild_trainer(
                        f"kernel_{current_dilation_step}", self.epochs_dilation_step
                    )
                    self.trainer.fit(model)
                    ckpt_path = "best"
                    self.trainer.validate(ckpt_path=ckpt_path, verbose=True)
            msglogger.info("OFA completed dilation matrices.")

    def train_elastic_grouping(self, model, ofa_model):
        """
        > The function trains the model for a number of epochs, then adds a group
        step, and trains the model for a number of epochs, and repeats this process
        until the number of group steps is reached

        :param model: the model to be trained
        :param ofa_model: the model that will be trained
        """
        if self.elastic_grouping_allowed:
            # train elastic groups
            for current_grouping_step in range(1, self.grouping_step_count):
                # add a group step
                ofa_model.progressive_shrinking_add_group()
                if self.epochs_grouping_step > 0:
                    self.rebuild_trainer(
                        f"group_{current_grouping_step}", self.epochs_grouping_step
                    )
                    self.trainer.fit(model)
                    ckpt_path = "best"
                    self.trainer.validate(ckpt_path=ckpt_path, verbose=True)
            msglogger.info("OFA completed grouping matrices.")

    def train_elastic_dsc(self, model, ofa_model):
        """
        > The function trains the model for a number of epochs, then adds a dsc
        step (turns Depthwise Separable Convolution on and off), and trains the model for a number of epochs, and repeats this process
        until the number of dsc steps is reached

        :param model: the model to be trained
        :param ofa_model: the model that will be trained
        """
        if self.elastic_dsc_allowed is True:
            # train elastic groups
            for current_dsc_step in range(1, self.dsc_step_count):
                # add a group step
                ofa_model.progressive_shrinking_add_dsc()
                if self.epochs_dsc_step > 0:
                    self.rebuild_trainer(
                        f"dsc_{current_dsc_step}", self.epochs_dsc_step
                    )
                    self.trainer.fit(model)
                    ckpt_path = "best"
                    self.trainer.validate(ckpt_path=ckpt_path, verbose=True)
            msglogger.info("OFA completed dsc matrices.")

    def eval_elastic_width(
        self,
        method_stack,
        method_index,
        lightning_model,
        model,
        trainer_path,
        loginfo_output,
        metrics_output,
        metrics_csv,
    ):
        """
        > This function steps down the width of the model, and then calls the next
        method in the stack

        :param method_stack: a list of methods that will be called in order
        :param method_index: The index of the current method in the method stack
        :param lightning_model: the lightning model to be trained
        :param model: the model to be trained
        :param trainer_path: The path to the trainer
        :param loginfo_output: This is the string that will be printed to the
        console
        :param metrics_output: a string that will be written to the metrics csv file
        :param metrics_csv: a string that contains the metrics for the current model
        :return: The metrics_csv is being returned.
        """
        model.reset_all_widths()
        method = method_stack[method_index]

        for current_width_step in range(self.width_step_count):
            if current_width_step > 0:
                # iteration 0 is the full model with no stepping
                model.step_down_all_channels()

            trainer_path_tmp = trainer_path + f"W {current_width_step}, "
            loginfo_output_tmp = loginfo_output + f"Width {current_width_step}, "
            metrics_output_tmp = metrics_output + f"{current_width_step}, "

            metrics_csv = method(
                method_stack,
                method_index + 1,
                lightning_model,
                model,
                trainer_path_tmp,
                loginfo_output_tmp,
                metrics_output_tmp,
                metrics_csv,
            )

        return metrics_csv

    def eval_elastic_kernel(
        self,
        method_stack,
        method_index,
        lightning_model,
        model,
        trainer_path,
        loginfo_output,
        metrics_output,
        metrics_csv,
    ):
        """
        > This function steps down the kernel size of the model, and then calls the
        next method in the stack

        :param method_stack: The list of methods to be called
        :param method_index: The index of the current method in the method stack
        :param lightning_model: the lightning model to be trained
        :param model: the model to be trained
        :param trainer_path: The path to the trainer
        :param loginfo_output: This is the string that will be printed to the
        console
        :param metrics_output: This is the string that will be printed to the
        console
        :param metrics_csv: a string that contains the metrics for the current model
        :return: The metrics_csv is being returned.
        """
        model.reset_all_kernel_sizes()
        method = method_stack[method_index]

        for current_kernel_step in range(self.kernel_step_count):
            if current_kernel_step > 0:
                # iteration 0 is the full model with no stepping
                model.step_down_all_kernels()

            trainer_path_tmp = trainer_path + f"K {current_kernel_step}, "
            loginfo_output_tmp = loginfo_output + f"Kernel {current_kernel_step}, "
            metrics_output_tmp = metrics_output + f"{current_kernel_step}, "

            metrics_csv = method(
                method_stack,
                method_index + 1,
                lightning_model,
                model,
                trainer_path_tmp,
                loginfo_output_tmp,
                metrics_output_tmp,
                metrics_csv,
            )

        return metrics_csv

    def eval_elastic_dilation(
        self,
        method_stack,
        method_index,
        lightning_model,
        model,
        trainer_path,
        loginfo_output,
        metrics_output,
        metrics_csv,
    ):
        """
        > This function evaluates the model with a different dilation size for each
        layer

        :param method_stack: The list of methods to be called
        :param method_index: The index of the method in the method stack
        :param lightning_model: the lightning model to be trained
        :param model: the model to be evaluated
        :param trainer_path: The path to the trainer
        :param loginfo_output: This is the string that will be printed to the
        console
        :param metrics_output: a string that will be written to the metrics csv file
        :param metrics_csv: a string that contains the csv data for the metrics
        :return: The metrics_csv is being returned.
        """
        model.reset_all_dilation_sizes()
        method = method_stack[method_index]

        for current_dilation_step in range(self.dilation_step_count):
            if current_dilation_step > 0:
                # iteration 0 is the full model with no stepping
                model.step_down_all_dilations()

            trainer_path_tmp = trainer_path + f"K {current_dilation_step}, "
            loginfo_output_tmp = loginfo_output + f"Dilation {current_dilation_step}, "
            metrics_output_tmp = metrics_output + f"{current_dilation_step}, "

            metrics_csv = method(
                method_stack,
                method_index + 1,
                lightning_model,
                model,
                trainer_path_tmp,
                loginfo_output_tmp,
                metrics_output_tmp,
                metrics_csv,
            )

        return metrics_csv

    def eval_elastic_depth(
        self,
        method_stack,
        method_index,
        lightning_model,
        model,
        trainer_path,
        loginfo_output,
        metrics_output,
        metrics_csv,
    ):
        """
        > This function will run the next method in the stack for each depth step,
        and then return the metrics_csv

        :param method_stack: The list of methods to be called
        :param method_index: The index of the current method in the method stack
        :param lightning_model: the lightning model to be trained
        :param model: The model to be trained
        :param trainer_path: The path to the trainer, which is used to save the
        model
        :param loginfo_output: This is the string that will be printed to the
        console
        :param metrics_output: This is the string that will be printed to the
        console
        :param metrics_csv: This is the CSV file that we're writing to
        :return: The metrics_csv is being returned.
        """
        model.reset_active_depth()
        method = method_stack[method_index]

        for current_depth_step in range(self.depth_step_count):
            if current_depth_step > 0:
                # iteration 0 is the full model with no stepping
                model.active_depth -= 1

            trainer_path_tmp = trainer_path + f"D {current_depth_step}, "
            loginfo_output_tmp = loginfo_output + f"Depth {current_depth_step}, "
            metrics_output_tmp = metrics_output + f"{current_depth_step}, "

            metrics_csv = method(
                method_stack,
                method_index + 1,
                lightning_model,
                model,
                trainer_path_tmp,
                loginfo_output_tmp,
                metrics_output_tmp,
                metrics_csv,
            )

        return metrics_csv

    def eval_elastic_grouping(
        self,
        method_stack,
        method_index,
        lightning_model,
        model,
        trainer_path,
        loginfo_output,
        metrics_output,
        metrics_csv,
    ):
        """
        > This function evaluates the model with a different group size for each
        layer

        :param method_stack: The list of methods to be called
        :param method_index: The index of the method in the method stack
        :param lightning_model: the lightning model to be trained
        :param model: the model to be evaluated
        :param trainer_path: The path to the trainer
        :param loginfo_output: This is the string that will be printed to the
        console
        :param metrics_output: a string that will be written to the metrics csv file
        :param metrics_csv: a string that contains the csv data for the metrics
        :return: The metrics_csv is being returned.
        """
        model.reset_all_group_sizes()
        method = method_stack[method_index]
        for current_group_step in range(self.grouping_step_count):
            if current_group_step > 0:
                # iteration 0 is the full model with no stepping
                model.step_down_all_groups()

            trainer_path_tmp = trainer_path + f"G {current_group_step}, "
            loginfo_output_tmp = loginfo_output + f"Group {current_group_step}, "
            metrics_output_tmp = metrics_output + f"{current_group_step}, "

            metrics_csv = method(
                method_stack,
                method_index + 1,
                lightning_model,
                model,
                trainer_path_tmp,
                loginfo_output_tmp,
                metrics_output_tmp,
                metrics_csv,
            )

        return metrics_csv

    def eval_elastic_dsc(
        self,
        method_stack,
        method_index,
        lightning_model,
        model,
        trainer_path,
        loginfo_output,
        metrics_output,
        metrics_csv,
    ):
        """
        > This function evaluates the model with a different dsc  for each
        layer

        :param method_stack: The list of methods to be called
        :param method_index: The index of the method in the method stack
        :param lightning_model: the lightning model to be trained
        :param model: the model to be evaluated
        :param trainer_path: The path to the trainer
        :param loginfo_output: This is the string that will be printed to the
        console
        :param metrics_output: a string that will be written to the metrics csv file
        :param metrics_csv: a string that contains the csv data for the metrics
        :return: The metrics_csv is being returned.
        """
        model.reset_all_dsc()
        method = method_stack[method_index]
        for current_dsc_step in range(self.dsc_step_count):
            if current_dsc_step > 0:
                # iteration 0 is the full model with no stepping
                model.step_down_all_dsc()

            trainer_path_tmp = trainer_path + f"DSC {current_dsc_step}, "
            loginfo_output_tmp = loginfo_output + f"DSC {current_dsc_step}, "
            metrics_output_tmp = metrics_output + f"{current_dsc_step}, "

            metrics_csv = method(
                method_stack,
                method_index + 1,
                lightning_model,
                model,
                trainer_path_tmp,
                loginfo_output_tmp,
                metrics_output_tmp,
                metrics_csv,
            )

        return metrics_csv

    def eval_single_model(
        self,
        method_stack,
        method_index,
        lightning_model,
        model,
        trainer_path,
        loginfo_output,
        metrics_output,
        metrics_csv,
    ):
        """
        > This function takes in a model, a trainer, and a bunch of other stuff,
        evaluates the model and tracks the results in der in the given strings and
        returns a string of metrics

        :param method_stack: The list of methods that we're evaluating
        :param method_index: The index of the method in the method stack
        :param lightning_model: the lightning model that we want to evaluate
        :param model: The model to be evaluated
        :param trainer_path: the path to the trainer object
        :param loginfo_output: This is the string that will be printed to the
        console when the model is being evaluated
        :param metrics_output: This is the string that will be written to the
        metrics file. It contains the method name, the method index, and the method
        stack
        :param metrics_csv: a string that will be written to a csv file
        :return: The metrics_csv is being returned.
        """
        self.rebuild_trainer(trainer_path, self.epochs_tuning_step, tensorboard=False)
        msglogger.info(loginfo_output)

        validation_model = model.build_validation_model()

        lightning_model.model = validation_model
        assert model.eval_mode is True

        if self.epochs_tuning_step > 0:
            self.trainer.fit(lightning_model)

        validation_results = self.trainer.validate(
            lightning_model, ckpt_path=None, verbose=True
        )

        lightning_model.model = model

        metrics_csv += metrics_output
        results = validation_results[0]
        torch_params = model.get_validation_model_weight_count()
        metrics_csv += f"{results['val_accuracy']}, {results['total_macs']}, {results['total_weights']}, {torch_params}"
        metrics_csv += "\n"
        return metrics_csv

    def eval_model(self, lightning_model, model):
        """
        First the method stack for the evaluation ist build and then it is according to this evaluated

        :param lightning_model: the lightning model
        :param model: the model to be evaluated
        """
        # disable sampling in forward during evaluation.
        model.eval_mode = True

        eval_methods = []

        if self.elastic_width_allowed:
            eval_methods.append(self.eval_elastic_width)

        if self.elastic_kernels_allowed:
            eval_methods.append(self.eval_elastic_kernel)

        if self.elastic_dilation_allowed:
            eval_methods.append(self.eval_elastic_dilation)

        if self.elastic_depth_allowed:
            eval_methods.append(self.eval_elastic_depth)

        if self.elastic_grouping_allowed:
            eval_methods.append(self.eval_elastic_grouping)

        if self.elastic_dsc_allowed:
            eval_methods.append(self.eval_elastic_dsc)

        if len(eval_methods) > 0:
            eval_methods.append(self.eval_single_model)
            self.submodel_metrics_csv = eval_methods[0](
                eval_methods,
                1,
                lightning_model,
                model,
                "Eval ",
                "OFA validating ",
                "",
                self.submodel_metrics_csv,
            )

        if self.random_evaluate:
            self.eval_random_combination(lightning_model, model)

        model.eval_mode = False

    def eval_random_combination(self, lightning_model, model):
        # sample a few random combinations

        random_eval_number = self.random_eval_number
        prev_max_kernel = model.sampling_max_kernel_step
        prev_max_depth = model.sampling_max_depth_step
        prev_max_width = model.sampling_max_width_step
        prev_max_dilation = model.sampling_max_dilation_step
        prev_max_grouping = model.sampling_max_grouping_step
        prev_max_dsc = model.sampling_max_dsc_step
        model.sampling_max_kernel_step = model.ofa_steps_kernel - 1
        model.sampling_max_dilation_step = model.ofa_steps_dilation - 1
        model.sampling_max_depth_step = model.ofa_steps_depth - 1
        model.sampling_max_width_step = model.ofa_steps_width - 1
        model.sampling_max_grouping_step = model.ofa_steps_grouping - 1
        model.sampling_max_dsc_step = model.ofa_steps_dsc - 1
        assert model.eval_mode is True
        for i in range(random_eval_number):
            model.reset_validation_model()
            random_state = model.sample_subnetwork()

            loginfo_output = f"OFA validating random sample:\n{random_state}"
            trainer_path = "Eval random sample: "
            metrics_output = ""

            if self.elastic_width_allowed:
                selected_widths = random_state["width_steps"]
                selected_widths_string = str(selected_widths).replace(",", ";")
                metrics_output += f"{selected_widths_string}, "
                trainer_path += f"Ws {selected_widths}, "

            if self.elastic_kernels_allowed:
                selected_kernels = random_state["kernel_steps"]
                selected_kernels_string = str(selected_kernels).replace(",", ";")
                metrics_output += f" {selected_kernels_string}, "
                trainer_path += f"Ks {selected_kernels}, "

            if self.elastic_dilation_allowed:
                selected_dilations = random_state["dilation_steps"]
                selected_dilations_string = str(selected_dilations).replace(",", ";")
                metrics_output += f" {selected_dilations_string}, "
                trainer_path += f"Dils {selected_dilations}, "

            if self.elastic_grouping_allowed:
                selected_groups = random_state["grouping_steps"]
                selected_groups_string = str(selected_groups).replace(",", ";")
                metrics_output += f" {selected_groups_string}, "
                trainer_path += f"Gs {selected_groups_string}, "

            if self.elastic_dsc_allowed:
                selected_dscs = random_state["dsc_steps"]
                selected_dscs_string = str(selected_dscs).replace(",", ";")
                metrics_output += f" {selected_dscs_string}, "
                trainer_path += f"DSCs {selected_dscs_string}, "

            if self.elastic_depth_allowed:
                selected_depth = random_state["depth_step"]
                trainer_path += f"D {selected_depth}, "
                metrics_output += f"{selected_depth}, "
            if self.extract_model_config:
                model.print_config("r" + str(i))

            self.random_metrics_csv = self.eval_single_model(
                None,
                None,
                lightning_model,
                model,
                trainer_path,
                loginfo_output,
                metrics_output,
                self.random_metrics_csv,
            )

        # revert to normal operation after eval.
        model.sampling_max_kernel_step = prev_max_kernel
        model.sampling_max_dilation_step = prev_max_dilation
        model.sampling_max_depth_step = prev_max_depth
        model.sampling_max_width_step = prev_max_width
        model.sampling_max_grouping_step = prev_max_grouping
        model.sampling_max_dsc_step = prev_max_dsc

    def rebuild_trainer(
        self, step_name: str, epochs: int = 1, tensorboard: bool = True
    ) -> Trainer:
        if tensorboard:
            logger = TensorBoardLogger(".", version=step_name)
        else:
            logger = CSVLogger(".", version=step_name)
        callbacks = common_callbacks(self.config)
        self.trainer = instantiate(
            self.config.trainer, callbacks=callbacks, logger=logger, max_epochs=epochs
        )
