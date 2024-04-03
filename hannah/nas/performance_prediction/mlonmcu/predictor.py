import os
import logging
from pathlib import Path

import torch
import mlonmcu.context
from mlonmcu.session.run import RunStage


msglogger = logging.getLogger(__name__)


class MLonMCUPredictor():
    def __init__(self,
                 model_name,
                 metrics=["Cycles", "Total ROM", "Total RAM"],
                 platform="mlif",
                 backend="tvmaot",
                 target="etiss_pulpino",
                 frontend="onnx",
                 postprocess=None,
                 feature=None,
                 configs=None,
                 parallel=None,
                 progress=False,
                 verbose=False,):
        self.model_name = model_name

        self.metrics = metrics
        self.platform = platform
        self.backend = backend
        self.target = target
        self.frontend = frontend
        self.postprocess = postprocess
        self.feature = feature
        self.configs = configs
        self.parallel = parallel
        self.progress = progress
        self.verbose = verbose

    def predict(self, model, input):
        if hasattr(model, 'model'):
            model = model.model  # FIXME: Decide when to use pl_module and when to use model

        # Convert PyTorch model to ONNX
        ckpt_path = Path("mlonmcu")
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        model_path = os.path.join(ckpt_path, f"{self.model_name}.onnx")
        convert_to_onnx(model, input, model_path)

        # Run MLonMCU
        print("MLonMCU evaluating {}".format(self.model_name))
        if not os.path.isdir(ckpt_path):
            raise Exception("INVALID MODEL PATH: ", ckpt_path)
        with mlonmcu.context.MlonMcuContext() as context:
            session = context.create_session()
            run = session.create_run(features=[], config={})
            run.add_frontend_by_name("onnx", context=context)
            run.add_model_by_name(model_path, context=context)
            run.add_backend_by_name(self.backend, context=context)
            run.add_platform_by_name(self.platform, context=context)
            run.add_target_by_name(self.target, context=context)
            # run.add_feature_by_name("vext", context=context)
            session.process_runs(until=RunStage.RUN, context=context)
            report = session.get_reports()
            print(report)

        # Return a dict of metric values
        mlonmcu_metrics = report.df
        result = {}
        for metric in self.metrics:
            if metric in mlonmcu_metrics:
                result[metric] = float(mlonmcu_metrics[metric])
            else:
                # raise Exception("Metric is not supported by MLonMCU: ", metric)
                msglogger.info(f"WARNING: Metric {metric} is not supported by MLonMCU ")

        return result

    def load(self, result_folder):
        pass

    def update(self, new_data, input):
        pass


def convert_to_onnx(pytorch_model, sample_input, onnx_model_path):
    # Export pytorch model to onnx
    torch.onnx.export(
        model=pytorch_model,
        args=sample_input,
        f=onnx_model_path,  # save path
        verbose=False,
    )

    return onnx_model_path
