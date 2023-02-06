import torch
import torch.nn.functional as F
import logging


msglogger = logging.getLogger(__name__)


class HardLabeling(torch.nn.Module):
    def __init__(self, model, loss=None, th_confdc_pos=None, th_uncert_pos=None, th_confdc_neg=None, th_uncert_neg=None):
        super().__init__()
        self.model = model
        self.loss = loss if loss is not None else torch.nn.CrossEntropyLoss()
        self.tau_p = th_confdc_pos
        self.tau_n = th_confdc_neg
        self.kappa_p = th_uncert_pos
        self.kappa_n = th_uncert_neg

        if self.tau_p is None:
            msglogger.warning("Performing Pseudo-Labeling without confidence threshold.")
        if (self.kappa_p is not None or self.kappa_n is not None) and len(self.get_dropout_layers()) == 0:
            # TODO use augmentations for uncertainty?
            msglogger.critical("Monte Carlo uncertainty threshold is specified, but no dropout layers in model. Set model.drop_rate.")

    def forward(self, unlabeled_data: torch.Tensor) -> torch.Tensor:
        """Calculate pseudo label loss from unlabeled data."""
        x = unlabeled_data["data"]
        prediction = self.model(x).logits
        prediction_sm = torch.nn.functional.softmax(prediction, dim=-1)

        if self.tau_p is not None:  # thresholded positive learning
            mask_pos = prediction_sm.ge(self.tau_p).any(dim=-1)
            if self.kappa_p is not None:
                # TODO x[mask_pos, :] for efficiency?
                uncertainty = self.compute_uncertainty(x)
                mask_uncert_pos = uncertainty.le(self.kappa_p)
                mask_pos = torch.logical_and(mask_pos, mask_uncert_pos)
            pseudo_labels_pos = torch.masked_select(prediction_sm.argmax(dim=-1), mask_pos)
            if torch.numel(pseudo_labels_pos) > 0:
                loss = self.loss(prediction[mask_pos, :], pseudo_labels_pos)
            else:
                loss = 0.0

            if self.tau_n is not None:  # thresholded negative learning
                mask_neg = prediction_sm.le(self.tau_n).any(dim=-1)
                if self.kappa_n is not None:
                    uncertainty = self.compute_uncertainty(x)
                    mask_uncert_neg = uncertainty.le(self.kappa_n)
                    mask_neg = torch.logical_and(mask_neg, mask_uncert_neg)
                pseudo_labels_neg = torch.masked_select(prediction_sm.argmax(dim=-1), mask_neg)
                if torch.numel(pseudo_labels_neg) > 0:
                    # negative learning requires multiclass and multilabel loss
                    ones = torch.ones_like(prediction[mask_neg, :])
                    prediction_sm_inv = ones - prediction_sm[mask_neg, :]
                    prediction_onehot = F.one_hot(pseudo_labels_neg, num_classes=prediction.size(dim=-1))
                    pseudo_labels_neg_inv = ones - prediction_onehot
                    CRE_inv = -torch.sum(pseudo_labels_neg_inv * torch.log(prediction_sm_inv), dim=-1)
                    loss += torch.mean(CRE_inv / pseudo_labels_neg.sum(dim=-1))

        else:  # positive learning without thresholds
            pseudo_labels = prediction_sm.argmax(dim=-1)
            loss = loss(prediction, pseudo_labels)
        return loss

    def compute_uncertainty(self, data: torch.Tensor, num_forward_passes: int = 10) -> torch.Tensor:
        """Compute Monte Carlo uncertainty using standard deviation."""
        # deactivate batch normalization and other layers, enable dropout layers during test time
        self.model.eval()
        for layer in self.get_dropout_layers():
            layer.train()
        softmax = torch.nn.Softmax(dim=-1)
        with torch.no_grad():
            predictions = [softmax(self.model(data).logits) for i in range(num_forward_passes)]
        std = torch.stack(predictions)[:, :, :1].std(dim=0)
        self.model.train()
        return std.reshape(-1)

    def get_dropout_layers(self):
        """Returns all model layers of class dropout, dropblock."""
        classes = ["dropout", "drop_block", "dropblock"]  # "drop_path", "droppath"
        layers = [m for m in self.model.modules() if any(c in m.__class__.__name__.lower() for c in classes)]
        return layers
