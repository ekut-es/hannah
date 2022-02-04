import copy
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from hydra.utils import get_class, instantiate
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from .base import SearchTask


class DartsSearchTask(SearchTask):
    def __init__(self, budget=2000, *args, **kwargs):
        super().__init__(*args, budget=budget, **kwargs)

    def fit(self, module: LightningModule):
        architect = Architect(module, 0.9, 1e-4)
        alpha_optimizer = torch.optim.SGD(lr=0.001, momentum=0.9, weight_decay=3e-4, params=architect.alphas(module))
        darts_opt = DartsOptimizerCallback(alpha_optimizer, architect)
        trainer = instantiate(self.config.trainer, callbacks=[darts_opt], limit_train_batches=10, limit_val_batches=100)
        trainer.fit(module)

    def run(self):

        # Prepare dataset
        get_class(self.config.dataset.cls).prepare(self.config.dataset)

        # Instantiate Dataset
        test_set, val_set, train_set = get_class(self.config.dataset.cls).splits(
            self.config.dataset
        )

        self.search_space.prepare_weight_sharing([1] + train_set.size())

        for i in range(self.budget):
            model = self.search_space.sample([1] + train_set.size())

            example_input_array = torch.rand([1] + train_set.size())
            module = instantiate(
                self.config.module,
                model=model,
                dataset=self.config.dataset,
                optimizer=self.config.optimizer,
                features=self.config.features,
                normalizer=self.config.get("normalizer", None),
                scheduler=self.config.scheduler,
                example_input_array=example_input_array,
                num_classes=len(train_set.class_names),
                _recursive_=False,
            )
            module.setup('stage')
            self.fit(module)


class DartsOptimizerCallback(Callback):
    def __init__(self, optimizer, architect, lr=1e-3) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.lr = lr
        self.architect = architect
        self.w_optim = None
        self.val_dataloader = None

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.architect.to(pl_module.device)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_dataloader = trainer.val_dataloaders[0]
        self.val_dataloader_iter = iter(self.val_dataloader)
        self.w_optim = trainer.optimizers[0]

    def on_train_batch_start(self,
                             trainer: "pl.Trainer",
                             pl_module: "pl.LightningModule",
                             batch: Any,
                             batch_idx: int,
                             unused: Optional[int] = 0) -> None:
        x_trn, x_trn_len, y_trn, y_trn_len = batch
        x_val, x_val_len, y_val, y_val_len = next(self.val_dataloader_iter)
        x_val = x_val.to(pl_module.device)
        y_val = y_val.to(pl_module.device)
        self.optimizer.zero_grad()
        self.architect.unrolled_backward(x_trn, y_trn, x_val, y_val, self.lr, self.w_optim)
        self.optimizer.step()


# from https://github.com/khanrc/pt.darts/blob/master/architect.py
class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def loss(self, net, X, y):
        logits = net.forward(X)
        return net.criterion(logits, y.squeeze())

    def to(self, device):
        self.net.to(device)
        self.v_net.to(device)

    def weights(self, net):
        return [p for n, p in net.named_parameters() if 'alpha' not in n]

    def alphas(self, net):
        return [p for n, p in net.named_parameters() if 'alpha' in n]

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)
        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient
        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.loss(self.net, trn_X, trn_y)  # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.weights(self.net))

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.weights(self.net), self.weights(self.v_net), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.alphas(self.net), self.alphas(self.v_net)):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        loss = self.loss(self.v_net, val_X, val_y)  # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.alphas(self.v_net))
        v_weights = tuple(self.weights(self.v_net))
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.alphas(self.net), dalpha, hessian):
                alpha.grad = da - xi*h

    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.weights(self.net), dw):
                p += eps * d
        loss = self.loss(self.net, trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.alphas(self.net))  # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.weights(self.net), dw):
                p -= 2. * eps * d
        loss = self.loss(self.net, trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.alphas(self.net))  # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.weights(self.net), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
