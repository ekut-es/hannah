import torch
import seaborn as sns
import matplotlib.pyplot as plt


class LogActDistributionHook:
    def __init__(self, name):
        self.name = name
        self.sum = None
        self.count = 0

    def __call__(self, _module, input, output):
        with torch.no_grad():
            if self.sum is None:
                self.sum = torch.clone(output)
            else:
                self.sum += output
            self.count += 1

        return output


_distribution_hooks = []


def log_distribution_install_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Conv2d)
            or isinstance(module, torch.nn.ReLU)
            or isinstance(module, torch.nn.BatchNorm2d)
        ):
            hook = LogActDistributionHook(name)
            handle = module.register_forward_hook(hook)
            hooks.append((handle, hook))

    return hooks


def log_distribution_plot(hooks):
    for handle, hook in hooks:
        handle.remove()

        data = hook.sum / hook.count
        data = data.detach().cpu().numpy()

        plot = sns.displot(data.flatten(), kind="hist", bins=20)
        plot.savefig(f"act_distribution_{hook.name}.png")
        plt.clf()