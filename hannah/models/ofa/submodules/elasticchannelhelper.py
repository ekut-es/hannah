import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn

from ..utilities import flatten_module_list


# helper module, deployed in an elastic width connection
# can zero out input channels to train elastic channels without weight modification
# must know the module it passes inputs to to compute channel priorities
# must know source modules to remove output channels in extraction
# This can be previous linears/convs, the skip layer of a previous residual block,
# and batchnorms placed in-between
class ElasticChannelHelper(nn.Module):
    def __init__(
        self,
        channel_counts: List[int],
        # sources: nn.ModuleList,
        # target: nn.Module,
        # additional_targets: nn.ModuleList = nn.ModuleList([]),
    ):
        super().__init__()
        # sort channel counts: largest -> smallest
        self.channel_counts = channel_counts
        self.channel_counts.sort(reverse=True)
        self.sources = nn.ModuleList([])
        self.target = None
        # additional target modules will not be used to compute channel priorities,
        # but will have to be known for reducing input channels
        # this may contain additional exits, the skip layer of a following residual block
        self.additional_targets = nn.ModuleList([])
        # initialize filter for channel reduction in training, channel priority list
        self.channel_pass_filter: List[int] = []
        # the first channel index in this list is least important, the last channel index ist most important
        self.channels_by_priority: List[int] = []
        # initially, all channels are used.
        self.max_channels: int = self.channel_counts[0]
        self.channel_step: int = 0
        self.current_channels: int = self.channel_counts[self.channel_step]

        # initialize the filter and the channel priority list
        for i in range(self.max_channels):
            self.channel_pass_filter.append(True)
            # to init with technically valid values, simply set starting priority based on index
            self.channels_by_priority.append(i)

    # compute channel priorities based on the l1 norm of the weights of whichever
    # target module follows this elastic channel section
    def compute_channel_priorities(self):
        target = self.target
        channel_norms = []

        # this will also include the elastic kernel convolutions
        # for elastic kernel convolutions, the priorities will then also be
        # computed on the base module (full kernel)
        if isinstance(target, nn.Conv1d):
            weights = target.weight.data
            norms_per_kernel_index = torch.linalg.norm(weights, ord=1, dim=0)
            channel_norms = torch.linalg.norm(norms_per_kernel_index, ord=1, dim=1)
        # the channel priorities for linears need to also be computable:
        # especially for the exit connections, a linear may follow after an elastic width
        elif isinstance(target, nn.Linear):
            weights = target.weight.data
            channel_norms = torch.linalg.norm(weights, ord=1, dim=0)
        else:
            # the channel priorities will keep their previous / default value in
            # this case. Reduction will probably occur by channel order
            logging.warning(
                f"Unable to compute channel priorities! Unsupported target module after elastic channels: {type(target)}"
            )

        # contains the indices of the channels, sorted from channel with smallest
        # norm to channel with largest norm
        # the least important channel index is at the beginning of the list,
        # the most important channel index is at the end
        self.channels_by_priority = np.argsort(channel_norms)

    # set the channel filter list based on the channel priorities and the current channel count
    def set_channel_filter(self):
        # get the amount of channels to be removed from the max and current channel counts
        channel_reduction_amount: int = self.max_channels - self.current_channels
        # start with an empty filter, where every channel passes through, then remove channels by priority
        self.channel_pass_filter = [True] * len(self.channel_pass_filter)

        # filter the least important n channels, specified by the reduction amount
        for i in range(channel_reduction_amount):
            # priority list of channels contains channel indices from least important to most important
            # the first n channel indices specified in this list will be filtered out
            filtered_channel_index = self.channels_by_priority[i]
            self.channel_pass_filter[filtered_channel_index] = False

        if isinstance(
            self.target,
            elastic_forward_type,
        ):
            self.apply_filter_to_module(self.target, is_target=True)
        else:
            logging.warn(
                f"Elastic channel helper has no defined behavior for primary target type: {type(self.target)}"
            )
        for item in self.additional_targets:
            self.apply_filter_to_module(item, is_target=True)
        for item in self.sources:
            self.apply_filter_to_module(item, is_target=False)

    # if is_target is set to true, the module is a target module (filter its input).
    # false -> source module -> filter its output
    def apply_filter_to_module(self, module, is_target: bool):
        if isinstance(
            module,
            elastic_forward_type,
        ):
            if is_target:
                # target module -> set module input filter
                if len(module.in_channel_filter) != len(self.channel_pass_filter):
                    logging.error(
                        f"Elastic channel helper filter length {len(self.channel_pass_filter)} does not match filter length {len(module.in_channel_filter)} of {type(module)}! "
                    )
                    return
                module.set_in_channel_filter(self.channel_pass_filter)
            else:
                # source module -> set module output filter
                if len(module.out_channel_filter) != len(self.channel_pass_filter):
                    logging.error(
                        f"Elastic channel helper filter length {len(self.channel_pass_filter)} does not match filter length {len(module.out_channel_filter)} of {type(module)}! "
                    )
                    return
                module.set_out_channel_filter(self.channel_pass_filter)

        elif isinstance(module, ElasticWidthBatchnorm1d):
            # this is normal for residual blocks with a norm after applying residual output to blocks output
            # if is_target:
            #    logging.warn("Batchnorm found in Elastic channel helper targets, it should usually be located in-front of the helper module.")
            if len(module.channel_filter) != len(self.channel_pass_filter):
                logging.error(
                    f"Elastic channel helper filter length {len(self.channel_pass_filter)} does not match filter length {len(module.channel_filter)} of {type(module)}!"
                )
                return
            module.channel_filter = self.channel_pass_filter
        else:
            logging.error(
                f"Elastic channel helper could not apply filter to module of unknown type: {type(module)}"
            )

    # step down channel count by one channel step
    def step_down_channels(self):
        if self.channel_step + 1 in range(len(self.channel_counts)):
            # if there is still channel steps available, step forward by one. Set new active channel count.
            self.channel_step += 1
            self.current_channels = self.channel_counts[self.channel_step]
            # after stepping down channels by one, set new channel filter.
            self.set_channel_filter()
            return True
        else:
            # if the last channel step is already reached, no additional step-down operation can be performed
            return False

    def set_channel_step(self, step: int):
        if step not in range(len(self.channel_counts)):
            logging.warn(
                f"Elastic channel helper step target {step} out of range for length {len(self.channel_counts)}. Defaulting to 0."
            )
            step = 0
        if step == self.channel_step:
            # only re-apply filters if there is actually going to be a change.
            return
        self.channel_step = step
        self.current_channels = self.channel_counts[self.channel_step]
        self.set_channel_filter()

    def reset_channel_step(self):
        self.channel_step = 0

    # set the primary target from an input module. For iterable inputs, extract additional secondary targets
    def set_primary_target(self, target: nn.Module):
        if hasattr(target, "__iter__"):
            # first, flatten the target, if it is iterable
            target = flatten_module_list(target)
            # the primary target is the first linear/conv in the sequence
            for item in target:
                if self.is_valid_primary_target(item):
                    self.target = item
                    # if the primary target was found in the sequence, any trailing
                    # modules must be ignored, as they are unaffected.
                    break
                else:
                    # if the module item is not a primary target, process it as a secondary target.
                    self.add_secondary_targets(item)
                    # this will check for other, invalid ElasticChannelHelper
                    # modules in targets and throw an error
        else:
            # if the input is not iterable, and is just a simple module, it is the target
            if not self.is_valid_primary_target(target):
                # if the standalone module is not actually a valid primary target, something went wrong!
                logging.warn(
                    f"ElasticChannelHelper target module is an invalid module: '{type(target)}'. Target reset to None."
                )
                self.target = None
            # if the input is valid as a target module, set it as the target
            self.target = target

    # check if a module is valid as a primary target (to compute channel priorities from)
    def is_valid_primary_target(self, module: nn.Module) -> bool:
        # legacy function
        return ElasticChannelHelper.is_primary_target(module)

    # check if a module is valid as a primary target (to compute channel priorities from)
    def is_primary_target(module: nn.Module) -> bool:
        return isinstance(
            module,
            elastic_forward_type,
        )

    # add additional target(s) which must also have their inputs adjusted when
    # stepping down channels
    def add_secondary_targets(self, target: nn.Module):
        if hasattr(target, "__iter__"):
            # if the input target is iterable, check every item
            target_flat = flatten_module_list(target)
            for item in target_flat:
                if isinstance(item, ElasticChannelHelper):
                    logging.error(
                        "ElasticChannelHelper target accumulation reached another ElasticChannelHelper, with no primary target in-between!"
                    )
                self.add_secondary_target_item(item)
                if self.is_valid_primary_target(item):
                    # if a valid primary target is found reached, the modules
                    # trailing it must not be affected by width changes
                    # only modules before a trailing linear/conv will be affected
                    break
        else:
            self.add_secondary_target_item(target)

    # TODO: logic for adding secondary items to target/source is pretty much a copy - could be cleaned up
    # check a module, add it as a secondary target if its weights would need modification when channel width changes
    def add_secondary_target_item(self, target: nn.Module):
        if self.is_valid_primary_target(target):
            self.additional_targets.append(target)
        elif isinstance(target, ElasticWidthBatchnorm1d):
            # trailing batchnorms between the channel helper and the next 'real'
            # module will also need to have their channels adjusted
            # this is normal for residual blocks with a norm after applying residual output to blocks output
            # logging.warn(
            #     "found loose BatchNorm1d module trailing an elastic channel helper. These should be located in-front of the helper"
            # )
            self.additional_targets.append(target)
        elif isinstance(target, nn.ReLU):
            logging.warn(
                "found loose ReLu module trailing an elastic channel helper. These should be located in-front of the helper"
            )
        else:
            logging.warn(
                f"module with undefined behavior found in ElasticChannelHelper targets: '{type(target)}'. Ignoring."
            )

    # add additional source(s) which must have their outputs adjusted if the channel width changes
    def add_sources(self, source: nn.Module):
        if hasattr(source, "__iter__"):
            # if the input source is iterable, check every item
            source_flat = flatten_module_list(source)
            for item in source_flat:
                # ascend the list of sources from the back
                if isinstance(item, ElasticChannelHelper):
                    logging.exception(
                        "ElasticChannelHelper source accumulation found another ElasticChannelHelper!"
                    )
                self.add_source_item(item)
        else:
            self.add_source_item(self, source)

    # add additional source(s) which must have their outputs adjusted if the channel width changes
    def add_targets(self, target: nn.Module):
        if hasattr(target, "__iter__"):
            # if the input source is iterable, check every item
            target_flat = flatten_module_list(target)

            for item in target_flat:
                # ascend the list of sources from the back
                if isinstance(item, ElasticChannelHelper):
                    logging.exception(
                        "ElasticChannelHelper source accumulation found another ElasticChannelHelper!"
                    )
                self.discover_target(item)
        else:
            self.discover_target(target)

    # check a module, add it as a source if its weights would need modification when channel width changes
    def add_source_item(self, source: nn.Module):
        if self.is_valid_primary_target(source):
            # modules which are valid primary targets (Convs, Linears) are also valid sources
            self.sources.append(source)
        elif isinstance(source, ElasticWidthBatchnorm1d):
            # batchnorms before the channel helper will need to be adjusted if channels are removed
            self.sources.append(source)
        elif isinstance(source, nn.ReLU):
            # ReLu preceding the channel helper can be ignored. It does not need adjustment.
            pass
        else:
            logging.warn(
                f"module with undefined behavior found in ElasticChannelHelper sources: '{type(source)}'. Ignoring."
            )

    def discover_target(self, new_target: nn.Module):
        # if no target is set yet, take this module as the primary target
        if self.is_valid_primary_target(new_target) and self.target is None:
            self.set_primary_target(new_target)
        else:
            self.add_secondary_targets(new_target)

    def get_available_width_steps(self):
        return len(self.channel_counts)


from ..type_utils import elastic_forward_type

# imports are located at the bottom to circumvent circular dependency import issues
from .elasticBatchnorm import ElasticWidthBatchnorm1d
