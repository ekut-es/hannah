from typing import List
import torch.nn as nn
import numpy as np
import logging
import torch
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
        sources: nn.ModuleList,
        target: nn.Module,
        additional_targets: nn.ModuleList = nn.ModuleList([]),
    ):
        super().__init__()
        # sort channel counts: largest -> smallest
        self.channel_counts = channel_counts
        self.channel_counts.sort(reverse=True)
        self.sources = sources
        self.target = target
        # additional target modules will not be used to compute channel priorities,
        # but will have to be known for reducing input channels
        # this may contain additional exits, the skip layer of a following residual block
        self.additional_targets = additional_targets
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
        # the channel priorities for lienars need to also be computable:
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
        self.input_channels_by_priority = np.argsort(channel_norms)

    # set the channel filter list based on the channel priorities and the current channel count
    def set_channel_filter(self):
        # get the amount of channels to be removed from the max and current channel counts
        channel_reduction_amount: int = self.max_channels - self.current_channels
        # start with an empty filter, where every channel passes through, then remove channels by priority
        for i in range(len(self.channel_pass_filter)):
            self.channel_pass_filter[i] = True

        # filter the least important n channels, specified by the reduction amount
        for i in range(channel_reduction_amount):
            # priority list of channels contains channel indices from least important to most important
            # the first n channel indices specified in this list will be filtered out
            filtered_channel_index = self.channels_by_priority[i]
            self.channel_pass_filter[filtered_channel_index] = False

        # store the channel filter on every affected module, to make extraction more straightforward later.
        # then, in the extraction step, we no longer need to know about
        # the relation of modules, each module knows which channels must be removed.
        setattr(self.target, "elastic_width_filter_input", self.channel_pass_filter)
        # any modules which also get inputs from this elastic width connection
        # should also know the filter for extraction
        # this may contain skip layers in following residual blocks or additional exit layers
        for additional_target in self.additional_targets:
            setattr(
                additional_target,
                "elastic_width_filter_input",
                self.channel_pass_filter,
            )
        for source in self.sources:
            setattr(source, "elastic_width_filter_output", self.channel_pass_filter)

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
        return isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear)

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
<<<<<<< HEAD
        elif isinstance(target, nn.BatchNorm1d):
            # trailing batchnorms between the channel helper and the next 'real' module will also need to have their channels adjusted
            logging.info("found loose BatchNorm1d module trailing an elastic channel helper. These are usually located in-front of the helper")
=======
        if isinstance(target, nn.BatchNorm1d):
            # trailing batchnorms between the channel helper and the next 'real'
            # module will also need to have their channels adjusted
            logging.info(
                "found loose BatchNorm1d module trailing an elastic channel helper. These are usually located in-front of the helper"
            )
>>>>>>> aa76597b243f707659c082f0407c5acd67cda03f
            self.additional_targets.append(target)
        elif isinstance(target, nn.ReLU):
            logging.info(
                "found loose ReLu module trailing an elastic channel helper. These are usually located in-front of the helper"
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
            for item in source_flat[::-1]:
                # ascend the list of sources from the back
                if isinstance(item, ElasticChannelHelper):
                    logging.error(
                        "ElasticChannelHelper source accumulation reached another ElasticChannelHelper, with no primary target in-between!"
                    )
                self.add_source_item(self, item)
                if self.is_valid_primary_target(item):
                    # if a valid primary target is found in the sources, the
                    # modules above it must not be affected by width changes
                    # only modules after a previous linear/conv will be affected
                    break
        else:
            self.add_source_item(self, source)

    # check a module, add it as a source if its weights would need modification when channel width changes
    def add_source_item(self, source: nn.Module):
        if self.is_valid_primary_target(source):
            # modules which are valid primary targets (Convs, Linears) are also valid sources
            self.sources.append(source)
        if isinstance(source, nn.BatchNorm1d):
            # batchnorms before the channel helper will need to be adjusted if channels are removed
            self.sources.append(source)
        elif isinstance(source, nn.ReLU):
            # ReLu preceding the channel helper can be ignored. It does not need adjustment.
            pass
        else:
            logging.warn(
                f"module with undefined behavior found in ElasticChannelHelper sources: '{type(source)}'. Ignoring."
            )

    # in forward, zero out filtered channels
    def forward(self, x):
        input = x
        null_input = torch.zeros_like(input)
        # work on a copy of the input to avoid in-place operations on the input tensor
        input_copy = torch.clone(input)
        zeroed = 0
        for input_index in range(len(input)):
            # for every input index
            for channel_index in range(len(input[input_index])):
                # for every channel index within that input
                # print(self.channel_pass_filter)
                # print(channel_index)
                if not self.channel_pass_filter[channel_index]:
                    zeroed += 1
                    # if this channel index is supposed to be filtered, copy
                    # over zeroes from the equivalent null input
                    input_copy[input_index][channel_index] = null_input[input_index][
                        channel_index
                    ]
        # sanity check
        removed_channels_count = self.max_channels - self.current_channels
        if (zeroed / len(input) != removed_channels_count) or (
            self.channel_step > 0 and zeroed == 0
        ):
            logging.warn(
                f"ElasticChannelHelper zeroed channel count {zeroed/len(input)} does not match expected {removed_channels_count}"
            )
        return input_copy

    def __call__(self, x):
        return self.forward(x)
