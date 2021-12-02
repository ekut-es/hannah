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

        if isinstance(
            self.target,
            (
                ElasticConv1d,
                ElasticConvBn1d,
                ElasticConvBnReLu1d,
                ElasticQuantConv1d,
                ElasticQuantConvBn1d,
                ElasticQuantConvBnReLu1d,
                ElasticWidthLinear,
            ),
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
        if isinstance(module, ElasticWidthLinear):
            if is_target:
                # target module -> set module input filter
                if len(module.in_channel_filter) != len(self.channel_pass_filter):
                    logging.error(
                        f"Elastic channel helper filter length {len(self.channel_pass_filter)} does not match filter length {len(module.in_channel_filter)} of {type(module)}! "
                    )
                    return
                module.in_channel_filter = self.channel_pass_filter
            else:
                # source module -> set module output filter
                if len(module.out_channel_filter) != len(self.channel_pass_filter):
                    logging.error(
                        f"Elastic channel helper filter length {len(self.channel_pass_filter)} does not match filter length {len(module.out_channel_filter)} of {type(module)}! "
                    )
                    return
                module.out_channel_filter = self.channel_pass_filter
        elif isinstance(
            module,
            (
                ElasticConv1d,
                ElasticConvBn1d,
                ElasticConvBnReLu1d,
                ElasticQuantConv1d,
                ElasticQuantConvBn1d,
                ElasticQuantConvBnReLu1d,
            ),
        ):
            if is_target:
                # target module -> set module input filter
                if len(module.in_channel_filter) != len(self.channel_pass_filter):
                    logging.error(
                        f"Elastic channel helper filter length {len(self.channel_pass_filter)} does not match filter length {len(module.in_channel_filter)} of {type(module)}! "
                    )
                    return
                module.in_channel_filter = self.channel_pass_filter
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
            (
                ElasticConv1d,
                ElasticConvBn1d,
                ElasticConvBnReLu1d,
                ElasticQuantConv1d,
                ElasticQuantConvBn1d,
                ElasticQuantConvBnReLu1d,
                ElasticWidthLinear,
            ),
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

    def forward(self, x):
        if isinstance(x, SequenceDiscovery):
            return x.discover(self)
        return x


class SequenceDiscovery:
    def __init__(self, is_accumulating_sources: bool = True):
        super().__init__()
        # mode is either: accumulating sources for an upcoming elastic width connection
        # or collecting targets of an elastic width connection
        self.is_accumulating_sources = is_accumulating_sources
        # when accumulating sources, they are added to the moduleList
        self.accumulated_sources = nn.ModuleList([])
        # when seeking targets, the relevant helper module is stored here.
        self.helper: ElasticChannelHelper = None

    # process a module, return the next sequence discovery for it to pass forward
    # simply pass a reference to the new module, this includes helper modules.
    def discover(self, new_module, force_secondary_target: bool = False):
        # primary targets will re-set collected modules, and change mode to source accumulation
        if (
            ElasticChannelHelper.is_primary_target(new_module)
            and not force_secondary_target
        ):
            if not self.is_accumulating_sources:
                # if in target finding mode, pass the newly discovered module as a target to the helper
                if self.helper is None:
                    logging.error(
                        "SequenceDiscovery is in target mode, but has no helper module!"
                    )
                else:
                    self.helper.discover_target(new_module)
            # with primary modules, a new source sequence is always started:
            # after a primary module, we are no longer seeking a target for a previous helper (if applicable)
            # and the channel width of previous sources is irrelevant to subsequent helpers
            new_discovery = SequenceDiscovery(is_accumulating_sources=True)
            new_discovery.accumulated_sources.append(new_module)
            return new_discovery

        elif ElasticChannelHelper.is_primary_target(new_module):
            # if the module is technically a primary target, but force_secondary_target is specified
            # (this will happen on skip connection convs of a residual block, for example)
            # process it as a secondary target in target discovery mode.
            if not self.is_accumulating_sources:
                # if accumulating targets with force_secondary_target, add module as secondary target
                if self.helper is None:
                    logging.error(
                        "SequenceDiscovery is in target mode, but has no helper module!"
                    )
                else:
                    self.helper.add_secondary_target_item(new_module)
            # since the module is still a primary module, switch to a new source discovery (regardless of previous mode)
            new_discovery = SequenceDiscovery(is_accumulating_sources=True)
            new_discovery.accumulated_sources.append(new_module)
            return new_discovery

        elif isinstance(new_module, ElasticChannelHelper):
            if self.is_accumulating_sources:
                # if in source accumulation mode, pass accumulated sources to the helper.
                print(
                    f"Passing to elastic helper: {len(self.accumulated_sources)} modules as sources"
                )
                new_module.add_sources(self.accumulated_sources)
            else:
                logging.error(
                    "Elastic width helper target accumulation reached another helper module!"
                )
            # after discovery reaches a helper module, the mode is switched to target discovery.
            new_discovery = SequenceDiscovery(is_accumulating_sources=False)
            new_discovery.helper = new_module
            return new_discovery

        else:
            # for secondary modules, simply add them to the known sources, or pass them as secondary targets
            if self.is_accumulating_sources:
                self.accumulated_sources.append(new_module)
            else:
                if self.helper is None:
                    logging.error(
                        "SequenceDiscovery is in target mode, but has no helper module!"
                    )
                else:
                    self.helper.add_secondary_target_item(new_module)
            return self

    # process when two discovery modules would be added together (with parallel modules, e.g. skip connections)
    def merge_sequence_discovery(self, second_discovery):
        if self.is_accumulating_sources and second_discovery.is_accumulating_sources:
            # if both are in source discovery mode, simply pass the sum of sources on.
            for new_source in second_discovery.accumulated_sources:
                self.accumulated_sources.append(new_source)
            return self
        elif (
            not self.is_accumulating_sources
            and second_discovery.is_accumulating_sources
        ):
            # if this module is in find target mode, and is being merged with sources, add sources to this helper
            # then, pass on the target discovery unmodified.
            if self.helper is None:
                logging.error(
                    "SequenceDiscovery is in target mode, but has no helper module!"
                )
            else:
                self.helper.add_sources(second_discovery.accumulated_sources)
            return self
        elif (
            self.is_accumulating_sources
            and not second_discovery.is_accumulating_sources
        ):
            # if the other discovery is in target mode, and this discovery is not,
            # simply re-call the merge on the other module to avoid code duplication
            return second_discovery.merge_sequence_discovery(self)
        else:
            # if both discoveries are in target mode, the next target(s) could theoretically be passed to both helpers
            # as they would calculate the same norms, they should both pass forward the same filter
            # optimally, both helpers should probably be merged
            raise NotImplementedError(
                "merging target discoveries of separate helpers from parallel blocks is NYI."
            )
            # skip connections on residual blocks don't need to contain their own helper module.
            # for parallel blocks, simply having the the last connection be elastic on only one of the blocks is sufficient:
            # the 'sources' of the other block will be added to the helper of the 'primary' block during the merge

    # return a second, new discovery with the same references, for splitting off a connection
    # this may be done at the input of a resblock for example.
    def split(self):
        new_discovery = SequenceDiscovery(
            is_accumulating_sources=self.is_accumulating_sources
        )
        # yield shallow copy of sources list
        new_discovery.accumulated_sources = list(self.accumulated_sources)
        new_discovery.helper = self.helper
        return new_discovery


# imports are located at the bottom to circumvent circular dependency import issues
from .elasticwidthmodules import ElasticWidthBatchnorm1d, ElasticWidthLinear
from .elastickernelconv import (
    ElasticConv1d,
    ElasticConvBn1d,
    ElasticConvBnReLu1d,
    ElasticQuantConv1d,
    ElasticQuantConvBn1d,
    ElasticQuantConvBnReLu1d,
)
