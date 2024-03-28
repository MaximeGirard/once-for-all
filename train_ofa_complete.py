import argparse
import numpy as np
import os
import random

import torch.nn as nn

import horovod.torch as hvd
import torch

from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicSeparableConv2d,
)
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from ofa.imagenet_classification.run_manager import DistributedImageNetRunConfig
from ofa.imagenet_classification.networks import MobileNetV3Large
from ofa.imagenet_classification.run_manager.distributed_run_manager import (
    DistributedRunManager,
)
from ofa.utils import download_url, MyRandomResizedCrop
from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import (
    load_models,
)

from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import (
    validate,
    train,
)

from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import (
    train_elastic_depth,
    train_elastic_expand,
)

args = {}

args["path"] = "exp/normal2kernel"
args["dynamic_batch_size"] = 1
args["n_epochs"] = 1
args["base_lr"] = 3e-2
args["warmup_epochs"] = 0
args["warmup_lr"] = -1
args["ks_list"] = [3, 5, 7]
args["expand_list"] = [3, 4, 6]
args["depth_list"] = [2, 3, 4]

args["label_mapping"] = label_mapping = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
args["remap_imagenette"] = True

args["manual_seed"] = 0

args["lr_schedule_type"] = "cosine"

args["base_batch_size"] = 64
args["valid_size"] = 100

args["opt_type"] = "sgd"
args["momentum"] = 0.9
args["no_nesterov"] = False
args["weight_decay"] = 3e-5
args["label_smoothing"] = 0.1
args["no_decay_keys"] = "bn#bias"
args["fp16_allreduce"] = False

args["model_init"] = "he_fout"
args["validation_frequency"] = 1
args["print_frequency"] = 10

args["n_worker"] = 8
args["resize_scale"] = 0.08
args["distort_color"] = "tf"
args["image_size"] = [128, 160, 192, 224]
args["continuous_size"] = True
args["not_sync_distributed_image_size"] = False

args["bn_momentum"] = 0.1
args["bn_eps"] = 1e-5
args["dropout"] = 0.1
args["base_stage_width"] = "proxyless"

args["width_mult_list"] = 1.0
args["dy_conv_scaling_mode"] = 1
args["independent_distributed_sampling"] = False

args["kd_ratio"] = 1.0
args["kd_type"] = "ce"

os.makedirs(args["path"], exist_ok=True)

# Initialize Horovod
hvd.init()
# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())

args["teacher_path"] = download_url(
    "https://raw.githubusercontent.com/han-cai/files/master/ofa/ofa_checkpoints/ofa_D4_E6_K7",
    model_dir=".torch/ofa_checkpoints/%d" % hvd.rank(),
)

num_gpus = hvd.size()

torch.manual_seed(args["manual_seed"])
torch.cuda.manual_seed_all(args["manual_seed"])
np.random.seed(args["manual_seed"])
random.seed(args["manual_seed"])


MyRandomResizedCrop.CONTINUOUS = args["continuous_size"]
MyRandomResizedCrop.SYNC_DISTRIBUTED = not args["not_sync_distributed_image_size"]

# build run config from args
args["lr_schedule_param"] = None
args["opt_param"] = {
    "momentum": args["momentum"],
    "nesterov": not args["no_nesterov"],
}
args["init_lr"] = args["base_lr"] * num_gpus  # linearly rescale the learning rate
if args["warmup_lr"] < 0:
    args["warmup_lr"] = args["base_lr"]
args["train_batch_size"] = args["base_batch_size"]
args["test_batch_size"] = args["base_batch_size"] * 4
run_config = DistributedImageNetRunConfig(
    **args, num_replicas=num_gpus, rank=hvd.rank()
)

# print run config information
if hvd.rank() == 0:
    print("Run config:")
    for k, v in run_config.config.items():
        print("\t%s: %s" % (k, v))

if args["dy_conv_scaling_mode"] == -1:
    args["dy_conv_scaling_mode"] = None
DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args["dy_conv_scaling_mode"]

net = OFAMobileNetV3(
    n_classes=run_config.data_provider.n_classes,
    bn_param=(args["bn_momentum"], args["bn_eps"]),
    dropout_rate=args["dropout"],
    base_stage_width=args["base_stage_width"],
    width_mult=args["width_mult_list"],
    ks_list=args["ks_list"],
    expand_ratio_list=args["expand_list"],
    depth_list=args["depth_list"],
)
# teacher model
# Used for soft targets (knowledge distillation)
if args["kd_ratio"] > 0:
    args["teacher_model"] = MobileNetV3Large(
        n_classes=run_config.data_provider.n_classes,
        bn_param=(args["bn_momentum"], args["bn_eps"]),
        dropout_rate=0,
        width_mult=1.0,
        ks=7,
        expand_ratio=6,
        depth_param=4,
    )
    args["teacher_model"].cuda()

""" Distributed RunManager """
# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args["fp16_allreduce"] else hvd.Compression.none
run_manager = DistributedRunManager(
    args["path"],
    net,
    run_config,
    compression,
    backward_steps=args["dynamic_batch_size"],
    is_root=(hvd.rank() == 0),
)
run_manager.save_config()
# hvd broadcast
run_manager.broadcast()

# load teacher net weights
if args["kd_ratio"] > 0:
    load_models(
        run_manager,
        args["teacher_model"],
        model_path=args["teacher_path"],
    )

# training

tasks = ["kernel", "depth", "depth", "expand", "expand"]
# Define the phases for the depth and expand tasks
depth_phases = [1, 2]
expand_phases = [1, 2]


def get_validation_func_dict():
    validate_func_dict = {
        "image_size_list": (
            {224} if isinstance(args["image_size"], int) else sorted({160, 224})
        ),
        "ks_list": (
            sorted(args["ks_list"])
            if task == "kernel"
            else sorted({min(args["ks_list"]), max(args["ks_list"])})
        ),
        "expand_ratio_list": sorted(
            {min(args["expand_list"]), max(args["expand_list"])}
        ),
        "depth_list": sorted({min(args["depth_list"]), max(args["depth_list"])}),
    }
    print("Validation function parameters:", validate_func_dict)
    return validate_func_dict


def set_net_constraint():
    dynamic_net = run_manager.net
    dynamic_net.set_constraint(args["ks_list"], constraint_type="kernel_size")
    dynamic_net.set_constraint(args["expand_list"], constraint_type="expand_ratio")
    dynamic_net.set_constraint(args["depth_list"], constraint_type="depth")


# Iterate over the tasks list
for task in tasks:
    # Create a dictionary to store the validation function parameters
    # Execute the corresponding code block based on the task
    if task == "kernel":
        # Define parameters for the kernel task
        args["path"] = "exp/normal2kernel"
        args["dynamic_batch_size"] = 1
        args["n_epochs"] = 1
        args["base_lr"] = 3e-2
        args["warmup_epochs"] = 0
        args["warmup_lr"] = -1
        args["ks_list"] = [3, 5, 7]
        args["expand_list"] = [6]
        args["depth_list"] = [4]

        validate_func_dict = get_validation_func_dict()

        # The original (non elastic) net is trained for 150 epochs
        # according to https://openreview.net/forum?id=HylxE1HKwS&noteId=ByxzZ83IYH
        args["ofa_checkpoint_path"] = download_url(
            "https://raw.githubusercontent.com/han-cai/files/master/ofa/ofa_checkpoints/ofa_D4_E6_K7",
            model_dir=".torch/ofa_checkpoints/%d" % hvd.rank(),
        )
        load_models(
            run_manager,
            run_manager.net,
            args["ofa_checkpoint_path"],
        )
        # run_manager.write_log(
        #     "%.3f\t%.3f\t%.3f\t%s"
        #     % validate(run_manager, is_test=True, **validate_func_dict),
        #     "valid",
        # )

        set_net_constraint()
        train(
            run_manager,
            args,
            lambda _run_manager, epoch, is_test: validate(
                _run_manager, epoch, is_test, **validate_func_dict
            ),
        )
    elif task == "depth":
        # Iterate over the depth phases
        for phase in depth_phases:
            args["phase"] = phase
            args["path"] = f"exp/kernel2kernel_depth/phase{phase}"
            args["dynamic_batch_size"] = 2
            if phase == 1:
                args["n_epochs"] = 1
                args["base_lr"] = 2.5e-3
                args["warmup_epochs"] = 0
                args["warmup_lr"] = -1
                args["ks_list"] = [3, 5, 7]
                args["expand_list"] = [6]
                args["depth_list"] = [3, 4]

                args["ofa_checkpoint_path"] = download_url(
                    "https://raw.githubusercontent.com/han-cai/files/master/ofa/ofa_checkpoints/ofa_D4_E6_K357",
                    model_dir=".torch/ofa_checkpoints/%d" % hvd.rank(),
                )
            else:
                args["n_epochs"] = 1
                args["base_lr"] = 7.5e-3
                args["warmup_epochs"] = 1
                args["warmup_lr"] = -1
                args["ks_list"] = [3, 5, 7]
                args["expand_list"] = [6]
                args["depth_list"] = [2, 3, 4]

                args["ofa_checkpoint_path"] = download_url(
                    "https://raw.githubusercontent.com/han-cai/files/master/ofa/ofa_checkpoints/ofa_D34_E6_K357",
                    model_dir=".torch/ofa_checkpoints/%d" % hvd.rank(),
                )

            validate_func_dict = get_validation_func_dict()
            # Train the depth model
            dynamic_net = run_manager.net
            if isinstance(dynamic_net, nn.DataParallel):
                dynamic_net = dynamic_net.module

            depth_stage_list = args["depth_list"].copy()
            # We begin to train with the largest values
            depth_stage_list.sort(reverse=True)
            n_stages = len(depth_stage_list) - 1
            current_stage = n_stages - 1

            print("- Train with elastic depth using depth_list:", depth_stage_list)
            print("- Current stage: %d/%d" % (current_stage, n_stages - 1))

            # load pretrained models
            validate_func_dict["depth_list"] = sorted(dynamic_net.depth_list)

            # load_models(
            #     run_manager, dynamic_net, model_path=args["ofa_checkpoint_path"]
            # )
            # validate after loading weights
            # run_manager.write_log(
            #     "%.3f\t%.3f\t%.3f\t%s"
            #     % validate(run_manager, is_test=True, **validate_func_dict),
            #     "valid",
            # )

            run_manager.write_log(
                "-" * 30
                + "Supporting Elastic Depth: %s -> %s"
                % (
                    depth_stage_list[: current_stage + 1],
                    depth_stage_list[: current_stage + 2],
                )
                + "-" * 30,
                "valid",
            )

            # add depth list constraints
            if (
                len(set(dynamic_net.ks_list)) == 1
                and len(set(dynamic_net.expand_ratio_list)) == 1
            ):
                validate_func_dict["depth_list"] = depth_stage_list
            else:
                validate_func_dict["depth_list"] = sorted(
                    {min(depth_stage_list), max(depth_stage_list)}
                )

            set_net_constraint()
            # train
            train(
                run_manager,
                args,
                lambda _run_manager, epoch, is_test: validate(
                    _run_manager, epoch, is_test, **validate_func_dict
                ),
            )
    elif task == "expand":
        # Iterate over the expand phases
        for phase in expand_phases:
            args["phase"] = phase
            args["path"] = f"exp/kernel_depth2kernel_depth_width/phase{phase}"
            args["dynamic_batch_size"] = 4
            if phase == 1:
                args["n_epochs"] = 1
                args["base_lr"] = 2.5e-3
                args["warmup_epochs"] = 0
                args["warmup_lr"] = -1
                args["ks_list"] = [3, 5, 7]
                args["expand_list"] = [4, 6]
                args["depth_list"] = [2, 3, 4]

                args["ofa_checkpoint_path"] = download_url(
                    "https://raw.githubusercontent.com/han-cai/files/master/ofa/ofa_checkpoints/ofa_D234_E6_K357",
                    model_dir=".torch/ofa_checkpoints/%d" % hvd.rank(),
                )
            else:
                args["n_epochs"] = 1
                args["base_lr"] = 7.5e-3
                args["warmup_epochs"] = 5
                args["warmup_lr"] = -1
                args["ks_list"] = [3, 5, 7]
                args["expand_list"] = [3, 4, 6]
                args["depth_list"] = [2, 3, 4]

                args["ofa_checkpoint_path"] = download_url(
                    "https://raw.githubusercontent.com/han-cai/files/master/ofa/ofa_checkpoints/ofa_D234_E46_K357",
                    model_dir=".torch/ofa_checkpoints/%d" % hvd.rank(),
                )

            validate_func_dict = get_validation_func_dict()
            # Train the expand model

            dynamic_net = run_manager.net
            if isinstance(dynamic_net, nn.DataParallel):
                dynamic_net = dynamic_net.module

            expand_stage_list = args["expand_list"].copy()
            expand_stage_list.sort(reverse=True)
            n_stages = len(expand_stage_list) - 1
            current_stage = n_stages - 1

            print(
                "- Train with elastic expand using expand_ratio_list:",
                expand_stage_list,
            )
            print("- Current stage: %d/%d" % (current_stage, n_stages - 1))

            # load pretrained models
            validate_func_dict["expand_ratio_list"] = sorted(
                dynamic_net.expand_ratio_list
            )

            # load_models(
            #     run_manager, dynamic_net, model_path=args["ofa_checkpoint_path"]
            # )
            dynamic_net.re_organize_middle_weights(expand_ratio_stage=current_stage)
            # run_manager.write_log(
            #     "%.3f\t%.3f\t%.3f\t%s"
            #     % validate(run_manager, is_test=True, **validate_func_dict),
            #     "valid",
            # )

            print(
                "Supporting Elastic Expand Ratio: %s -> %s"
                % (
                    expand_stage_list[: current_stage + 1],
                    expand_stage_list[: current_stage + 2],
                )
            )
            if (
                len(set(dynamic_net.ks_list)) == 1
                and len(set(dynamic_net.depth_list)) == 1
            ):
                validate_func_dict["expand_ratio_list"] = expand_stage_list
            else:
                validate_func_dict["expand_ratio_list"] = sorted(
                    {min(expand_stage_list), max(expand_stage_list)}
                )

            set_net_constraint()
            # train
            train(
                run_manager,
                args,
                lambda _run_manager, epoch, is_test: validate(
                    _run_manager, epoch, is_test, **validate_func_dict
                ),
            )
