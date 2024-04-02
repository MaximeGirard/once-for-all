import os
import torch
import random
from tqdm import tqdm
from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from ofa.utils import AverageMeter
from ofa.nas.accuracy_predictor import MobileNetArchEncoder
import pickle

# Define arguments
args = {
    "dataset_path": "imagenette2/",
    "device": "cuda",
    "ks_list": [3, 5, 7],
    "expand_list": [3, 4, 6],
    "depth_list": [2, 3, 4],
    "image_size": [128, 160, 192, 224],
    "label_mapping": [0, 217, 482, 491, 497, 566, 569, 571, 574, 701],
    "remap_imagenette": True,
    "base_batch_size": 64,
    "n_workers": 1,
    "bn_momentum": 0.1,
    "bn_eps": 1e-5,
    "dropout": 0.1,
    "base_stage_width": "proxyless",
    "width_mult_list": 1.0,
}

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set default path for Imagenet data
ImagenetDataProvider.DEFAULT_PATH = args["dataset_path"]

# Initialize Imagenet run configuration
run_config = ImagenetRunConfig(
    test_batch_size=args["base_batch_size"], n_worker=args["n_workers"]
)

# Initialize OFA MobileNetV3 network
ofa_network = OFAMobileNetV3(
    n_classes=run_config.data_provider.n_classes,
    bn_param=(args["bn_momentum"], args["bn_eps"]),
    dropout_rate=args["dropout"],
    base_stage_width=args["base_stage_width"],
    width_mult=args["width_mult_list"],
    ks_list=args["ks_list"],
    expand_ratio_list=args["expand_list"],
    depth_list=args["depth_list"],
)

arch_encoder = MobileNetArchEncoder(
    image_size_list=args["image_size"],
    depth_list=args["depth_list"],
    expand_list=args["expand_list"],
    ks_list=args["ks_list"],
)
    
# Function to validate model
def validate_model(model, data_loader, device):
    accuracies = AverageMeter()
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(data_loader), desc="Validate", position=0, leave=True)
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            accuracy = (output.argmax(1) == labels).float().mean()
            accuracies.update(accuracy.item(), images.size(0))
            # Update tqdm description with accuracy info
            pbar.set_postfix({"acc": accuracies.avg, "img_size": images.size(2)})
            pbar.update(1)
    return accuracies.avg


def load_model(path):
    run_manager = RunManager(path, ofa_network, run_config, init=False)
    run_manager.load_model()
    return run_manager


def test_subnet(run_manager, config):
    # print the config
    print("Testing subnet with config:")
    print(config)
    # set the active subnet
    ofa_network = run_manager.net
    ofa_network.set_active_subnet(ks=config["ks"], expand_ratio=config["e"], depth=config["d"])
    run_config.data_provider.assign_active_img_size(config["image_size"])
    run_manager.reset_running_statistics(net=ofa_network)
    data_loader = run_manager.run_config.test_loader
    accuracy = validate_model(ofa_network, data_loader, args["device"])
    print(f"Accuracy: {accuracy}")
    return accuracy


def test_random_subnet(run_manager, n_subnet=100):
    # randomly sample a sub-network
    configs = []
    accuracies = []
    for i in range(n_subnet):
        print(f"Testing subnet {i+1}/{n_subnet}")
        config = ofa_network.sample_active_subnet()
        config["image_size"] = random.choice(args["image_size"])
        configs.append(config)
        acc = test_subnet(run_manager, config)
        accuracies.append(acc)
    return configs, accuracies


# Load and test trained model
path = "trained_model"
run_manager = load_model(path)
run_manager.net.set_max_net()

# Test random subnets
configs, accuracies = test_random_subnet(run_manager, n_subnet=1000)

# Map all configs to a feature
features = []
for config in configs:
    feature = arch_encoder.arch2feature(config)
    features.append(feature)
    
# save the dataset into a pickle
with open("imagenette_features.pkl", "wb") as f:
    pickle.dump({
        "features": features,
        "accuracies": accuracies
    }, f)
