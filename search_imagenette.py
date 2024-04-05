import pickle
import numpy as np
import torch
from ofa.nas.accuracy_predictor import MobileNetArchEncoder
from ofa.nas.efficiency_predictor import Mbv3FLOPsModel
from ofa.nas.search_algorithm import EvolutionFinder
from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from predictor_imagenette import Predictor
from viz import draw_arch
from peak_memory_efficiency import PeakMemoryEfficiency

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

# Set default path for Imagenet data
ImagenetDataProvider.DEFAULT_PATH = args["dataset_path"]

# Initialize Imagenet run configuration
run_config = ImagenetRunConfig(
    test_batch_size=args["base_batch_size"], n_worker=args["n_workers"]
)

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

# Load the pickle file
with open("imagenette_features.pkl", "rb") as f:
    data = pickle.load(f)

features = np.array(data["features"])
accuracies = np.array(data["accuracies"]).reshape(-1, 1)

X_test = torch.tensor(features[int(0.8 * len(features)):], dtype=torch.float32)
y_test = torch.tensor(accuracies[int(0.8 * len(accuracies)):], dtype=torch.float32)

X_test, y_test = X_test.to(args["device"]), y_test.to(args["device"])

arch_encoder = MobileNetArchEncoder(
    image_size_list=args["image_size"],
    depth_list=args["depth_list"],
    expand_list=args["expand_list"],
    ks_list=args["ks_list"],
)

# Load the model
model = Predictor.load_model("imagenette_acc_predictor.pth", input_size=124, arch_encoder=arch_encoder, device=args["device"])

# Evaluate the model
print("Verifying the model...")
evaluation_loss = model.evaluate(X_test, y_test)
print(f"Evaluation Loss: {evaluation_loss:.3e}")

efficiency_predictor = PeakMemoryEfficiency(ofa_net=ofa_network)
#efficiency_predictor = Mbv3FLOPsModel(ofa_net=ofa_network)

finder = EvolutionFinder(accuracy_predictor=model, efficiency_predictor=efficiency_predictor, max_time_budget=10)
best_valids, best_info = finder.run_evolution_search(500e3, verbose=True)

config = best_info[1]

draw_arch(
    ofa_net=ofa_network,
    resolution=config["image_size"],
    out_name="nets_graphs/search_result",
)


print("Best Information:", best_info)