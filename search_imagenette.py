import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ofa.nas.accuracy_predictor import MobileNetArchEncoder
from ofa.nas.efficiency_predictor import Mbv3FLOPsModel, BaseEfficiencyModel
from ofa.nas.search_algorithm import EvolutionFinder
from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from mcunet.utils.pytorch_utils import count_peak_activation_size

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

# Define the model architecture
class Predictor(nn.Module):
    def __init__(self, input_size, base_acc):
        super(Predictor, self).__init__()

        self.device = args["device"]
        self.base_acc = base_acc
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1, bias=False),
        )
        
        self.model = self.model.to(self.device)

    def forward(self, x):
        return self.model(x) + self.base_acc
    
    def predict_acc(self, arch_dict_list):
        X = [self.arch_encoder.arch2feature(arch_dict) for arch_dict in arch_dict_list]
        X = torch.tensor(np.array(X)).float().to(self.device)
        return self.forward(X)

# Load the trained model
def load_model(model_path, input_size):
    model = Predictor(input_size, 0.9774)
    model.load_state_dict(torch.load(model_path))
    return model

# Perform evaluation check
def evaluate(model, X, y, batch_size=32):
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.L1Loss()

    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(inputs)

    avg_loss = total_loss / len(dataset)
    return avg_loss

# Load the pickle file
with open("imagenette_features.pkl", "rb") as f:
    data = pickle.load(f)

features = np.array(data["features"])
accuracies = np.array(data["accuracies"]).reshape(-1, 1)

X_test = torch.tensor(features[int(0.8 * len(features)):], dtype=torch.float32)
y_test = torch.tensor(accuracies[int(0.8 * len(accuracies)):], dtype=torch.float32)

X_test, y_test = X_test.to(args["device"]), y_test.to(args["device"])

# Load the model
model = load_model("imagenette_acc_predictor.pth", input_size=X_test.shape[1])

# Evaluate the model
evaluation_loss = evaluate(model, X_test, y_test)
print("Evaluation Loss:", evaluation_loss)

############ CODE FOR NAS ############

class PeakMemoryEfficiency(BaseEfficiencyModel):
    def get_efficiency(self, arch_dict):
        self.ofa_net.set_active_subnet(**arch_dict)
        subnet = self.ofa_net.get_active_subnet()
        if torch.cuda.is_available():
            subnet = subnet.cuda()
        data_shape = (1, 3, arch_dict["image_size"], arch_dict["image_size"])
        peak_memory = count_peak_activation_size(subnet, data_shape)
        print(peak_memory)
        return peak_memory

arch_encoder = MobileNetArchEncoder(
    image_size_list=args["image_size"],
    depth_list=args["depth_list"],
    expand_list=args["expand_list"],
    ks_list=args["ks_list"],
)

model.arch_encoder = arch_encoder

#efficiency_predictor = PeakMemoryEfficiency(ofa_net=ofa_network)
efficiency_predictor = Mbv3FLOPsModel(ofa_net=ofa_network)

finder = EvolutionFinder(accuracy_predictor=model, efficiency_predictor=efficiency_predictor)
best_valids, best_info = finder.run_evolution_search(100, verbose=True)

print("Best Information:", best_info)