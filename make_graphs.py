import os
import json
import matplotlib.pyplot as plt
import random

folder_path = 'subnets_data/'
search_folders = [folder for folder in os.listdir(folder_path) if folder.startswith('search_')]
info_array = []

for folder in search_folders:
    info_file = os.path.join(folder_path, folder, 'info.json')
    with open(info_file) as f:
        info_data = json.load(f)
        info_array.append(info_data)
        
# array element example : {"accuracy": 0.9824778437614441, "peak_memory": 327680, "config": {"ks": [5, 3, 3, 7, 7, 7, 5, 7, 5, 7, 5, 7, 3, 3, 5, 7, 3, 3, 7, 3], "e": [4, 6, 6, 4, 4, 6, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3], "d": [4, 4, 4, 4, 4], "image_size": 128}, "memory_history": [114688.0, 196608.0, 131072.0, 327680.0, 327680.0, 90112.0, 172032.0, 319488.0, 172032.0, 172032.0, 319488.0, 172032.0, 122880.0, 221184.0, 122880.0, 122880.0, 122880.0, 34816.0, 71680.0, 133120.0, 71680.0, 40960.0, 71680.0, 40960.0, 71680.0, 133120.0, 71680.0, 71680.0, 76800.0, 20480.0, 35840.0, 66560.0, 35840.0, 35840.0, 66560.0, 35840.0, 35840.0, 66560.0, 35840.0, 35840.0, 61440.0, 37888.0, 50176.0, 93184.0, 50176.0, 50176.0, 93184.0, 50176.0, 50176.0, 93184.0, 50176.0, 50176.0, 53760.0, 13312.0, 17920.0, 33280.0, 17920.0, 10240.0, 17920.0, 10240.0, 10240.0, 17920.0, 10240.0, 17920.0, 2240.0, 2280.0]}

# graph accuracy vs memory
for info in info_array:
    plt.scatter(info['peak_memory'], info['accuracy'])
plt.savefig('graphs/subnets/accuracy_vs_memory.png')
plt.clf()

# graph number of layers vs stage with colored lines
min_acc = min([info['accuracy'] for info in info_array])
max_acc = max([info['accuracy'] for info in info_array])
colors = ['red', 'orange', 'yellow', 'green', 'blue']
fig, axs = plt.subplots(4, 4, figsize=(12, 12))

threshold = 0.986
samples = [info for info in info_array if info['accuracy'] >= threshold]
random.shuffle(samples)
samples = samples[:16]

for i, info in enumerate(samples):
    acc = info['accuracy']
    layers = info['config']['d']
    color_index = int((acc - min_acc) / (max_acc - min_acc) * (len(colors) - 1))
    color = colors[color_index]
    ax = axs[i // 4, i % 4]
    ax.plot(layers, color=color, marker='o', linestyle='--')
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.set_ylim(1.7, 4.3)
    ax.set_yticks([2, 3, 4])
    ax.set_yticklabels([2, 3, 4])
    ax.text(0.85, 0.1, round(acc, 3), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

plt.savefig('graphs/subnets/number_of_layers_vs_stage_good.png')
plt.clf()