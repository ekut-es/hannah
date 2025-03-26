import os
import re
import json
import plotext as plt
import matplotlib
import argparse

def extract_val_errors(folder_path):
    val_errors = []
    pattern = re.compile(r"model_(\d+).json")

    for root, _, files in os.walk(folder_path):
        for file in files:
            match = pattern.match(file)
            if match:
                model_num = int(match.group(1))
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    if "val_error" in data["metrics"]:
                        val_errors.append((model_num, data["metrics"]["val_error"]))
        break

    # Sort by model number
    val_errors.sort(key=lambda x: x[0])
    return val_errors

def plot_val_errors(val_errors):
    models, errors = zip(*val_errors)
    plt.plot(range(len(models)), errors, color='skyblue')
    plt.xlabel("Model Number")
    plt.ylabel("Validation Error")
    plt.title("Validation Errors for Models")
    # plt.xticks(models, [f"Model {m}" for m in models], rotation=45)
    # plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Plot val_errors from performance_data")
    # parser.add_argument("folder", help="Path to the folder containing the performance data files")
    # args = parser.parse_args()
    os.chdir(os.path.dirname(__file__))
    f = "./trained_models/ae_nas_cifar10_weight_250k/embedded_vision_net/performance_data"
    val_errors = extract_val_errors(f) # args.folder)
    if val_errors:
        plot_val_errors(val_errors)
    else:
        print("No val_error values found in the specified folder.")