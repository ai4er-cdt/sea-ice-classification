"""
AI4ER GTC - Sea Ice Classification
Script for feeding test data into scikit-learn
classifiers saving the model output to wandb
"""
import os
import wandb
import numpy as np
import xarray as xr
import rioxarray as rxr
from train_scikit import define_band3, normalize_sar, recategorize_chart
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from argparse import ArgumentParser
from pathlib import Path
from constants import new_classes
from joblib import load

if __name__ == "__main__":

    parser = ArgumentParser(description="Sea Ice Segmentation Test")
   
    parser.add_argument("--username", type=str, help="wandb username")
    parser.add_argument("--name", type=str, help="Name of wandb run")
    parser.add_argument("--model_name", type=str, help="path to the model")
    parser.add_argument("--classification_type", default=None, type=str, help="[binary,ternary,multiclass]")
    parser.add_argument("--sar_band3", default="angle", type=str, choices=["angle", "ratio"],
                        help="Whether to use incidence angle or HH/HV ratio in third band")
    parser.add_argument("--sar_folder", default='sar', type=str, help="SAR output folder name")
    parser.add_argument("--chart_folder", default='chart', type=str, help="Ice Chart output folder name")
    args = parser.parse_args()

    # standard input dirs
    output_folder = Path(open("tile.config").read().strip())
    sar_folder = f"{output_folder}/test/{args.sar_folder}"
    chart_folder = f"{output_folder}/test/{args.chart_folder}"
    class_categories = new_classes[args.classification_type]
    n_classes = len(class_categories)
    is_binary = True if args.classification_type == 'binary' else False
    
    sar_filenames = os.listdir(sar_folder)
    chart_filenames = os.listdir(chart_folder)
    
    if args.sample == 'True':
        
        sample_n = np.random.randint(len(sar_filenames), size=(50))
        sar_filenames = [sar_filenames[i] for i in sample_n]
        chart_filenames = [chart_filenames[i] for i in sample_n]

    test_x_lst = [normalize_sar(define_band3(rxr.open_rasterio(os.path.join(sar_folder, x), parse_coordinates=True), sar_band3=args.sar_band3), sar_band3=args.sar_band3).values for x in sar_filenames]
    test_y_lst = [recategorize_chart(rxr.open_rasterio(os.path.join(chart_folder, x), parse_coordinates=True, masked=True).values, class_categories) for x in chart_filenames]   
    
    test_x = np.stack(test_x_lst)
    test_y = np.stack(test_y_lst)

    # Reorder dimensions
    x_test = np.moveaxis(test_x, 1, -1)
    X_test_data = x_test.reshape(-1, 3)
    y_test = np.moveaxis(test_y, 1, -1)
    Y_test_data = y_test.reshape(-1, 1)

    # wandb logging
    wandb.init(id=args.name, project="sea-ice-classification", resume="must")
    api = wandb.Api()
    run = api.run(f"{args.username}/sea-ice-classification/{args.name}")
    
    model = load(Path('scikit_models/' + args.model_name))
    
    y_pred = model.predict(X_test_data, )
    y_prob = model.predict_proba(X_test_data)

    labels = list(class_categories.keys())
    
    print(f"Accuracy: {accuracy_score(Y_test_data, y_pred)*100}")

    print(classification_report(Y_test_data, y_pred))
    print(confusion_matrix(Y_test_data, y_pred))
