# (DecisionTree, binary, angle)
python test_scikit.py model_name TODO --sar_folder sar --chart_folder chart
# (DecisionTree, binary, ratio)
python test_scikit.py model_name TODO --sar_folder sar --chart_folder chart --sar_band3 ratio

# (DecisionTree, ternary, angle)
python test_scikit.py model_name TODO --sar_folder sar --chart_folder chart --classification_type ternary
# (DecisionTree, ternary, ratio)
python test_scikit.py model_name TODO --sar_folder sar --chart_folder chart --sar_band3 ratio --classification_type ternary
