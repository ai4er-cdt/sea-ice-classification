# (DecisionTree, binary, angle)
python test_scikit.py model_name='' --sar_folder sar_no_stride --chart_folder chart_no_stride --sample --pct_sample 0.1
# (DecisionTree, binary, ratio)
python test_scikit.py model_name='' --sar_folder sar_no_stride --chart_folder chart_no_stride --sample --pct_sample 0.1 --sar_band3 ratio

# (DecisionTree, ternary, angle)
python test_scikit.py model_name='' --sar_folder sar_no_stride --chart_folder chart_no_stride --sample --pct_sample 0.1 --classification_type ternary
# (DecisionTree, ternary, ratio)
python test_scikit.py model_name='' --sar_folder sar_no_stride --chart_folder chart_no_stride --sample --pct_sample 0.1 --sar_band3 ratio --classification_type ternary
