
# source file pairs 
# (chart_file_identifier,sar_file_identifier,region_identifier)
chart_sar_pairs = [("20181203", "S1B_EW_GRDM_1SDH_20181202T081815_20181202T081914_013860_019B24_C946", "AP"),
                   ("20181209", "S1B_EW_GRDM_1SDH_20181210T085100_20181210T085208_013977_019EF9_260F", "AP"),
                   ("20181210", "S1B_EW_GRDM_1SDH_20181210T235006_20181210T235105_013986_019F43_DA57", "AP"), 
                   # ("20181217", "S1B_EW_GRDH_1SDH_20181215T235912_20181216T000012_014059_01A1A8_3C08", "AP"), # Same ice chart
                   # ("20181217", "S1B_EW_GRDH_1SDH_20181217T234258_20181217T234358_014088_01A2A3_7147", "AP"), # Same ice chart
                   ("20181220", "S1A_EW_GRDM_1SDH_20181220T081915_20181220T082019_025106_02C581_BEF8", "AP"), 
                   # ("20181228", "S1A_EW_GRDM_1SSH_20181228T071250_20181228T071350_025222_02C9CE_F20B", "AP"), # Only HH band
                   ("20171106", "S1B_EW_GRDM_1SDH_20171107T222550_20171107T222655_008181_00E75F_78D0", "WS"),
                   ("20171223", "S1B_EW_GRDM_1SDH_20171223T224200_20171223T224304_008852_00FC64_0911", "WS"), 
                   ("20180104", "S1B_EW_GRDM_1SDH_20180104T224159_20180104T224303_009027_010213_47AC", "WS"), 
                   ("20180222", "S1B_EW_GRDM_1SDH_20180222T232234_20180222T232338_009742_01197F_E3BC", "WS"), 
                   ("20180223", "S1B_EW_GRDM_1SDH_20180223T005954_20180223T010058_009743_011988_1573", "WS"), 
                   ("20180226", "S1B_EW_GRDM_1SDH_20180226T012413_20180226T012517_009787_011B06_A02E", "WS"), 
                   ("20190313", "S1B_EW_GRDM_1SDH_20190313T232241_20190313T232345_015342_01CB99_7DC1", "WS"), 
                   ("20200117", "S1B_EW_GRDM_1SDH_20200117T220139_20200117T220243_019862_02590A_7B65", "WS"), 
                   ("20200305", "S1B_EW_GRDM_1SDH_20200305T003547_20200305T003652_020549_026F0F_6ED1", "WS"), 
                   ("20200313", "S1B_EW_GRDM_1SDH_20200313T010815_20200313T010920_020666_0272CA_D7C3", "WS"), 
                   # ("20201104", "S1B_EW_GRDM_1SDH_20201105T220953_20201105T221057_024135_02DE11_C88F", "WS"), # Weird cropping
                   # ("20201105", "S1B_EW_GRDM_1SDH_20211104T231454_20211104T231545_029444_038388_741F", "WS"), # Weird cropping
                   ("20211223", "S1B_EW_GRDM_1SDH_20211222T231452_20211222T231543_030144_039978_72FF", "WS")]

test_chart_sar_pairs = [('20221216', 'S1A_EW_GRDM_1SDH_20221216T232348_20221216T232453_046363_058DAD_0BBF', 'WS'),
                        ('20221222', 'S1A_EW_GRDM_1SDH_20221222T010923_20221222T011027_046437_059035_3C1', 'WS'),
                        ('20230112', 'S1A_EW_GRDM_1SDH_20230112T203241_20230112T203345_046755_059AF6_1E95', 'WS')]

# bespoke sea ice concentration categories / classes
# these classes will supercede the original ice chart categories if specified
new_classes = {None: None,  # use original categories
                "binary": {0: range(0, 2), 
                           1: range(13, 93)},
                "ternary": {0: range(0, 2), 
                            1: range(13, 78),
                            2: range(78, 93)}, # This division was arbitrary, we might review this carefully later                           
                "multiclass": [0, 1, 13, 14, 24, 46, 47, 68, 78, 79, 81, 90, 91, 92]}

# Scikit learn hyperparameters for tuning classification algorithms

model_parameters = {'RandomForest': {'bootstrap': [True, False],
                                     'max_depth': [25, 50, 75, 100, None],
                                    #  'min_samples_leaf': [100, 2000, 10000],
                                    #  'min_samples_split': [2000, 5000, 10000],
                                     'n_estimators': [200, 1000, 2000]},
                    'DecisionTree': {'splitter': ['best', 'random'],
                                     'min_samples_leaf': [100, 2000, 10000],
                                     'min_samples_split': [2000, 5000, 10000],
                                     'max_depth': [25, 50, 75, 100, None]},
                    'KNeighbors': {},
                    'SGD': {},
                    'MLP': {},
                    'SVC': {'kernel': ['linear', 'poly'],
                            'degree': [3, 10, 50],
                            'C': [1.0, 5.0, 10.0]},
                    'LogisticRegression': {'penalty': ['l1', 'l2', 'elasticnet'],
                                           'C': [1.0, 5.0, 10.0],
                                           'l1_ratio': [0.0, 0.5, 1.0]}}