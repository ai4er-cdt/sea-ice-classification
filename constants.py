
# source file pairs 
# (chart_file_identifier,sar_file_identifier,region_identifier)
chart_sar_pairs = [('20181203', 'S1B_EW_GRDM_1SDH_20181202T081815_20181202T081914_013860_019B24_C946', 'AP'),
                   ('20181209', 'S1B_EW_GRDM_1SDH_20181210T085100_20181210T085208_013977_019EF9_260F', 'AP'),
                   ('20181210', 'S1B_EW_GRDM_1SDH_20181210T235006_20181210T235105_013986_019F43_DA57', 'AP'), 
                   # ('20181217', 'S1B_EW_GRDH_1SDH_20181215T235912_20181216T000012_014059_01A1A8_3C08', 'AP'), # Same ice chart
                   # ('20181217', 'S1B_EW_GRDH_1SDH_20181217T234258_20181217T234358_014088_01A2A3_7147', 'AP'), # Same ice chart
                   ('20181220', 'S1A_EW_GRDM_1SDH_20181220T081915_20181220T082019_025106_02C581_BEF8', 'AP'), 
                   # ('20181228', 'S1A_EW_GRDM_1SSH_20181228T071250_20181228T071350_025222_02C9CE_F20B', 'AP'), # Only HH band
                   ('20171106', 'S1B_EW_GRDM_1SDH_20171107T222550_20171107T222655_008181_00E75F_78D0', 'WS'),
                   ('20171223', 'S1B_EW_GRDM_1SDH_20171223T224200_20171223T224304_008852_00FC64_0911', 'WS'), 
                   ('20180104', 'S1B_EW_GRDM_1SDH_20180104T224159_20180104T224303_009027_010213_47AC', 'WS'), 
                   ('20180222', 'S1B_EW_GRDM_1SDH_20180222T232234_20180222T232338_009742_01197F_E3BC', 'WS'), 
                   ('20180223', 'S1B_EW_GRDM_1SDH_20180223T005954_20180223T010058_009743_011988_1573', 'WS'), 
                   ('20180226', 'S1B_EW_GRDM_1SDH_20180226T012413_20180226T012517_009787_011B06_A02E', 'WS'), 
                   ('20190313', 'S1B_EW_GRDM_1SDH_20190313T232241_20190313T232345_015342_01CB99_7DC1', 'WS'), 
                   ('20200117', 'S1B_EW_GRDM_1SDH_20200117T220139_20200117T220243_019862_02590A_7B65', 'WS'), 
                   ('20200305', 'S1B_EW_GRDM_1SDH_20200305T003547_20200305T003652_020549_026F0F_6ED1', 'WS'), 
                   ('20200313', 'S1B_EW_GRDM_1SDH_20200313T010815_20200313T010920_020666_0272CA_D7C3', 'WS'), 
                   # ('20201104', 'S1B_EW_GRDM_1SDH_20201105T220953_20201105T221057_024135_02DE11_C88F', 'WS'), # Weird cropping
                   # ('20201105', 'S1B_EW_GRDM_1SDH_20211104T231454_20211104T231545_029444_038388_741F', 'WS'), # Weird cropping
                   ('20211223', 'S1B_EW_GRDM_1SDH_20211222T231452_20211222T231543_030144_039978_72FF', 'WS')]

# bespoke sea ice concentration categories / classes
# these classes will supercede the original ice chart categories if specified
new_classes = {None: None,  # use original categories
                'binary': {0: [247, 0, 1], 
                          1: [13, 14, 24, 46, 47, 68, 78, 79, 81, 90, 91, 92], # Perhaps use list(range(13,93))?
                          None: None}, # Do we need None here as well as overall? 
                'ternary': {0: [247, 0, 1], 
                           1: [13, 14, 24, 46, 47, 68], 
                           2: [78, 79, 81, 90, 91, 92], # This division was arbitrary, we might review this carefully later
                           None: None}}
