# Each group is a list of save_path values from run_all.py that will be
# overlaid on the same plot.  Groups can hold any combination:
#
#   Train/test/valid splits:  ['ptb_train', 'ptb_test', 'ptb_valid']
#   A single full dataset:    ['ptb_full']
#   Cross-dataset comparison: ['ptb_full', 'wiki2_full']
#   Mixed:                    ['wiki2_train', 'wiki2M_test1', 'wiki2M_test2']
#
# Missing files are silently skipped, so unused entries are harmless.

DATASETS = [
    ['ptb_train', 'ptb_test', 'ptb_valid'],
    ['wiki2_train', 'wiki2_test', 'wiki2_valid'],
    ['wiki2C_train', 'wiki2C_test', 'wiki2C_valid'],
    ['wiki2R_train', 'wiki2R_test', 'wiki2R_valid'],
    ['wiki2_train', 'wiki2M_test1', 'wiki2M_test2', 'wiki2M_valid1'],
    ['wiki2C_train', 'wiki2MC_test1', 'wiki2MC_test2', 'wiki2MC_valid1'],
    ['wiki2Samples1_train', 'wiki2Samples1_test', 'wiki2Samples1_valid'],
    ['wiki2Samples2_train', 'wiki2Samples2_test', 'wiki2Samples2_valid'],
    ['wiki2Samples3_train', 'wiki2Samples3_test', 'wiki2Samples3_valid'],
    ['wiki2Samples4_train', 'wiki2Samples4_test', 'wiki2Samples4_valid'],
    ['wiki2Resample1_train', 'wiki2Resample1_test', 'wiki2Resample1_valid'],
    ['wiki2Resample2_train', 'wiki2Resample2_test', 'wiki2Resample2_valid'],
    ['wiki2Resample3_train', 'wiki2Resample3_test', 'wiki2Resample3_valid'],
    ['wiki2Resample4_train', 'wiki2Resample4_test', 'wiki2Resample4_valid'],
    ['wiki2HomogenousH1_train', 'wiki2HomogenousH1_test', 'wiki2HomogenousH1_valid'],
    ['wiki2HomogenousH2_train', 'wiki2HomogenousH2_test', 'wiki2HomogenousH2_valid'],
    ['wiki2HomogenousCH1_train', 'wiki2HomogenousCH1_test', 'wiki2HomogenousCH1_valid'],
    ['wiki2HomogenousCH2_train', 'wiki2HomogenousCH2_test', 'wiki2HomogenousCH2_valid'],
    ['text8w_small_train', 'text8w_small_test', 'text8w_small_valid'],
    ['ptb_text8_train', 'ptb_text8_test', 'ptb_text8_valid'],
    ['text8w_S_train', 'text8w_S_test', 'text8w_S_valid'],
    ['text8w_train', 'text8w_test', 'text8w_valid'],
    ['wiki19_train', 'wiki19_test', 'wiki19_valid'],
    ['wiki19C_train', 'wiki19C_test', 'wiki19C_valid'],
    ['wiki19L_train', 'wiki19L_test', 'wiki19L_valid'],
    ['wiki19Lwor_train', 'wiki19Lwor_test', 'wiki19Lwor_valid'],
    ['wiki103_train', 'wiki103_test', 'wiki103_valid'],
    ['wiki103C_train', 'wiki103C_test', 'wiki103C_valid'],
    # ['wiki103R_train', 'wiki103R_test', 'wiki103R_valid'],
]
