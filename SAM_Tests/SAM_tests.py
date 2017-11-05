import pytest
import SAM
import os
from os.path import join
from ConfigParser import SafeConfigParser
import itertools
import collections
import numpy as np

curr_dir = os.getcwd()
data_dir_ext = "SAM_Data_Models/Data/"
model_dir_ext = "SAM_Data_Models/Models/"
conf_dir_ext = "../conf/model_conf"

error_list = []
def flatten_tuple(inp):
    return [element for tupl in inp for element in tupl]

# Train Model Function Parameters
func_params_dict = collections.OrderedDict()
func_params_dict['model'] = ["FacesTest"]
func_params_dict['driver'] = ['SAMDriver_interaction']
func_params_dict['update_mode'] = ['new', "update"]
func_params_dict['update_mode'] = [[x] for x in func_params_dict['update_mode']]
model_driver_pairs = list(itertools.izip(func_params_dict['model'], func_params_dict['driver']))
func_params_combs = list(itertools.product(model_driver_pairs, func_params_dict['update_mode'], [["None"]]))
func_params_combs = [flatten_tuple(x) for x in func_params_combs]

# Load config.ini from conf dir for each model
parser_dict = dict()
for model in func_params_dict['model']:
    conf_dir = join(curr_dir, conf_dir_ext, model)
    print conf_dir
    parser_dict[model] = SafeConfigParser()
    parser_dict[model].optionxform = str
    assert parser_dict[model].read(conf_dir + "/config.ini")
    assert parser_dict[model].has_section(model)
    print parser_dict[model].items(model)

# Config file parameters
comb_dict = collections.OrderedDict()
comb_dict['model_type'] = ['mrd', 'bgplvm']
comb_dict['model_mode'] = ['single', 'multiple']
comb_dict['Quser'] = range(0, 3, 1)
comb_dict['calibrateUnknown'] = ["False", "True"]
comb_dict['labelsAllowedList'] = ['Daniel,Greg,Sock,Tobias', 'Daniel,Tobias']
comb_dict['useBinWidth'] = ["False", "True"]
comb_dict['method'] = ['sumProb', 'mulProb']
test_comb_parameters = list(itertools.product(*comb_dict.values()))
test_comb_parameters = list(itertools.product(model_driver_pairs, [["new"]], test_comb_parameters))
test_comb_parameters = [flatten_tuple(x) for x in test_comb_parameters]


comb_keys = func_params_dict.keys() + comb_dict.keys()
combs = func_params_combs+test_comb_parameters
# combs = combs[50:]
# 33 errors with 20 items per class 10 inducing 70 init 70 training
# __ errors with 200 items per class 10 inducing 70 init 70 training
# __ errors with 200 items per class 100 inducing 120 init 120 training
print '\n-----------------------------------------------------\n'
print "Parameter Combinations\n"
print len(combs), " tests"
print comb_keys
print '\n'.join([str(d) for d in combs])
print '\n-----------------------------------------------------\n'

# @pytest.fixture
# def prt(a):
#     '''Returns a Wallet instance with a zero balance'''
#     print "fixture", a
#     return True


@pytest.mark.parametrize("params", combs)
def test_complete_train_model(params):
    for z, x in enumerate(params):
        print comb_keys[z], " = ", x
    model = params[comb_keys.index('model')]
    driver = params[comb_keys.index('driver')]
    update_mode = params[comb_keys.index('update_mode')]

    args = ["", join(curr_dir, data_dir_ext, model), join(curr_dir, model_dir_ext), driver, update_mode, model, False]

    cfg = parser_dict[model]
    if params[comb_keys.index('model_type')] is not 'None':
        for j in range(comb_keys.index('model_type'), len(comb_keys)):
            cfg.set(model, comb_keys[j], str(params[j]))

    try:
        conf_matrix = SAM.SAM_Core.trainSAMModel(args, config=cfg)
        print conf_matrix
        np.fill_diagonal(conf_matrix, 0)
        total_error = np.sum(conf_matrix)
        print total_error
        error_list.append(total_error)
        assert total_error < 150
    except Exception as e:
        print e
        assert False