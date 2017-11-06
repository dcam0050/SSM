import pytest
import SAM
import os
from os.path import join
from ConfigParser import SafeConfigParser
import itertools
import collections
import numpy as np
import time

yarpFound = True
rosFound = True
try:
    import yarp
    yarp.Network.init()
except ImportError:
    yarpFound = False

try:
    import ros
except ImportError:
    rosFound = False

assert yarpFound or rosFound

curr_dir = os.getcwd()
data_dir_ext = "SAM_Data_Models/Data/"
model_dir_ext = "SAM_Data_Models/Models/"
conf_dir_ext = "../conf/model_conf"
sensory_dir_ext = "../conf/samSupervisor"


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
config_parser_dict = dict()
for model in func_params_dict['model']:
    conf_dir = join(curr_dir, conf_dir_ext, model)
    print conf_dir
    config_parser_dict[model] = SafeConfigParser()
    config_parser_dict[model].optionxform = str
    assert config_parser_dict[model].read(conf_dir + "/config.ini")
    assert config_parser_dict[model].has_section(model)
    print config_parser_dict[model].items(model)

sensory_dir = join(curr_dir, sensory_dir_ext, "sensory_level_conf.ini")
print sensory_dir
sensory_parser = SafeConfigParser()
assert sensory_parser.read(sensory_dir)
for model in func_params_dict['model']:
    assert sensory_parser.has_section(model)
    print sensory_parser.items(model)

# Config file parameters
comb_dict = collections.OrderedDict()
comb_dict['model_type'] = ['mrd', 'bgplvm']
comb_dict['model_mode'] = ['single', 'multiple']
comb_dict['Quser'] = range(1, 2, 1)
comb_dict['calibrateUnknown'] = ["False", "True"]
comb_dict['labelsAllowedList'] = ['Daniel,Greg,Sock,Tobias', 'Daniel,Tobias']
comb_dict['useBinWidth'] = ["False", "True"]
comb_dict['method'] = ['sumProb', 'mulProb']
test_comb_parameters = list(itertools.product(*comb_dict.values()))
test_comb_parameters = list(itertools.product(model_driver_pairs, [["new"]], test_comb_parameters))
test_comb_parameters = [flatten_tuple(x) for x in test_comb_parameters]

comb_keys = func_params_dict.keys() + comb_dict.keys()
combs = func_params_combs+test_comb_parameters
combs = combs[:2]
# 33 errors with 20 items per class 10 inducing 70 init 70 training
# __ errors with 200 items per class 10 inducing 70 init 70 training
# __ errors with 200 items per class 100 inducing 120 init 120 training
print '\n-----------------------------------------------------\n'
print "Parameter Combinations\n"
print len(combs), " tests"
print comb_keys
print '\n'.join([str(d) for d in combs])
print '\n-----------------------------------------------------\n'


@pytest.mark.usefixtures("get_error_list")
@pytest.mark.parametrize("params", combs)
def test_complete_train_model(params, get_error_list):
    error_list = get_error_list
    for z, x in enumerate(params):
        print comb_keys[z], " = ", x
    model = params[comb_keys.index('model')]
    driver = params[comb_keys.index('driver')]
    update_mode = params[comb_keys.index('update_mode')]

    model_config = config_parser_dict[model]
    if params[comb_keys.index('model_type')] is not 'None':
        for j in range(comb_keys.index('model_type'), len(comb_keys)):
            model_config.set(model, comb_keys[j], str(params[j]))
        model_name = model + "__" + driver + "__" + comb_keys.index('model_type') + "__exp"
    else:
        model_name = model + "__" + driver + "__mrd__exp"

    train_args = ["", join(curr_dir, data_dir_ext, model), join(curr_dir, model_dir_ext), driver, update_mode,
                  model, False]
    interaction_args = ["_", join(curr_dir, data_dir_ext, model), join(curr_dir, model_dir_ext, model_name),
                        sensory_parser, driver, "False"]

    message_list = ['heartbeat', 'EXIT']
    response_list = ['ack', 'ack']

    try:
        # Train model configuration
        conf_matrix = SAM.SAM_Core.trainSAMModel(train_args, model_config=model_config)
        assert conf_matrix is not False

        # Extract results from conf matrix
        np.fill_diagonal(conf_matrix, 0)
        total_error = np.sum(conf_matrix)
        print "Confusion Matrix", conf_matrix
        print "Total training error:", total_error
        error_list.append(total_error)

        # Load interaction
        mod = SAM.SAM_Core.interaction_yarp(interaction_args, model_config=model_config)
        rf = yarp.ResourceFinder()
        rf.setVerbose(True)
        rf.configure([])
        config_ret = mod.runModuleThreaded(rf)
        assert config_ret == 0

        for idx, msg in enumerate(message_list):
            message = yarp.Bottle()
            message.addString(msg)
            respond = yarp.Bottle()
            ret_bool = mod.safeRespond(message, respond)
            assert ret_bool
            assert respond.toString() == response_list[idx]

        mod.stopModule(wait=True)
        del mod
        print "check"
    except Exception as e:
        print e
        assert False
