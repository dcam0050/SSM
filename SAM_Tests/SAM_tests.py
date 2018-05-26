import pytest
import SAM
import os
from os.path import join
from ConfigParser import SafeConfigParser
import itertools
import collections
import numpy as np

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
func_params_dict['model'] = ["FacesTest", "ActionsTest"]
func_params_dict['driver'] = ['SAMDriver_interaction', 'SAMDriver_ARWin']
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

sensory_dir = join(curr_dir, sensory_dir_ext, "sensory_level_conf.ini")
print sensory_dir
sensory_parser = SafeConfigParser()
assert sensory_parser.read(sensory_dir)
for model in func_params_dict['model']:
    assert sensory_parser.has_section(model)
    print sensory_parser.items(model)

collectionMethod = collections.OrderedDict()
collectionMethod['collectionMethod'] = ['buffered', 'future_buffered', 'continuous']
collectionMethod['collectionMethod_buffer'] = ['0', '5']
collectionMethod_parameters = list(itertools.product(*collectionMethod.values()))
collectionMethod_parameters = [' '.join(j) for j in collectionMethod_parameters]

sensory_dict = collections.OrderedDict()
sensory_dict['collectionMethod'] = collectionMethod_parameters
# sensory_dict['visualise'] = ['False']
sensory_comb_parameters = [list(g) for g in list(itertools.product(*sensory_dict.values()))]
sensory_comb_keys = sensory_dict.keys()
sensory_key_offset = len(sensory_comb_keys)

comb_keys = func_params_dict.keys() + comb_dict.keys() + ['sensedict_' + k for k in sensory_comb_keys]
train_combs = func_params_combs+test_comb_parameters
train_combs_sub = []
train_combs_sub += (train_combs[:3])
# train_combs_sub += (train_combs[51:52])
train_combs = train_combs_sub
combs = list(itertools.product(train_combs, sensory_comb_parameters))
combs = [flatten_tuple(g) for g in combs]

# 33 errors with 20 items per class 10 inducing 70 init 70 training
# __ errors with 200 items per class 10 inducing 70 init 70 training
# __ errors with 200 items per class 100 inducing 120 init 120 training
print '\n-----------------------------------------------------\n'
print "Parameter Combinations\n"
print len(combs), " tests"
print comb_keys
print '\n'.join(['\t'.join(d) for d in combs])
print '\n-----------------------------------------------------\n'

message_list = [['heartbeat', 'ack'],
                # ['information', 'ack'],
                # ['portNames', 'ack'],
                ['reload', 'ack'],
                ['toggleVerbose', 'ack'],
                ['ask_X_label', 'ack'],
                ['ask_X_instance', 'ack'],
                ['EXIT', 'ack']]

class permanent_values:
    def __init__(self):
        self.mod = None
        self.this_model = None
        self.model_config = None
        self.train_args = None
        self.interaction_args = None
        self.val = None
        self.error_list = []
        self.do_train = True
        self.old_params = ''


pv = permanent_values()


class TestEverything(object):
    scenario_keys = ['params']
    scenario_parameters = combs

    def test_param_setup(self, params):
        if pv.old_params != params[:-sensory_key_offset]:
            pv.old_params = params[:-sensory_key_offset]
            pv.do_train = True

        pv.sense_conf = params[-sensory_key_offset:]
        if pv.do_train:
            for z, b in enumerate(params):
                print comb_keys[z], ' = ', b
            pv.this_model = params[comb_keys.index('model')]
            pv.driver = params[comb_keys.index('driver')]
            pv.update_mode = params[comb_keys.index('update_mode')]

            pv.model_config = config_parser_dict[pv.this_model]
            if params[comb_keys.index('model_type')] is not 'None':
                for m in range(comb_keys.index('model_type'), len(comb_keys)):
                    pv.model_config.set(pv.this_model, comb_keys[m], str(params[m]))
                pv.model_name = pv.this_model + "__" + pv.driver + "__" + comb_keys.index('model_type') + "__exp"
            else:
                pv.model_name = pv.this_model + "__" + pv.driver + "__mrd__exp"

            pv.train_args = ["", join(curr_dir, data_dir_ext, pv.this_model), join(curr_dir, model_dir_ext),
                               pv.driver, pv.update_mode, pv.this_model, False]

            pv.interaction_args = ["_", join(curr_dir, data_dir_ext, pv.this_model),
                                     join(curr_dir, model_dir_ext, pv.model_name), sensory_parser, pv.driver, "False"]

            pv.mod = None
            pv.ret_reply = []

        assert True

    def test_train_model(self, params):
        try:
            if pv.do_train:
                # Train model configuration
                conf_matrix = SAM.SAM_Core.trainSAMModel(pv.train_args, model_config=pv.model_config)
                assert conf_matrix is not False

                # Extract results from conf matrix
                np.fill_diagonal(conf_matrix, 0)
                total_error = np.sum(conf_matrix)
                print "Confusion Matrix", conf_matrix
                print "Total training error:", total_error
                pv.error_list.append(total_error)
                pv.do_train = False
            else:
                assert True
        except Exception as e:
            print e
            assert False

    def test_interaction_model_loading(self, params):
        try:
            for m in range(len(sensory_comb_keys)):
                print sensory_parser.get(pv.this_model, sensory_comb_keys[m])
                sensory_parser.set(pv.this_model, sensory_comb_keys[m], str(pv.sense_conf[m]))
                print sensory_parser.get(pv.this_model, sensory_comb_keys[m])

            pv.mod = SAM.SAM_Core.interaction_yarp(pv.interaction_args, model_config=pv.model_config)
            rf = yarp.ResourceFinder()
            rf.setVerbose(True)
            rf.configure([])
            config_ret = pv.mod.runModuleThreaded(rf)
            if config_ret != 0:
                pv.mod = None
            assert config_ret == 0
        except Exception as e:
            print e
            assert False

    def test_interaction_model_respond(self, params):
        try:
            assert pv.mod is not None
            respond_list = []
            for msg in message_list:
                if 'X' in msg[0]:
                    subs = pv.this_model[:-5].lower()
                    msg[0] = msg[0].replace('X', subs)

                message = yarp.Bottle()
                message.addString(msg[0])
                respond = yarp.Bottle()
                ret_bool = pv.mod.safeRespond(message, respond)
                assert ret_bool
                respond_list.append(respond.toString() == msg[1])
            assert all(respond_list)
        except Exception as e:
            print e
            assert False

    def test_interaction_model_close(self, params):
        try:
            ret = pv.mod.stopModule(wait=True)
            del pv.mod
            pv.mod = None
            assert pv.mod is None
        except Exception as e:
            print e
            assert False
