#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import sys
import time
from ConfigParser import SafeConfigParser
from SAM.SAM_Core import SAMDriver as Driver
from SAM.SAM_Core.SAM_utils import initialiseModels, timeout
import readline
import warnings
import numpy as np
import yarp
import logging
from os.path import join
import os
from operator import itemgetter
import thread
warnings.simplefilter("ignore")
np.set_printoptions(precision=2)
matplotlib.use("agg")

## @ingroup icubclient_SAM_Core


class interactionSAMModel(yarp.RFModule):
    """Generic interaction function

        Description:
            Generic interaction function that carries out live collection of data, classification of said data,
            interaction with other modules requesting the classification and generation of training outputs for recall.
            The parameters for the interaction function are loaded from the config file specified in the `config_path`
            parameter of the default context file for samSupervisor.py. An example of the configuration structure is
            shown below.

        Example:
            [Model Name]
            dataIn = <portName/ofInputData>:i <dataType of Port>
            dataOut = <portName/ofOutputData>:o <dataType of Port>
            rpcBase = <portName/ofRpcPort>
            latentModelPort = <portName/ofLatentModelPort>
            call_sign = ask_<X>_label,ask_<X>_instance
            collectionMethod = collectionMethod lengthOfBuffer

            [Faces]
            dataIn = /sam/faces/imageData:i ImageRgb
            dataOut = /sam/faces/imageData:o ImageMono
            rpcBase = /sam/faces/rpc
            latentModelPort = /sam/faces/latentModel
            callSign = ask_face_label,ask_face_instance
            collectionMethod = future_buffered 3

        Args:
            dataIn : The port name for the port that will received the data to be classified and the dataType to be expected.
            dataOut : The port name for the port that will output generated data and the dataType to be transmitted.
            rpcBase : The rpc port that will receive external requests. This is usually controlled by samSupervisor.py.
            latentModelPort : The port name for the port that will transmit the image showing the latent model of the current model
            call_sign : The commands that will trigger a classify from data event or a generate from label event.
            collectionMethod : The collection method to be used. Either `buffered`, `future_buffered` or `continuous`.
                               Followed by an integer indicating the length of the buffer to be used. In the case of
                               `continuous` the buffer length is the maximum number of classification labels to be
                               stored in a First In Last Out (FILO) configuration. Otherwise the buffer length indicates
                                the number of data frames that are required for classification to take place.
    """
    def __init__(self):
        """
        Initialisation of the interaction function
        """
        yarp.RFModule.__init__(self)
        self.mm = None
        self.dataPath = None
        self.configPath = None
        self.modelPath = None
        self.driverName = ''
        self.model_type = None
        self.model_mode = None
        self.textLabels = None
        self.classifiers = None
        self.classif_thresh = None
        self.verbose = None
        self.Quser = None
        self.listOfModels = None
        self.portsList = []
        self.svPort = None
        self.latentPort = None
        self.labelPort = None
        self.instancePort = None
        self.callSignList = []
        self.inputBottle = yarp.Bottle()
        self.outputBottle = yarp.Bottle()
        self.portNameList = []

        self.rpcConnected = False
        self.dataInConnected = False
        self.dataOutConnected = False
        self.collectionMethod = ''
        self.bufferSize = None

        self.falseCount = 0
        self.noDataCount = 0
        self.inputType = None
        self.outputType = None
        self.errorRate = 50
        self.dataList = []
        self.classificationList = []
        self.closeFlag = False
        self.instancePortName = ''
        self.labelPortName = ''
        self.verboseSetting = False
        self.exitFlag = False
        self.recordingFile = ''
        self.additionalInfoDict = dict()
        self.modelLoaded = False
        self.attentionMode = 'continue'
        self.baseLogFileName = 'interactionErrorLog'
        self.windowedMode = True
        self.modelRoot = None
        self.eventPort = None
        self.eventPortName = None
        self.classTimestamps = None
        self.probClassList = None
        self.recency = None
        self.useRecentClassTime = True
        self.drawLatent = False
        self.latentPlots = None
        self.my_mutex = thread.allocate_lock()

    def configure(self, rf):
        """
         Configure interactionSAMModel yarp module

        Args:
            rf: Yarp RF context input
            argv[1] : String containing data path.
            argv[2] : String containing model path.
            argv[3] : String containing config file path (from `config_path` parameter of samSupervisor config file).
            argv[4] : String driver name corresponding to a valid driver present in SAM_Drivers folder.
            argv[5] : String `'True'` or `'False'` to switch formatting of logging depending on whether interaction is
                      logging to a separate window or to the stdout of another process.
        Returns:
            Boolean indicating success or no success in initialising the yarp module
        """
        string_command = 'from SAM.SAM_Drivers import ' + sys.argv[4] + ' as Driver'
        exec string_command

        self.mm = [Driver()]
        self.dataPath = sys.argv[1]
        self.modelPath = sys.argv[2]
        self.driverName = sys.argv[4]
        self.configPath = sys.argv[3]
        self.windowedMode = sys.argv[5] == 'True'
        self.modelRoot = self.dataPath.split('/')[-1]

        file_i = 0
        logger_file_name = join(self.dataPath, self.baseLogFileName + '_' + str(file_i) + '.log')

        # check if file exists
        while os.path.isfile(logger_file_name) and os.path.getsize(logger_file_name) > 0:
            logger_file_name = join(self.dataPath, self.baseLogFileName + '_' + str(file_i) + '.log')
            file_i += 1

        if self.windowedMode:
            log_formatter = logging.Formatter("[%(levelname)s]  %(message)s")
        else:
            log_formatter = logging.Formatter("\033[31m%(asctime)s [%(name)-33s] [%(levelname)8s]  %(message)s\033[0m")

        root_logger = logging.getLogger('interaction ' + self.driverName)
        root_logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(logger_file_name)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)
        logging.root = root_logger

        off = 17
        logging.info('Arguments: ' + str(sys.argv))
        logging.info(string_command)
        logging.info('Using log' + str(logger_file_name))
        logging.info('-------------------')
        logging.info('Interaction Settings:')
        logging.info('Data Path: '.ljust(off) + str(self.dataPath))
        logging.info('Model Path: '.ljust(off) + str(self.modelPath))
        logging.info('Config Path: '.ljust(off) + str(self.configPath))
        logging.info('Driver: '.ljust(off) + str(self.driverName))
        logging.info('-------------------')
        logging.info('Configuring Interaction...')
        logging.info('')

        # parse settings from config file
        parser2 = SafeConfigParser()
        parser2.read(self.configPath)
        proposed_buffer = 5
        if self.modelRoot in parser2.sections():
            self.portNameList = parser2.items(self.dataPath.split('/')[-1])
            logging.info(str(self.portNameList))
            self.portsList = []
            for j in range(len(self.portNameList)):
                if self.portNameList[j][0] == 'rpcbase':
                    self.portsList.append(yarp.Port())
                    self.portsList[j].open(self.portNameList[j][1]+":i")
                    self.svPort = j
                    self.attach(self.portsList[j])
                elif self.portNameList[j][0] == 'visualise':
                    if self.portNameList[j][1] == "True":
                        self.drawLatent = True
                elif self.portNameList[j][0] == 'callsign':
                    # should check for repeated call signs by getting list from samSupervisor
                    self.callSignList = self.portNameList[j][1].split(',')
                elif self.portNameList[j][0] == 'latentmodelport':
                    self.portsList.append(yarp.BufferedPortImageRgb())
                    self.latentPort = j
                    ports = self.portNameList[j][1].split(',')
                    self.portsList[j].open(ports[0])
                    yarp.Network.connect(ports[0], ports[1])
                elif self.portNameList[j][0] == 'collectionmethod':
                    self.collectionMethod = self.portNameList[j][1].split(' ')[0]
                    try:
                        proposed_buffer = int(self.portNameList[j][1].split(' ')[1])
                    except ValueError:
                        logging.error('collectionMethod bufferSize is not an integer')
                        logging.error('Should be e.g: collectionMethod = buffered 3')
                        return False

                    if self.collectionMethod not in ['buffered', 'continuous', 'future_buffered']:
                        logging.error('collectionMethod should be set to buffered / continuous / future_buffered')
                        return False
                elif self.portNameList[j][0] == 'recency':
                        try:
                            self.recency = int(self.portNameList[j][1])
                        except ValueError:
                            logging.error('Recency value for ' + str(self.driverName) + ' is not an integer')
                            self.recency = 5
                else:
                    parts = self.portNameList[j][1].split(' ')
                    logging.info(parts)

                    if parts[1].lower() == 'imagergb':
                        self.portsList.append(yarp.BufferedPortImageRgb())
                        # self.portsList[j].open(parts[0])

                    elif parts[1].lower() == 'imagemono':
                        self.portsList.append(yarp.BufferedPortImageMono())
                        # self.portsList[j].open(parts[0])

                    elif parts[1].lower() == 'bottle':
                        self.portsList.append(yarp.BufferedPortBottle())
                        # self.portsList[j].open(parts[0])

                    else:
                        logging.error('Data type ' + str(parts[1]) + ' for ' +
                                      str(self.portNameList[j][0]) + ' unsupported')
                        return False
                    # mrd models with label/instance training will always have:
                    # 1 an input data line which is used when a label is requested
                    # 2 an output data line which is used when a generated instance is required
                    if parts[0][-1] == 'i':
                        self.labelPort = j
                        self.labelPortName = parts[0]
                    elif parts[0][-1] == 'o':
                        self.instancePort = j
                        self.instancePortName = parts[0]

            if self.collectionMethod == 'continuous':
                self.portsList.append(yarp.BufferedPortBottle())
                self.eventPort = len(self.portsList) - 1
                self.eventPortName = '/'.join(self.labelPortName.split('/')[:3])+'/event'
                self.portsList[self.eventPort].open(self.eventPortName)
                self.classTimestamps = []
                if self.recency is None:
                    logging.warning('No recency value specified for ' + self.driverName)
                    logging.warning('Setting value to default of 5 seconds')
                    self.recency = 5

            if self.svPort is None or self.labelPort is None or self.instancePort is None:
                logging.warning('Config file properties incorrect. Should look like this:')
                logging.warning('[Actions]')
                logging.warning('dataIn = /sam/actions/actionData:i Bottle')
                logging.warning('dataOut = /sam/actions/actionData:o Bottle')
                logging.warning('rpcBase = /sam/actions/rpc')
                logging.warning('callSign = ask_action_label, ask_action_instance')
                logging.warning('collectionMethod = buffered 3')

            # self.mm[0].configInteraction(self)
            self.inputType = self.portNameList[self.labelPort][1].split(' ')[1].lower()
            self.outputType = self.portNameList[self.labelPort][1].split(' ')[1].lower()
            self.dataList = []
            self.classificationList = []
            self.probClassList = []
            self.classTimestamps = []
            yarp.Network.init()

            self.mm = initialiseModels([self.dataPath, self.modelPath, self.driverName], 'update', 'interaction',
                                       drawLatent=self.drawLatent)
            self.modelLoaded = True
            if self.drawLatent:
                self.latentPlots = dict()
                self.latentPlots['ax'], self.latentPlots['dims'] = self.mm[0].SAMObject.visualise(plot_scales=True)
                self.send_latent(self.latentPlots['ax'])

            if self.mm[0].model_mode != 'temporal':
                self.bufferSize = proposed_buffer
            elif self.mm[0].model_mode == 'temporal':
                self.bufferSize = self.mm[0].temporalModelWindowSize

            self.portsList[self.labelPort].open(self.labelPortName)
            self.portsList[self.instancePort].open(self.instancePortName)
            # self.test()

            return True
        else:
            logging.error('Section ' + str(self.modelRoot) + ' not found in ' + str(self.configPath))
            return False

    def send_latent(self, latent_plots):
        print "entered saved latent"
        data = np.fromstring(latent_plots.figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(latent_plots.figure.canvas.get_width_height()[::-1] + (3,))
        print data.shape

        yarp_image = self.portsList[self.latentPort].prepare()
        yarp_image.resize(data.shape[1], data.shape[0])
        data = data.astype(np.uint8)
        yarp_image.setExternal(data, data.shape[1], data.shape[0])
        self.portsList[self.latentPort].write()

    def close(self):
        """
            Close Yarp module

            Args:

            Returns:
                Boolean indicating success or no success in closing the Yarp module
        """
        # close ports of loaded models
        logging.info('Exiting ...')
        for j in self.portsList:
            self.closePort(j)
        return False

    @timeout(3)
    def close_port(self, j):
        """
            Helper function to close ports with an enforced timeout of 3 seconds so the module doesn't hang.

            Args:
                j: Yarp Port

            Returns:
                None
        """
        j.interrupt()
        time.sleep(1)
        j.close()

    def respond(self, command, reply):
        """
            Respond to external requests

            Description:
                Available requests \n
                1) __heartbeat__      :  Sanity check request to make sure module is still alive. \n
                2) __information__    :  Utility request to pass in contextual information. \n
                3) __portNames__      :  Request to return the name of the currently open ports for samSupervisor to
                                         keep a list of open ports. \n
                4) __reload__         :  Request to reload model from disk. \n
                5) __toggleVerbose__  :  Switch logging to stdout on or off. \n
                6) __EXIT__           :  Abort and close the module. \n
                7) __ask_X_label__    :  Request a classification from the module. \n
                8) __ask_X_instance__ :  Request a generative output from the module. \n

            Args:
                command : Incoming Yarp bottle containing external request.
                reply : Outgoing Yarp bottle containing reply to processed request.

            Returns:
                Boolean indicating success or no success in responding to external requests.
        """
        # this method responds to samSupervisor commands
        reply.clear()
        action = command.get(0).asString()

        count = 0
        while not self.modelLoaded:
            count += 1
            time.sleep(0.5)
            if count == 10:
                break

        if self.modelLoaded:
            if action != 'heartbeat' or action != 'information':
                logging.info(action + ' received')
                logging.info('responding to ' + action + ' request')

            if action == "portNames":
                reply.addString('ack')
                reply.addString(self.labelPortName)
                reply.addString(self.instancePortName)
                if self.collectionMethod == 'continuous':
                    reply.addString(self.eventPortName)
            # -------------------------------------------------
            elif action == "reload":
                # send a message to the interaction model to check version of currently loaded model
                # and compare it with that stored on disk. If model on disk is more recent reload model
                # interaction model to return "model reloaded correctly" or "loaded model already up to date"
                logging.info("reloading model")
                try:
                    self.mm = initialiseModels([self.dataPath, self.modelPath, self.driverName],
                                               'update', 'interaction')
                    reply.addString('ack')
                except:
                    reply.addString('nack')
            # -------------------------------------------------
            elif action == "heartbeat":
                reply.addString('ack')
            # -------------------------------------------------
            elif action == "toggleVerbose":
                self.verboseSetting = not self.verboseSetting
                reply.addString('ack')
            # -------------------------------------------------
            # elif action == "attention":
            #     self.attentionMode = command.get(1).asString()
            #     reply.addString('ack')
            # -------------------------------------------------
            elif action == "information":
                if command.size() < 3:
                    reply.addString('nack')
                else:
                    try:
                        self.additionalInfoDict[command.get(1).asString()] = command.get(2).asString()
                        reply.addString('ack')
                    except:
                        reply.addString('nack')
                    logging.info(self.additionalInfoDict)
            # -------------------------------------------------
            elif action == "EXIT":
                reply.addString('ack')
                self.close()
            # -------------------------------------------------
            elif action in self.callSignList:
                logging.info('call sign command recognized')
                if 'label' in action:
                    self.classify_instance(reply)
                elif 'instance' in action:
                    self.generate_instance(reply, command.get(1).asString())
            # -------------------------------------------------
            else:
                reply.addString("nack")
                reply.addString("Command not recognized")
        else:
            reply.addString("nack")
            reply.addString("Model not loaded")

        return True

    def classify_instance(self, reply):
        """
            Classify a live collected data instance

            Description:
                This method responds to an `ask_x_label` request sent via the rpc port of the module. \n
                In the case of __collectionMethod__ = `buffered`, the data currently in the buffer is sent to
                                                       processLiveData() method for the current driver which returns a
                                                       classification label that is embedded in reply.\n
                In the case of __collectionMethod__ = `future_buffered`, this method reads incoming frames from the
                                                      `dataIn` port until the collection buffer is full at which point
                                                      it calls processLiveData() to get a classification label.\n
                In the case of __collectionMethod__ = `continuous`, this model returns the most recent label in the
                                                       FILO buffer containing classification labels.\n

            Args:
                reply : Outgoing Yarp bottle containing classification label.

            Returns:
                None
        """
        if self.portsList[self.labelPort].getInputCount() > 0:
            if self.verboseSetting:
                logging.info('-------------------------------------')
            data_list = None
            if self.collectionMethod == 'buffered':
                if self.modelLoaded:
                    logging.debug('going in process live')
                    if self.drawLatent:
                        self.latentPlots['ax'], _ = self.mm[0].SAMObject.visualise(plot_scales=True)
                    this_class, prob_class, data_list = \
                        self.mm[0].processLiveData(self.dataList, self.mm, verbose=self.verboseSetting,
                                                   additionalData=self.additionalInfoDict,
                                                   visualiseInfo=self.latentPlots)
                    if self.drawLatent:
                        self.send_latent(self.latentPlots['ax1'][0].axes)
                    logging.debug('exited process live')
                else:
                    this_class = None
                logging.debug(this_class)
                logging.debug('object thisclass' + str(this_class is None))
                logging.debug('object datalist' + str(data_list is None))
                logging.debug('string thisclass' + str(this_class == 'None'))
                logging.debug('string datalist' + str(data_list == 'None'))
                if this_class is None or data_list is None:
                    logging.debug('None reply')
                    reply.addString('nack')
                else:
                    logging.debug('correct reply')
                    reply.addString('ack')
                    reply.addString(this_class)
                logging.debug('finish reply')
                # reply.addDouble(probClass)
            # -------------------------------------------------
            elif self.collectionMethod == 'continuous':
                # mutex lock classificationList
                self.my_mutex.acquire()
                logging.debug(self.classificationList)
                logging.debug(self.classTimestamps)
                # check last n seconds
                if len(self.classificationList) > 0:
                    if self.useRecentClassTime:
                        min_t = self.classTimestamps[-1] - self.recency
                    else:
                        min_t = time.time() - self.recency

                    logging.debug('min_t ' + str(min_t))
                    logging.debug('recency ' + str(self.recency))
                    for index, value in enumerate(self.classTimestamps):
                        logging.debug(str(index) + ' ' + str(value) + ' ' + str(value > min_t))
                    valid_list = [index for index, value in enumerate(self.classTimestamps) if value > min_t]
                    logging.debug('validList ' + str(valid_list))
                    min_idx = min(valid_list)
                    logging.debug('min_idx ' + str(min_idx))
                    valid_class_list = self.classificationList[min_idx:]
                    valid_prob_list = self.probClassList[min_idx:]
                    logging.debug('validClassList ' + str(valid_class_list))
                    logging.debug('classify classList' + str(self.classificationList))
                    logging.debug('classify probclassList' + str(self.probClassList))
                    logging.debug('classify classTimeStamps' + str(self.classTimestamps))

                    if len(valid_class_list) > 0:

                        # combine all classifications
                        if len(valid_class_list) == 1:
                            logging.debug('validClassList is of len 1')
                            decision = valid_class_list[0]
                        else:
                            logging.debug('validClassList is of len ' + str(len(valid_class_list)))
                            set_class = list(set(valid_class_list))
                            logging.debug('set_class ' + str(set_class))
                            if len(set_class) == len(valid_class_list):
                                logging.debug('len set_class = len validClassList' + str(len(set_class)) + ' ' +
                                              str(len(valid_class_list)))
                                decision = valid_class_list[valid_prob_list.index(max(valid_prob_list))]
                            else:
                                dict_results = dict()
                                for m in set_class:
                                    logging.debug('currentM ' + str(m))
                                    idx_m = [idx for idx, name in enumerate(valid_class_list) if name == m]
                                    logging.debug('idx ' + str(idx_m))
                                    prob_vals = itemgetter(*idx_m)(valid_prob_list)
                                    logging.debug('probs ' + str(prob_vals))
                                    try:
                                        prob_sum = sum(prob_vals)
                                    except TypeError:
                                        prob_sum = prob_vals
                                    logging.debug('sum ' + str(prob_sum))
                                    dict_results[m] = prob_sum
                                    logging.debug('')
                                logging.debug('dict_results ' + str(dict_results))
                                max_dict_prob = max(dict_results.values())
                                logging.debug('max_dict_prob = ' + str(max_dict_prob))
                                decisions = [key for key in dict_results.keys() if dict_results[key] == max_dict_prob]
                                logging.info('Decision: ' + str(decisions))
                                logging.info('We have resolution')
                                decision = ' and '.join(decisions)
                        logging.info('Decision: ' + decision)

                        reply.addString('ack')
                        reply.addString(decision)
                        # reply.addString(validClassList[-1])
                        # self.classificationList.pop(-1)

                        # remove validclassList from self.classificationList / probClassList / classTimeStamps
                        self.classificationList = self.classificationList[:min_idx]
                        self.probClassList = self.probClassList[:min_idx]
                        self.classTimestamps = self.classTimestamps[:min_idx]

                    else:
                        logging.info('No valid classifications')
                        reply.addString('nack')
                    logging.debug('replying ' + reply.toString())
                    logging.debug('classify classList' + str(self.classificationList))
                    logging.debug('classify probclassList' + str(self.probClassList))
                    logging.debug('classify classTimeStamps' + str(self.classTimestamps))
                else:
                    logging.info('No classifications yet')
                    reply.addString('nack')
                self.my_mutex.release()
            # -------------------------------------------------
            elif self.collectionMethod == 'future_buffered':
                self.dataList = []
                for j in range(self.bufferSize):
                    self.dataList.append(self.read_frame())
                if self.modelLoaded:
                    if self.drawLatent:
                        self.latentPlots['ax'], _ = self.mm[0].SAMObject.visualise(plot_scales=True)
                    this_class, prob_class, data_list = \
                        self.mm[0].processLiveData(self.dataList, self.mm, verbose=self.verboseSetting,
                                                   additionalData=self.additionalInfoDict,
                                                   visualiseInfo=self.latentPlots)
                    if self.drawLatent:
                        self.send_latent(self.latentPlots['ax1'][0].axes)
                        # self.latentPlots['ax1'].pop(0).remove()
                else:
                    this_class = None

                if this_class is None or data_list is None:
                    logging.info('thisClass or dataList returned None')
                    logging.debug('thisClass: ' + str(type(this_class)) + ' ' + str(this_class))
                    logging.debug('dataList: ' + str(type(data_list)) + ' ' + str(data_list))
                    reply.addString('nack')
                else:
                    reply.addString('ack')
                    reply.addString(this_class)
                    # reply.addDouble(probClass)
        else:
            reply.addString('nack')
            reply.addString('No input connections to ' + str(self.portsList[self.labelPort].getName()))
        logging.info('--------------------------------------')

    def generate_instance(self, reply, instance_name):
        """Responds to an ask_X_instance request

        Description:
            Implements the logic for responding to an `ask_X_instance` rpc request for __instanceName__.
            This method responds with an `ack` or `nack` on the rpc port indicating success of memory generation and
            outputs the generated instance returned by recall_from_label on the `dataOut` port.

        Args:
            reply : Yarp Bottle to embed the rpc response.
            instance_name : Name of class to generate.

        Returns:
            None
        """
        if self.portsList[self.instancePort].getOutputCount() != 0:
            if instance_name in self.mm[0].textLabels:
                instance = self.recall_from_label(instance_name)
                # send generated instance to driver where it is converted into the proper format
                formatted_data = self.mm[0].formatGeneratedData(instance)
                # check formatted_data is of correct data type
                if str(type(self.portsList[self.instancePort])).split('\'')[1].split('Port')[1] \
                        in str(type(formatted_data)):
                    try:
                        img = self.portsList[self.instancePort].prepare()
                        img.copy(formatted_data)
                        self.portsList[self.instancePort].write()
                        reply.addString('ack')
                        reply.addString('Generated instance of ' + instance_name + ' as ' +
                                        str(type(formatted_data)))
                    except:
                        reply.addString('nack')
                        reply.addString('Failed to write ' + instance_name + ' as ' +
                                        str(type(self.portsList[self.instancePort])))
                else:
                    reply.addString('nack')
                    reply.addString('Output of ' + self.driverName + '.formatGeneratedData is of type: ' +
                                    str(type(formatted_data)) + '. Should be type: ' +
                                    str(type(self.portsList[self.instancePort])))
            else:
                reply.addString('nack')
                reply.addString('Instance name not found. Available instance names are: ' + str(self.mm[0].textLabels))
        else:
            reply.addString('nack')
            reply.addString('No outgoing connections on ' + str(self.portsList[self.instancePort].getName()))

    def recall_from_label(self, label):
        """
            Generates instance based on label.

            Args:
                label : String containing the class label for the requested generated instance.

            Returns:
               Serialised vector representing the generated instance.
        """
        ind = self.mm[0].textLabels.index(label)
        if len(self.mm) > 1:
            inds_to_choose_from = self.mm[ind + 1].SAMObject.model.textLabelPts[ind]
            chosen_ind = np.random.choice(inds_to_choose_from, 1)
            yrecall = self.mm[ind + 1].SAMObject.recall(chosen_ind)
        else:
            inds_to_choose_from = self.mm[0].SAMObject.model.textLabelPts[ind]
            chosen_ind = np.random.choice(inds_to_choose_from, 1)
            yrecall = self.mm[0].SAMObject.recall(chosen_ind)

        return yrecall

    def interruptModule(self):
        """
        Module interrupt logic.

        Returns : Boolean indicating success of logic or not.
        """
        return True

    def getPeriod(self):
        """
           Module refresh rate.

           Returns : The period of the module in seconds.
        """
        return 0.1

    def updateModule(self):
        """
            Logic to execute every getPeriod() seconds.

            Description:
                This function makes sure important ports are connected. Priority 1 is the rpc port.
                Priority 2 is the data in port. If both are connected this function triggers collect_data().

            Returns: Boolean indicating success of logic or not.
        """
        out = self.portsList[self.svPort].getOutputCount() + self.portsList[self.svPort].getInputCount()
        if out != 0:
            if not self.rpcConnected:
                logging.info("Connection received")
                logging.info('\n')
                logging.info('-------------------------------------')
                self.rpcConnected = True
                self.falseCount = 0
            else:
                self.dataInConnected = self.portsList[self.labelPort].getInputCount() > 0
                if self.dataInConnected:
                    self.collect_data()
                else:
                    self.noDataCount += 1
                    if self.noDataCount == self.errorRate:
                        self.noDataCount = 0
                        logging.info('No data in connection. Waiting for ' +
                                     self.portNameList[self.labelPort][1] + ' to receive a connection')
        else:
            self.rpcConnected = False
            self.falseCount += 1
            if self.falseCount == self.errorRate:
                self.falseCount = 0
                logging.info('Waiting for ' + self.portNameList[self.svPort][1] +
                             ' to receive a connection')

        time.sleep(0.05)
        return True

    def read_frame(self):
        """
            Logic to read an available data frame.

            Description:
                This function first checks the required data type of the frame to be received and instantiates the
                required yarp data object. This function then subsequently reads in the latest frame from the
                __dataIn__ port which it returns. Return is `None` if the data type is not recognised. This is
                currently a limitation because `ImageRgb`, `ImageMono` and `Bottle` are so far the only supported
                bottle types. This can be easily extended in this section by adding more cases.

            Returns: Boolean indicating success of logic or not.
        """
        if self.inputType == 'imagergb':
            frame = yarp.ImageRgb()
        elif self.inputType == 'imagemono':
            frame = yarp.ImageMono()
        elif self.inputType == 'bottle':
            frame = yarp.Bottle()
        else:
            return None

        frame_read = self.portsList[self.labelPort].read(True)

        if self.inputType == 'bottle':
            frame.fromString(frame_read.toString())
        elif 'image' in self.inputType:
            frame.copy(frame_read)

        return frame

    def collect_data(self):
        """Collect data function

        Description:
            This function implements three types of data collection procedures: \n

            1) __buffered__ : Collects data in a fixed length FIFO buffer of past frames.
                              This buffer is read by classify_instance(). \n
            2) __future_buffered__ : No operation. \n
            3) __continuous__ : Collect data until a buffer of length windowSize is full and then perform a
                                classification on this data. The classification is then stored in a buffer with is read
                                by classify_instance(). \n

            Returns:
                None
        """
        self.noDataCount = 0

        if self.collectionMethod == 'buffered':
            frame = self.read_frame()
            # append frame to buffer
            if len(self.dataList) == self.bufferSize:
                # FIFO buffer first item in list most recent
                self.dataList.pop(0)
                self.dataList.append(frame)
            else:
                self.dataList.append(frame)
        # -------------------------------------------------
        elif self.collectionMethod == 'continuous' and self.attentionMode == 'continue':
            # read frame of data
            frame = self.read_frame()
            # append frame to data_list

            if self.dataList is None:
                self.dataList = []

            self.dataList.append(frame)
            # process list of frames for a classification
            data_list = []
            prob_class = None
            if self.modelLoaded:
                if self.drawLatent:
                    self.latentPlots['ax'], _ = self.mm[0].SAMObject.visualise(plot_scales=True)
                this_class, prob_class, data_list = \
                    self.mm[0].processLiveData(self.dataList, self.mm, verbose=self.verboseSetting,
                                               additionalData=self.additionalInfoDict,
                                               visualiseInfo=self.latentPlots)
                if self.drawLatent:
                    self.send_latent(self.latentPlots['ax1'][0].axes)
            else:
                this_class = None
            # if proper classification
            if this_class is not None:
                # empty data_list

                # mutex data_list lock
                self.my_mutex.acquire()
                self.dataList = data_list
                # mutex data_list release

                if this_class != 'None':
                    t_stamp = time.time()
                    event_bottle = self.portsList[self.eventPort].prepare()
                    event_bottle.clear()
                    event_bottle.addString('ack')
                    self.portsList[self.eventPort].write()
                    # add classification to classificationList to be retrieved during respond method

                    # mutex classificationList lock

                    # Time based method
                    logging.info('classList len: ' + str(len(self.classificationList)))
                    logging.debug('thisclass ' + str(this_class))
                    self.classificationList = self.classificationList + this_class
                    logging.debug('classificationList ' + str(self.classificationList))
                    self.probClassList = self.probClassList + prob_class
                    logging.debug('prob_class ' + str(self.probClassList))
                    self.classTimestamps = self.classTimestamps + [t_stamp]*len(this_class)
                    logging.debug('self.classTimestamps ' + str(self.classTimestamps))
                    # remove timestamps older than memory duration (self.bufferSize in seconds)
                    logging.debug('last time stamp: ' + str(self.classTimestamps[-1]))
                    min_mem_t = self.classTimestamps[-1] - self.bufferSize
                    logging.debug('min_mem_t ' + str(min_mem_t))
                    good_idxs = [idx for idx, timeVal in enumerate(self.classTimestamps) if timeVal > min_mem_t]
                    logging.debug('good_idxs ' + str(good_idxs))
                    min_idx = min(good_idxs)
                    self.classificationList = self.classificationList[min_idx:]
                    self.probClassList = self.probClassList[min_idx:]
                    self.classTimestamps = self.classTimestamps[min_idx:]

                    logging.debug('classificationList ' + str(self.classificationList))
                    logging.debug('prob_class ' + str(self.probClassList))
                    logging.debug('self.classTimestamps ' + str(self.classTimestamps))

                    # Old method
                    # if len(self.classificationList) == self.bufferSize:
                    #     # FIFO buffer first item in list is oldest
                    #     self.classificationList.pop(0)
                    #     self.classTimestamps.pop(0)
                    #     self.classificationList.append(this_class)
                    #     self.classTimestamps.append(t_stamp)
                    # else:
                    #     self.classificationList.append(this_class)
                    #     self.classTimestamps.append(t_stamp)

                # mutex release
                self.my_mutex.release()
        # -------------------------------------------------
        elif self.collectionMethod == 'future_buffered':
            pass

    def test(self):
        """
            Utility function to test data collection procedures for debugging purposes.
        """
        count = 0
        if self.collectionMethod == 'continuous':
            classify_block = self.mm[0].paramsDict['windowSize']
        elif self.collectionMethod == 'buffered':
            classify_block = self.bufferSize
        else:
            classify_block = 10

        while True:
            out = (self.portsList[self.svPort].getOutputCount() + self.portsList[self.svPort].getInputCount()) > 0
            data_in_connected = self.portsList[self.labelPort].getInputCount() > 0

            if out and data_in_connected:
                if self.collectionMethod == 'future_buffered':
                    reply = yarp.Bottle()
                    self.classify_instance(reply)
                    logging.info(reply.toString())
                elif self.collectionMethod == 'continuous':
                    self.collect_data()
                    count += 1
                    if count == classify_block:
                        count = 0
                        reply = yarp.Bottle()
                        self.classify_instance(reply)
                        logging.info('CLASSIFICATION: ' + reply.toString())

                # self.dataList = []
                # for j in range(self.bufferSize):
                #     self.dataList.append(self.read_frame())

                # if thisClass is None:
                #     logging.info('None')
                # else:
                #     logging.info(thisClass, ' ', likelihood)

            time.sleep(0.05)


def exception_hook(exc_type, exc_value, exc_traceback):
    """Callback function to record any errors that occur in the log files.

        Documentation:
            Substitutes the standard python exception_hook with one that records the error into a log file.
            Can only work if interactionSAMModel.py is called from python and not ipython because ipython
            overrides this substitution.
        Args:
            exc_type: Exception Type.
            exc_value: Exception Value.
            exc_traceback: Exception Traceback.

        Returns:
            None
    """
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = exception_hook

if __name__ == '__main__':
    plt.ion()
    yarp.Network.init()
    mod = interactionSAMModel()
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    # rf.setDefaultContext("samSupervisor")
    # rf.setDefaultConfigFile("default.ini")
    rf.configure(sys.argv)

    mod.runModule(rf)
