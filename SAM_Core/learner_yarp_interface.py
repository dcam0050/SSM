#!/usr/bin/env python

import sys
import time
from SAM.SAM_Core.SAM_utils import timeout
import warnings
from os.path import join
import os
import numpy as np
import yarp
import pickle
import logging
import thread
from neurolearning import nlearn
import itertools
warnings.simplefilter("ignore")
np.set_printoptions(precision=2)


class node_info():
    def __init__(self, name=None, values=[], edges_list=[], type=''):
        self.name = name
        self.values = values
        self.edges_list = edges_list
        self.type = type
        # type == '' -> learned
        # type == 'in' -> input
        # type == 'out' -> output

    def print_node(self):
        return str(self.name) + ' ' + str(self.values) + ' ' + str(self.edges_list) + ' ' + str(self.type)


class Learner(yarp.RFModule):
    """Memory visualisation function
    """
    # def __init__(self, sysargs, additional_loggers=None):
    def __init__(self, additional_loggers=None):
        """
        Initialisation of the interaction function
        """
        yarp.RFModule.__init__(self)
        yarp.Network.init()
        self.ports_list = []
        self.sv_port = 0
        self.pause = False
        self.windowed_mode = True
        self.sv_port_name = "/SAM/learner/rpc"
        self.t = 0
        self.cmd = yarp.Bottle()
        self.rep = yarp.Bottle()
        self.received_data = None
        self.connected_to_visualiser = False
        self.pauseVal = 0.1
        self.node_data = dict()
        self.edge_data = np.zeros((0, 2))
        self.prev_data = None
        self.learner_initialised = False
        self.processing_underway = False
        self.initialising_graph = False
        # self.args = sysargs
        self.my_mutex = thread.allocate_lock()
        self.vis_graph_parameters = dict()
        self.vis_graph_parameters['num_inputs'] = 0
        self.vis_graph_parameters['num_outputs'] = 0
        self.transmit_output = False

        file_i = 0
        logger_f_name = join(os.curdir, 'learner_' + str(file_i) + '.log')

        # check if file exists
        while os.path.isfile(logger_f_name) and os.path.getsize(logger_f_name) > 0:
            logger_f_name = join(os.curdir, 'visualiser_' + str(file_i) + '.log')
            file_i += 1

        if self.windowed_mode:
            log_formatter = logging.Formatter("[%(levelname)s]  %(message)s")
        else:
            log_formatter = logging.Formatter("\033[31m%(asctime)s [%(name)-33s] [%(levelname)8s]  %(message)s\033[0m")

        root_logger = logging.getLogger('visualiser')
        root_logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(logger_f_name)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

        if additional_loggers is not None:
            for j in additional_loggers:
                root_logger.addHandler(j)

        logging.root = root_logger
        logging.debug("Test logging")
        self.received_data = None

    def configure(self, rf):
        """
         Configure interactionSAMModel yarp module

        Args:
            rf: Yarp RF context input

        Returns:
            Boolean indicating success or no success in initialising the yarp module
        """

        self.ports_list.append(yarp.Port())
        self.ports_list[self.sv_port].open(self.sv_port_name)
        self.attach(self.ports_list[self.sv_port])

        return True

    def close(self):
        """
            Close Yarp module

            Args:

            Returns:
                Boolean indicating success or no success in closing the Yarp module
        """
        # close ports of loaded models
        logging.info('Exiting ...')
        for j in self.ports_list:
            self.close_port(j)
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
                2) __EXIT__           :  Abort and close the module. \n

            Args:
                command : Incoming Yarp bottle containing external request.
                reply : Outgoing Yarp bottle containing reply to processed request.

            Returns:
                Boolean indicating success or no success in responding to external requests.
        """
        # this method responds to samSupervisor commands
        reply.clear()
        action = command.get(0).asString()

        logging.info(action + ' received')
        logging.info('responding to ' + action + ' request')

        if action == "heartbeat":
            reply.addString('ack')
        # -------------------------------------------------
        elif action == "EXIT":
            reply.addString('ack')
            self.close()
        # -------------------------------------------------
        elif action == "pause":
            reply.addString('ack')
            if self.pause:
                self.pause = False
            else:
                self.pause = True
        # -------------------------------------------------
        elif action == "init_in_out":
            # command[1] int num inputs
            # command[2] int num outputs
            # command[3] list input names
            # command[4] list output names
            try:
                logging.info(command.toString())
                num_inputs = int(command.get(1).asString())
                num_outputs = int(command.get(2).asString())
                input_names = command.get(3).asString()[1:-1].replace("'", '').split(', ')
                output_names = command.get(4).asString()[1:-1].replace("'", '').split(', ')
                self.init_in_out(num_inputs, num_outputs, input_names, output_names)
                reply.addString("ack")
            except Exception as e:
                logging.warning(str(e))
                reply.addString("nack")
                reply.addString(str(e))
                reply.addString(
                    "E.g 'init_in_out <num_inputs> <num_outputs> <list_input_names> <list_output_names>'")
        # -------------------------------------------------
        elif action == "process_data":
            if command.size() < 2:
                reply.addString("nack")
                reply.addString("data not included with command")
            else:
                self.process_data(command, reply)
        # -------------------------------------------------
        elif action == "get_node_data":
            if command.size() < 2:
                reply.addString("nack")
            else:
                curr_node = self.node_data[str(command.get(1).asInt())]
                reply.addString("ack")
                reply.addString(curr_node.name)
                reply.addString(str(curr_node.values))
                reply.addString(str(curr_node.edges_list))
        # -------------------------------------------------
        elif action == "trigger_graph_init":
            self.trigger_graph_init(command, reply)
        else:
            reply.addString("nack")

        return True

    def trigger_graph_init(self, command, reply):
        self.pause = True
        if self.learner_initialised:
            logging.info("learner initialised")
            self.cmd.clear()
            self.rep.clear()
            self.vis_graph_parameters['npts'] = len(self.node_data.keys())
            self.vis_graph_parameters['edges'] = self.edge_data
            graph_pickle = pickle.dumps(self.vis_graph_parameters)
            self.cmd.addString("init_graph")
            self.cmd.addString(graph_pickle)
            self.ports_list[self.sv_port].write(self.cmd, self.rep)
            logging.info(self.cmd.toString() + " reply: " + self.rep.toString())
            if self.rep.toString() == "ack":
                reply.addString("ack")
                self.connected_to_visualiser = True
            else:
                reply.addString("nack")
        else:
            logging.info("learner not yet initialised")
            reply.addString("nack")
            reply.addString("Learner not yet initialised")
        self.pause = False

    def process_data(self, command, reply):
        if self.learner_initialised:
            with self.my_mutex:
                try:
                    unpickled = pickle.loads(command.get(1).asString())
                    logging.info("received data with type {} and shape {}".format(type(unpickled),
                                                                                  str(unpickled.shape)))

                    if type(unpickled) is np.ndarray and unpickled.shape[0] > 0 \
                            and unpickled.shape[1] == self.vis_graph_parameters['num_inputs']:

                        if self.received_data is None:
                            self.received_data = unpickled
                        else:
                            self.received_data = np.vstack((self.received_data, unpickled))
                        reply.addString("ack")
                        self.processing_underway = True
                    else:
                        raise TypeError("type must be np.ndarray, number rows > 0 and number columns = {}".
                                        format(len(self.vis_graph_parameters['num_inputs'])))
                except Exception as e:
                    logging.warning(str(e))
                    reply.addString("nack")
                    reply.addString(str(e))
        else:
            reply.addString("nack")
            reply.addString("Learner not yet initialised")

    def init_in_out(self, num_inputs, num_outputs, input_names, output_names, transfer=True):
        npts = num_inputs + num_outputs
        self.vis_graph_parameters['num_inputs'] = num_inputs
        self.vis_graph_parameters['num_outputs'] = num_outputs

        stat_nodes_names = []
        if len(input_names) + len(output_names) < npts:
            logging.info("Lenght of input names {0}+ output names {1} less than number of points {2}".
                         format(len(num_inputs), len(num_outputs), npts))
            possible_letters = list(map(chr, range(97, 123)))
            mults = npts // len(possible_letters)
            rem = npts % len(possible_letters)
            for j in range(mults):
                if j == 0:
                    stat_nodes_names += possible_letters
                else:
                    stat_nodes_names += [possible_letters[j-1]+k for k in possible_letters]

            stat_nodes_names += [possible_letters[mults-1] + k for k in possible_letters[:rem]]
            input_names = stat_nodes_names[:num_inputs]
            output_names = stat_nodes_names[num_inputs:]

        if not transfer or len(self.node_data.keys()) == 0:
            self.node_data = dict()
            self.edge_data = np.empty((0, 2))
            self.prev_data = None

            for i in range(num_inputs):
                self.node_data[str(i)] = node_info(name=input_names[i], type='in')

            for i in range(num_outputs):
                self.node_data[str(i+num_inputs)] = node_info(name=output_names[i], type='out')
            rep = yarp.Bottle()
            cmd = yarp.Bottle()
            self.trigger_graph_init(cmd, rep)

        for k in self.node_data.keys():
            print "node", k, ":", self.node_data[k].print_node()

        self.learner_initialised = True

    def interruptModule(self):
        """
        Module interrupt logic.

        Returns : Boolean indicating success of logic or not.
        """
        self.close()
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
                This function makes sure important ports are connected. Priority 1 is the rpc port. Priority 2
                is the data in port. If both are connected this function triggers collectData().

            Returns: Boolean indicating success of logic or not.
        """
        logging.info("{}, {}".format(self.received_data is None, self.learner_initialised))
        if self.received_data is not None and self.learner_initialised:
            if not self.pause and self.received_data.shape[0] > 0:
                logging.info(str(self.received_data.shape[0]))
                currData = self.received_data[0, :]
                new_nodes, new_edges, all_edges = self.neurolearn(currData)
                if len(all_edges) > 0:
                    self.edge_data = np.vstack((self.edge_data, all_edges))
                with self.my_mutex:
                    if self.received_data.shape[0] == 1:
                        self.received_data = None
                        self.processing_underway = False
                    else:
                        self.received_data = self.received_data[1:]

                if self.connected_to_visualiser:
                    for nn in new_nodes:
                        self.cmd.clear()
                        self.rep.clear()
                        self.cmd.addString("add_node")
                        self.cmd.addInt(nn)
                        self.ports_list[self.sv_port].write(self.cmd, self.rep)
                    for ne in new_edges:
                        self.cmd.clear()
                        self.rep.clear()
                        self.cmd.addString("add_edge")
                        self.cmd.addInt(int(ne[0]))
                        self.cmd.addInt(int(ne[1]))
                        self.ports_list[self.sv_port].write(self.cmd, self.rep)
                    logging.info(self.cmd.toString() + " reply: " + self.rep.toString())

        time.sleep(self.pauseVal)
        return True

    def neurolearn(self, currDataSTR):
        new_nodes_list = []
        all_edges_list = []
        new_edges_list = []

        for i, j in enumerate(currDataSTR):
            curr_node_data = self.node_data[str(i)]
            if j not in curr_node_data.values:
                new_node_idx = len(self.node_data.keys())
                curr_node_data.values = curr_node_data.values + [j]
                curr_node_data.edges_list = curr_node_data.edges_list + [(i, new_node_idx)]
                new_edges_list += [(i, new_node_idx)]
                self.node_data[str(new_node_idx)] = node_info(edges_list=[(i, new_node_idx)])
                new_nodes_list.append(i)

        currData = np.array(map(float, currDataSTR))
        if self.prev_data is not None:
            diffInputs = currData - self.prev_data
            # print map(float, currData), self.prev_data, diffInputs
            diff_idx = np.where(diffInputs != 0)[0]
            edge_nodes = []
            if len(diff_idx) > 1:
                # print "ids", diff_idx
                for j in diff_idx:
                    # find value of index j from currData
                    # print self.node_data[str(j)].values
                    # find index of value in values list
                    nod = map(float, self.node_data[str(j)].values).index(currData[j])
                    # find corresponding edge pair from edges_list
                    curredgepair = list(self.node_data[str(j)].edges_list[nod])
                    # remove current index from edge pair
                    curredgepair.remove(j)
                    edge_nodes += curredgepair
            if len(edge_nodes) > 1:
                if len(edge_nodes) > 2:
                    edge_combinations = list(itertools.combinations(edge_nodes, 2))
                    new_edges_list += edge_combinations
                else:
                    new_edges_list += [tuple(edge_nodes)]

        # print new_nodes_list, new_edges_list
        self.prev_data = currData
        return new_nodes_list, np.asarray(new_edges_list, dtype=np.int32), \
               np.asarray(all_edges_list+new_edges_list, dtype=np.int32)


def exception_hook(exc_type, exc_value, exc_traceback):
    """Callback function to record any errors that occur in the log files.

        Documentation:
            Substitutes the standard python exception_hook with one that records the error into a log file.
            Can only work if interactionSAMModel.py is called from python and not ipython
            because ipython overrides this substitution.
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
    yarp.Network.init()

    mod = Learner()
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.configure(sys.argv)

    mod.runModule(rf)
