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
import argparse
import csv
import logging
from neurolearning import nlearn
warnings.simplefilter("ignore")
np.set_printoptions(precision=2)


class node_info():
    def __init__(self, name=None, values=[], edges_list=[]):
        self.name = name
        self.values = values
        self.edges_list = edges_list


class Learner(yarp.RFModule):
    """Memory visualisation function
    """
    def __init__(self):
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
        self.vis_port_name = "/SAM/visualiser/rpc"
        self.t = 0
        # self.stat_nodes = None
        # self.edge_color = None
        # self.edges = None
        # self.stat_node_spacing = None
        # self.pos = None
        # self.node_color = None
        # self.parameters = dict()
        self.cmd = yarp.Bottle()
        self.rep = yarp.Bottle()
        self.read_data = None
        self.init_complete = False
        self.pauseVal = 0.5
        self.node_data = dict()
        self.edge_data = dict()
        self.data = dict()
        self.prev_data = None

    def configure(self, rf):
        """
         Configure interactionSAMModel yarp module

        Args:
            rf: Yarp RF context input

        Returns:
            Boolean indicating success or no success in initialising the yarp module
        """

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

        # console_handler = logging.StreamHandler()
        # console_handler.setFormatter(log_formatter)
        # root_logger.addHandler(console_handler)
        logging.root = root_logger

        self.ports_list.append(yarp.Port())
        self.ports_list[self.sv_port].open(self.sv_port_name)
        self.attach(self.ports_list[self.sv_port])

        parser = argparse.ArgumentParser()
        parser.add_argument("--num_inputs", type=int)
        parser.add_argument("--num_outputs", type=int)
        parser.add_argument("--dimensions", type=int)
        parser.add_argument("--datafile")
        args = parser.parse_args()

        if args.datafile:
            # load data from file
            with open(args.datafile, "rb") as f:
                reader = csv.reader(f)
                interface = reader.next()
                self.read_data = np.array([row for row in reader])
            self.data['stat_nodes_names'] = interface
            npts = len(interface)
        else:
            npts = args.num_inputs + args.num_outputs

        dimensions = args.dimensions if args.dimensions else 3

        self.data['parameters'] = dict()
        self.data['parameters']['nodeSize'] = 30
        self.data['parameters']['scaling'] = False
        self.data['parameters']['stat_spacing'] = 1
        self.data['parameters']['symbol'] = 'o'
        self.data['parameters']['antialias'] = True
        self.data['parameters']['edge_width'] = 1
        self.data['parameters']['edge_render_method'] = 'gl'

        self.data['stat_node_spacing'] = 2
        self.data['stat_nodes'] = np.arange(0, npts, 1)
        self.data['pos'] = np.empty((npts, dimensions), dtype='float32')
        self.data['node_color'] = np.empty((npts, 3), dtype='float32')
        self.data['pos'] = np.random.normal(size=self.data['pos'].shape, scale=4.)
        self.data['node_color'][:] = np.array([(1, 0, 0),
                                              (1, 1, 0),
                                              (1, 0, 1),
                                              (0, 1, 1)])

        # initialise position of static nodes in a line
        self.data['pos'][self.data['stat_nodes'], 1:] = 0
        self.data['pos'][self.data['stat_nodes'], 0] = \
            np.arange(0, len(self.data['stat_nodes']) * self.data['stat_node_spacing'], self.data['stat_node_spacing'])

        self.data['edges'] = np.array([(0, 0)])
        self.data['edge_color'] = np.ones((self.data['pos'].shape[0], 4))
        for i, j in enumerate(self.data['stat_nodes_names']):
            self.node_data[str(i)] = node_info(name=j)
        for k in self.node_data.keys():
            print "node", k, ":", self.node_data[k].name, self.node_data[k].values, self.node_data[k].edges_list

        # for t in range(self.read_data.shape[0]):
        #     currData = self.read_data[t, :]
        #     new_nodes, new_edges = self.neurolearn(currData)

        self.wait_connection()

        return True

    def wait_connection(self):
        not_connected = True
        while not_connected:
            yarp.Network.connect(self.sv_port_name, self.vis_port_name)
            logging.info("Waiting for connection with visualiser at " + self.vis_port_name)
            time.sleep(1)
            if self.ports_list[self.sv_port].getOutputCount() != 0:
                not_connected = False

        self.cmd.clear()
        self.rep.clear()
        parameter_json = pickle.dumps(self.data)

        self.cmd.addString("check_graph_start")
        self.ports_list[self.sv_port].write(self.cmd, self.rep)
        if self.rep.toString() == "nack":
            self.cmd.clear()
            self.cmd.addString("init_graph_parameters")
            self.cmd.addString(parameter_json)
            self.ports_list[self.sv_port].write(self.cmd, self.rep)
            logging.info(self.cmd.toString() + " reply: " + self.rep.toString())

            self.init_complete = False
            while not self.init_complete:
                self.cmd.clear()
                self.cmd.addString("check_graph_start")
                self.ports_list[self.sv_port].write(self.cmd, self.rep)
                logging.info(self.cmd.toString() + " reply: " + self.rep.toString())
                time.sleep(0.5)
                if self.rep.get(0).asString() == 'ack':
                    self.init_complete = True
        else:
            self.init_complete = True

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
        elif action == "get_node_data":
            if command.size() < 2:
                reply.addString("nack")
            else:
                curr_node = self.node_data[str(command.get(1).asInt())]
                reply.addString("ack")
                reply.addString(curr_node.name)
                reply.addString(str(curr_node.values))
                reply.addString(str(curr_node.edges_list))
        else:
            reply.addString("nack")

        return True

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

        if self.ports_list[self.sv_port].getOutputCount() > 0:
            if not self.pause and self.init_complete and self.t < self.read_data.shape[0]:
                currData = self.read_data[self.t, :]
                new_nodes, new_edges = self.neurolearn(currData)
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
                    self.cmd.addInt(ne[0])
                    self.cmd.addInt(ne[1])
                    self.ports_list[self.sv_port].write(self.cmd, self.rep)
                self.t += 1
                logging.info(self.cmd.toString() + " reply: " + self.rep.toString())
                time.sleep(self.pauseVal)
            else:
                time.sleep(0.05)
        else:
            self.wait_connection()
        return True

    def neurolearn(self, currDataSTR):
        new_nodes_list = []
        new_edges_list = []

        for i, j in enumerate(currDataSTR):
            curr_node_data = self.node_data[str(i)]
            if j not in curr_node_data.values:
                new_node_idx = len(self.node_data.keys())
                curr_node_data.values = curr_node_data.values + [j]
                curr_node_data.edges_list = curr_node_data.edges_list + [(i, new_node_idx)]
                self.node_data[str(new_node_idx)] = node_info(edges_list=[(i, new_node_idx)])
                new_nodes_list.append(i)

        currData = np.array(map(float, currDataSTR))
        if self.prev_data is not None:
            diffInputs = currData - self.prev_data
            print map(float, currData), self.prev_data, diffInputs
            # Long term depresion, long term potentiation, changes in synapses that underly hebbian earning
            diff_idx = np.where(diffInputs != 0)[0]
            edge_nodes = []
            if len(diff_idx) > 1:
                print "ids", diff_idx
                for j in diff_idx:
                    # find value of index j from currData
                    print self.node_data[str(j)].values
                    # find index of value in values list
                    nod = map(float, self.node_data[str(j)].values).index(currData[j])
                    # task 3.3.1 hbp human investigation of association
                    # find corresponding edge pair from edges_list
                    curredgepair = list(self.node_data[str(j)].edges_list[nod])
                    # remove current index from edge pair
                    curredgepair.remove(j)
                    edge_nodes += curredgepair
            if len(edge_nodes) > 1:
                new_edges_list += [tuple(edge_nodes)]

        print new_nodes_list, new_edges_list
        self.prev_data = currData
        return new_nodes_list, new_edges_list


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
