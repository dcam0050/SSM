import logging
import numpy as np
from SAM.SAM_Core.SAM_utils import timeout
import time

try:
    import yarp
    yarpFound = True
except ImportError as e:
    yarpFound = False

try:
    import rospy
    import roslib
    import std_msgs.msg
    import std_srvs.srv
    rosFound = True
except ImportError as e:
    rosFound = False


class YarpNetworkModule(yarp.RFModule):
    def __init__(self):
        yarp.RFModule.__init__(self)

    def respond(self, command, reply):
        replylist = []
        reply.clear()

        commandlist = self.disassemble_bottle(command)
        print commandlist
        self.respond_method(commandlist, replylist)
        print replylist
        self.assemble_bottle(reply, replylist)
        return True

    def create_rpc(self, name):
        rpc_port = yarp.Port()
        rpc_port.open(name)
        self.attach(rpc_port)
        return rpc_port

    @staticmethod
    def create_port(name, dtype):
        dtype = dtype.lower()
        if dtype == "imagergb":
            p = yarp.BufferedPortImageRgb()
            p.open(name)
        elif dtype == "imagemono":
            p = yarp.BufferedPortImageMono()
            p.open(name)
        elif dtype == "string" or dtype == "bottle":
            p = yarp.BufferedPortBottle()
            p.open(name)
        else:
            raise TypeError("Type " + str(dtype) + " not yet supported")

        return p

    @staticmethod
    def connect_ports(dest, recv):
        yarp.Network.connect(dest, recv)

    @staticmethod
    def send_image(image_port, data):
        yarp_image = image_port.prepare()
        yarp_image.resize(data.shape[1], data.shape[0])
        data = data.astype(np.uint8)
        yarp_image.setExternal(data, data.shape[1], data.shape[0])
        image_port.write()

    def send_message(self, port, data):
        data_placeholder = port.prepare()
        data_placeholder.clear()
        self.assemble_bottle(data_placeholder, data)
        port.write()

    @staticmethod
    def assemble_bottle(data_placeholder, data):
        for m in data:
            if isinstance(m, str):
                data_placeholder.addString(m)
            elif isinstance(m, int):
                data_placeholder.addInt(m)
            elif isinstance(m, float):
                data_placeholder.addDouble(m)

    @staticmethod
    def disassemble_bottle(bottle):
        data = []

        for m in range(bottle.size()):
            currm = bottle.get(m)
            if currm.isString():
                data.append(currm.asString())
            elif currm.isInt():
                data.append(currm.asInt())
            elif currm.isDouble():
                data.append(currm.asDouble())
        return data

    def read_frame(self, port, dtype):
        dtype = dtype.lower()
        frame_read = port.read(True)

        if dtype == "imagergb":
            frame = yarp.ImageRgb()
        elif dtype == "imagemono":
            frame = yarp.ImageMono()
        elif dtype == "string" or dtype == "bottle":
            frame_read = self.disassemble_bottle(frame_read)
        else:
            raise TypeError("Type " + str(dtype) + " not yet supported")

        # returning data in yarp format
        return frame_read

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

    @staticmethod
    def getInputCount(port):
        return port.getInputCount()

    @staticmethod
    def getOutputCount(port):
        return port.getOutputCount()


class ROSNetworkModule(object):
    def __init__(self):
        rospy.init_node('interactionSAMModel', anonymous=True)
        self.rpc_service_name = None

    def runModule(self, rf):
        self.configure(rf)

    def configure(self, rf):
        pass

    def updateModule(self):
        pass

    def respond(self, command, reply=None):

        commandlist = command.split(' ')
        replylist = []

        self.respond_method(commandlist, replylist)
        reply = std_msgs.msg.String()
        reply.data = ' '.join(replylist)

        return reply

    def create_rpc(self, name):
        name = name[1:].replace('/', '_').split(':')[0]
        self.rpc_service_name = name
        s = rospy.Service(name, std_srvs.srv.Trigger, self.respond)
        rospy.wait_for_service(name)
        return s

    def create_port(self, name, dtype):
        return None

    def close_port(self, j):
        return None

    def connect_ports(self, dest, recv):
        pass

    @staticmethod
    def getInputCount(port):
        return 1

    @staticmethod
    def getOutputCount(port):
        return 1

    def read_frame(self, port, dtype):
        pass

    def write_frame(self, port, dtype):
        pass
