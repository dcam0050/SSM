import numpy as np
import thread


class interactionSAMModel:
    def __init__(self):
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
