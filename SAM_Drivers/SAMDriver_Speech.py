# """"""""""""""""""""""""""""""""""""""""""""""
# The University of Sheffield
# WYSIWYD Project
#
# SAMpy class for implementation of SAM module
#
# Created on 26 May 2015
#
# @authors: Uriel Martinez, Luke Boorman, Andreas Damianou
#
# """"""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import os
import readline
from SAM.SAM_Core import SAMDriver
from SAM.SAM_Core import SAMTesting
import logging
from scipy.signal import lfilter
from scipy.io import wavfile, loadmat
from scipy.fftpack import dct
from sklearn.mixture import GMM
import copy

## @ingroup icubclient_SAM_Drivers
class SAMDriver_Speech(SAMDriver):
    """
        Class developed for the implementation of face recognition.
        """

    def __init__(self):
        """
        Initialise class using SAMDriver.__init__ and augment with custom parameters.

        additionalParameterList is a list of extra parameters to preserve between training and interaction.
        """
        SAMDriver.__init__(self)
        self.gmm_data = dict()
        self.additionalParametersList = ['gmm_data']

    def loadParameters(self, parser, trainName):
        """
            Function to load parameters from the model config.ini file.

            Method to load parameters from file loaded in parser from within section trainName and store these parameters in self.paramsDict.

        Args:
            parser: SafeConfigParser with pre-read config file.
            trainName: Section from which parameters are to be read.

        Returns:
            None
        """
        if parser.has_option(trainName, 'delta'):
            self.paramsDict['delta'] = parser.get(trainName, 'delta') == 'True'
        else:
            self.paramsDict['delta'] = True

        if parser.has_option(trainName, 'context'):
            self.paramsDict['context'] = int(parser.get(trainName, 'context'))
        else:
            self.paramsDict['context'] = 200

        if parser.has_option(trainName, 'n_mixtures'):
            self.paramsDict['n_mixtures'] = int(parser.get(trainName, 'n_mixtures'))
        else:
            self.paramsDict['n_mixtures'] = 25

        if parser.has_option(trainName, 'image_suffix'):
            self.paramsDict['file_suffix'] = parser.get(trainName, 'file_suffix')
        else:
            self.paramsDict['file_suffix'] = '.wav'

        if parser.has_option(trainName, 'numBins'):
            self.paramsDict['numBins'] = int(parser.get(trainName, 'numBins'))
        else:
            self.paramsDict['numBins'] = 10

        if parser.has_option(trainName, 'useBinWidth'):
            self.paramsDict['useBinWidth'] = parser.get(trainName, 'useBinWidth') == 'True'
        else:
            self.paramsDict['useBinWidth'] = True

        if parser.has_option(trainName, 'binWidth'):
            self.paramsDict['binWidth'] = float(parser.get(trainName, 'binWidth'))
        else:
            self.paramsDict['binWidth'] = 0.001

        if parser.has_option(trainName, 'method'):
            self.paramsDict['method'] = parser.get(trainName, 'method')
        else:
            self.paramsDict['method'] = 'sumProb'

        if parser.has_option(trainName, 'labelsAllowedList'):
            self.paramsDict['labelsAllowedList'] = parser.get(trainName, 'labelsAllowedList').replace(' ', '').split(',')
        else:
            self.paramsDict['labelsAllowedList'] = ['Daniel']

    def saveParameters(self):
        """
            Executes SAMDriver.saveParameters to save default parameters.
        """
        SAMDriver.saveParameters(self)

    # """"""""""""""""
    def readData(self, root_data_dir, participant_index, *args, **kw):
        """
        Method which accepts a data directory, reads all the data in and outputs self.Y which is a numpy array with n instances of m length feature vectors and self.L which is a list of text Labels of length n.

        This method reads .ppm images from disk, converts the images to grayscale and serialises the data into a feature vector.

        Args:
            root_data_dir: Data directory.
            participant_index: List of subfolders to consider. Can be left as an empty list.

        Returns:
        """

        gmm_atts = None

        if not os.path.exists(root_data_dir):
            logging.error("CANNOT FIND:" + root_data_dir)
        else:
            logging.info("PATH FOUND")

        if gmm_atts is not None:
            self.gmm_data = gmm_atts

        data_file_count = np.zeros([len(participant_index)])
        data_file_database = {}
        for count_participant, current_participant in enumerate(participant_index):
            current_data_dir = os.path.join(root_data_dir, current_participant)
            data_file_database_p = np.empty(0, dtype=[('orig_file_id', 'a11'), ('file_id', 'i2'),
                                                         ('speech_fname', 'a100')])
            data_utt_count = 0
            if os.path.exists(current_data_dir):
                for file in os.listdir(current_data_dir):
                    fileName, fileExtension = os.path.splitext(file)
                    if fileExtension == self.paramsDict['file_suffix']:
                        file_ttt = np.empty(1, dtype=[('orig_file_id', 'a11'), ('file_id', 'i2'),
                                                         ('speech_fname', 'a100')])
                        file_ttt['orig_file_id'][0] = fileName
                        file_ttt['speech_fname'][0] = file
                        file_ttt['file_id'][0] = data_utt_count
                        data_file_database_p = np.append(data_file_database_p, file_ttt, axis=0)
                        data_utt_count += 1
                # Sort by file id before adding to dictionary
            data_file_database_p = np.sort(data_file_database_p, order=['orig_file_id'])
            data_file_count[count_participant] = len(data_file_database_p)
            data_file_database[current_participant] = data_file_database_p

        min_no_utts = int(np.min(data_file_count))

        # Data size
        logging.info("Found minimum number of utterances: " + str(min_no_utts))
        logging.info("Utterance count: " + str(data_file_count))

        # For each participant
        tmp_utt_data = {}
        for count_participant, current_participant in enumerate(participant_index):
            tmp_utt_data[current_participant] = {}
            # For each utterance
            for current_utt in range(min_no_utts):
                tmp_utt_data[current_participant][current_utt] = {}
                # Read in the file as a wav
                current_utt_path = os.path.join(root_data_dir, current_participant,
                                                data_file_database[current_participant][current_utt][2])

                data_utt = self.readFromFile(current_utt_path, delta=self.paramsDict['delta'],
                                             context=self.paramsDict['context'], spectrum_power=1)
                tmp_utt_data[current_participant][current_utt]['mfcc'] = data_utt

            # Train the GMM on this participants
            par_gmm = GMM(n_components=self.paramsDict['n_mixtures'], covariance_type='full', n_iter=5)
            tr_dat = np.vstack(x['mfcc'] for x in tmp_utt_data[current_participant].values())
            logging.info(str(tr_dat.shape) + " " + current_participant)
            self.gmm_data[current_participant] = par_gmm.fit(tr_dat)

        utt_data = None
        utt_label_data = []

        # Make Supervectors for each utterance for each participant
        print 'Training Speaker GMMs'
        for count_participant, current_participant in enumerate(participant_index):
            logging.info(str(self.paramsDict['n_mixtures']) + ' dim GMM for ' + current_participant)
            for current_utt in range(min_no_utts):
                g = self.gmm_data[current_participant]
                g_means = np.mean(g.predict_proba(tmp_utt_data[current_participant][current_utt]['mfcc']), axis=0)[None, :]
                if utt_data is None:
                    utt_data = g_means
                else:
                    utt_data = np.vstack([utt_data, g_means])
                utt_label_data += [current_participant]

        self.Y = utt_data
        self.L = utt_label_data
        logging.info(str(self.Y.shape) + " " + str(len(self.L)))
        self.allDataDict = dict()
        self.allDataDict['Y'] = copy.deepcopy(self.Y)
        self.allDataDict['L'] = copy.deepcopy(self.L)

        train_list = [n for n, j in enumerate(utt_label_data) if j in self.paramsDict['labelsAllowedList']]

        self.Y = self.Y[train_list]
        self.L = [self.L[index] for index in train_list]

        logging.info(str(self.allDataDict['Y'].shape) + " " + str(len(self.allDataDict['L'])))
        logging.info(str(self.Y.shape) + " " + str(len(self.L)))
        return self.Y.shape[1]

    def processLiveData(self, dataList, thisModel, verbose, additionalData=dict(), visualiseInfo=None):
        """
            Method which receives a list of data frames and outputs a classification if available
            or 'no_classification' if it is not

            Args:
                dataList: List of dataFrames collected. Length of list is variable.
                thisModel: List of models required for testing.
                verbose : Boolean turning logging to stdout on or off.
                additionalData : Dictionary containing additional data required for classification to occur.
            Returns:
               String with result of classification, likelihood of the classification, and list of frames with the
               latest x number of frames popped where x is the window length of the model. Classification result
               can be string `'None'` if the classification is unknown or message is invalid
               or `None` if a different error occurs.
        """
        # Data input in dataList needs to be in the form of 16bit integer data, 1 channel,
        # sampling at 44.1kHz and a duration of 2s

        logging.info('process live data')
        logging.info(len(dataList))
        numSegments = len(dataList)

        labels = [None] * numSegments
        likelihoods = [None] * numSegments

        if numSegments > 0:
            # average all faces
            for i in range(numSegments):
                logging.info('iterating' + str(i))
                data_in = np.fromstring(numSegments[i], dtype=np.int16)
                instance = self.pre_process(data_in)
                logging.info(instance.shape)
                logging.info("Segment: " + str(i))
                [labels[i], likelihoods[i]] = SAMTesting.testSegment(thisModel, instance, verbose,
                                                                     visualiseInfo=visualiseInfo)
            logging.info('Combining classifications')
            finalClassLabel, finalClassProb = SAMTesting.combineClassifications(thisModel, labels, likelihoods)
            logging.info('finalClassLabels ' + str(finalClassLabel))
            logging.info('finalClassProbs ' + str(finalClassProb))
            return finalClassLabel, finalClassProb, []
        else:
            return [None, 0, None]

    def formatGeneratedData(self, instance):
        """
        Method to transform a generated instance from the model into a formatted output.

        Args:
            instance: Feature vector returned during generation of a label.

        Returns:
            Formatted output for instance.
        """
        # normalise image between 0 and 1
        yMin = instance.min()
        instance -= yMin
        yMax = instance.max()
        instance /= yMax
        instance *= 255
        instance = instance.astype(np.uint8)

        return instance

    def pre_process(self, chunk):
        """
        Takes an audio segment and performs pre-processing on it.
        It should be noted that this is designed to be used with real-time audio, and it is assumed that
        the model has already been trained.
        """
        h = []
        f = self.feature_extract(chunk, delta=False)
        for g in self.gmm_data.values():
            # print g.means_.shape
            # print f.shape
            h.append(np.mean(g.score_samples(f)[1], axis=0))
            # h.append(np.mean(g.predict_proba(f), axis=0))

        return np.hstack(h)
        # return np.hstack(np.mean(g.predict_proba(feature_extract(chunk)), axis=0) for g in self.gmm_data.values())

    def readFromFile(self, current_utt_path, spectrum_power=1, delta=True, context=1):
        data_utt = self.read_file(current_utt_path)
        return self.feature_extract(data_utt, spectrum_power=spectrum_power, delta=delta, context=context)

    def feature_extract(self, data, spectrum_power=1, delta=True, context=1):
        # Pre-emphasis
        rate = data[0] / 1000
        try:
            mono_data = data[1].sum(axis=1) / float(data[1].shape[1])
        except:
            mono_data = data[1]
        test_file = self.preemp(np.array(mono_data), 0.97)

        # Frame correctly - Make def with window_size,shift,hamming, etc as options
        test_fr = self.frame(test_file, sample_rate=rate)

        # Take the fourier transform of the signal -
        test_sp = self.spectra(test_fr, p=spectrum_power)  # Power Spectrum for each frame
        test_en = np.sum(test_sp, axis=1)  # Energy for each frame
        test_en = np.where(test_en == 0, np.finfo(float).eps,
                              test_en)  # if energy is zero, we get problems with log

        # Get the MFCCs and make the deltas and delta-deltas
        feats = self.make_mfccs(test_sp, energy=test_en, delta=delta, d_context=context)

        # Add to data set
        return feats

    # -------------------------------------------------------------------------------------------- #
    # Adapted from James Lyons' 'python_speech_features' as allowed under MIT License
    # https://github.com/jameslyons/python_speech_features

    def hz2mel(self, hz):
        """Convert a value in Hertz to Mels
        :param hz: a value in Hz. This can also be a np array, conversion proceeds element-wise.
        :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
        """
        return 2595 * np.log10(1 + hz / 700.0)

    #     return 1127.01048 * np.log(hz/700 +1)

    def mel2hz(self, mel):
        """Convert a value in Mels to Hertz
        :param mel: a value in Mels. This can also be a np array, conversion proceeds element-wise.
        :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
        """
        return 700 * (10 ** (mel / 2595.0) - 1)

    #     return (np.exp(mel / 1127.01048) - 1) * 700

    def preemp(self, input, p):
        """Pre-emphasis filter."""
        return lfilter([1., -p], 1, input)

    #     return np.append(input[0],input[1:]-p*input[:-1])

    def filter_banks(self, nfft, nfilt=25, sample_rate=16, lowfreq=0, highfreq=None):
        """
        Creates the mel filterbanks used to create the MFCCs
        :param nfilt:
        :param sample_rate:
        :param lowfreq:
        :param highfreq:

        :returns: The filter banks for calculating the Mel-Frequency coefficients
        """
        # Sets highfrequency to half sample rate if not set
        # or too large
        if highfreq is None or highfreq > sample_rate * 1000 / 2:
            highfreq = sample_rate * 1000 / 2

        # Compute locations for the mel banks
        l_mel = self.hz2mel(lowfreq)
        h_mel = self.hz2mel(highfreq)
        mel_banks = np.linspace(l_mel, h_mel, nfilt + 2)  # +2 for the start and end values
        bins = np.floor(self.mel2hz(mel_banks) * (nfft) / (sample_rate * 1000))

        # Build the filter banks
        fbank = np.zeros([nfilt, nfft])
        for j in xrange(0, nfilt):
            for i in xrange(int(bins[j]), int(bins[j + 1])):
                fbank[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j])
            for i in xrange(int(bins[j + 1]), int(bins[j + 2])):
                fbank[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
        return fbank

    def read_file(self, file_name):
        """
        Reads the given file and returns an array-like
        """
        return wavfile.read(file_name)

    def frame(self, arr, nfft=None, window_size=25, shift=10, hamming=True, sample_rate=8):
        """
        Takes the signal as an array and windows (with window size supplied in ms)
        Also applies a Hamming window unless specified otherwise.
        """
        window_size = window_size * sample_rate
        shift = shift * sample_rate
        num_windows = np.ceil(arr.shape[0] / window_size)
        if nfft is None:
            nfft = window_size * sample_rate
        window = np.hamming(window_size)
        x = 0
        frames = []
        while x * shift + window_size < len(arr):
            sh_val = x * shift
            fr_range = range(sh_val, sh_val + window_size)
            frames.append(arr[fr_range])
            x += 1
        frames = np.vstack(frames)

        if hamming:
            for idx, fr in enumerate(frames):
                frames[idx] = np.multiply(window, fr)
                frames[idx] = list(frames[idx])
        return frames

    def spectra(self, y, nfft=None, p=1):
        """Returns an array of spectra for an array of waveform frames"""
        if nfft is None:
            nfft = y.shape[1] + 1
        sp = []
        for fr in y:
            fr = np.absolute(np.fft.fft(np.array(fr))) ** p
            sp.append(list(fr))
        return 1.0 / len(sp[0]) * np.vstack(sp)

    def round_base(x, base, r):
        """Rounds a number, x, to the nearest base"""
        return int(base * r(float(x) / base))

    def make_mfccs(self, spec, energy=None, delta=True, d_context=1, nfilt=25):
        x = self.filter_banks(spec.shape[1], nfilt)
        feats = np.dot(spec, x.T)
        feats = np.where(feats == 0, np.finfo(float).eps, feats)  # if feats is zero, we get problems with log
        feats = dct(np.log10(feats), axis=-1, norm='ortho')[:, :13]
        if energy is not None:
            feats[:, 0] = energy
        if delta:
            delts = self.make_deltas(feats, context=d_context)
            dd = self.make_deltas(delts, context=d_context)
            delts = np.vstack(np.append(delts[i], dd[i]) for i in range(delts.shape[0]))
            feats = np.vstack(np.append(feats[i], delts[i]) for i in range(feats.shape[0]))
        return feats

    def make_deltas(self, feats, context=1):
        d = np.zeros(feats.shape)
        for i in range(feats.shape[0]):
            tmp = np.zeros(feats.shape[1])
            for j in range(1, context + 1):
                if i - j < 0:
                    r = j * feats[i + j]
                elif i + j >= feats.shape[0]:
                    r = j * feats[i - j]
                else:
                    r = j * self.calc_range(feats[i + j], feats[i - j])
                tmp = tmp + r
                #         tmp = tmp / np.sum([i**2 for i in range(1,context+1)])
            d[i, :] = tmp
        return d

    def calc_range(self, x, y):
        return x - y