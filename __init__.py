# Copyright (c) 2015, Andreas Damianou

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as pb
## \defgroup icubclient_SAM_source SAM
##\brief@{
## Synthetic Autobiographical Memory for face recognition and action recognition
## @{
## @ingroup icubclient_modules
#\details@{
##\copyright Released under the terms of the GNU GPL v2.0
##\version Version 1.0
##\author Daniel Camilleri \ref <d.camilleri@sheffield.ac.uk>
##
#Description
#========================
#Parameters
#========================
#Input Ports
#========================
#Output Ports
#========================
#Services
#========================
from . import SAM_Core
from . import SAM_Drivers
from . import SAM_Tests
## @}

pb.ion()

default_seed = 123344