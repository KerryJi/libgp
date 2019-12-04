# libgp - Gaussian process library for Machine Learning
# Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
# All rights reserved.

ADD_EXECUTABLE(gpdense gp_example_dense.cc)
TARGET_LINK_LIBRARIES(gpdense gp)

ADD_EXECUTABLE(gp_lo_train gp_lo_train.cc)
TARGET_LINK_LIBRARIES(gp_lo_train gp)

ADD_EXECUTABLE(gp_lo_test gp_lo_test.cc)
TARGET_LINK_LIBRARIES(gp_lo_test gp)
