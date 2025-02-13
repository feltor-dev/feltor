# SPDX-License-Identifier: BSD-3-Clause
# Repository: https://github.com/CLIUtils/cmake

# Modified to include recommended fix:
# https://github.com/NVIDIA/thrust/blob/main/thrust/cmake/README.md#fixing-legacy-findthrustcmake

# LICENSE copy:

##=============================================================================
#
# BSD 3-Clause License
#
# Copyright (c) 2017, University of Cincinnati
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
##=============================================================================


##=============================================================================
##
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2012 Sandia Corporation.
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##
##=============================================================================

#
# FindThrust
#
# This module finds the Thrust header files and extrats their version.  It
# sets the following variables.
#
# THRUST_INCLUDE_DIR -  Include directory for thrust header files.  (All header
#                       files will actually be in the thrust subdirectory.)
# THRUST_VERSION -      Version of thrust in the form "major.minor.patch".
#

find_path( THRUST_INCLUDE_DIR
  HINTS ./  
        ../thrust
        ../../thrust
        ../../../thrust
        /usr/include/cuda
        /usr/local/include
        /usr/local/cuda/include
        ${CUDA_INCLUDE_DIRS}
        $ENV{HOME}/include
        $ENV{HOME}/include/thrust
  NAMES thrust/version.h
  DOC "Thrust headers"
  )
if( THRUST_INCLUDE_DIR )
  list( REMOVE_DUPLICATES THRUST_INCLUDE_DIR )

  # Find thrust version
  file( STRINGS ${THRUST_INCLUDE_DIR}/thrust/version.h
    version
    REGEX "#define THRUST_VERSION[ \t]+([0-9x]+)"
    )
  string( REGEX REPLACE
    "#define THRUST_VERSION[ \t]+"
    ""
    version
    "${version}"
    )

  # string( REGEX MATCH "^[0-9]" major ${version} )
  # string( REGEX REPLACE "^${major}00" "" version "${version}" )
  # string( REGEX MATCH "^[0-9]" minor ${version} )
  # string( REGEX REPLACE "^${minor}0" "" version "${version}" )
  math(EXPR major "${version} / 100000")
  math(EXPR minor "(${version} / 100) % 1000")
  math(EXPR version "${version} % 100")
  set( THRUST_VERSION "${major}.${minor}.${version}")
  set( THRUST_MAJOR_VERSION "${major}")
  set( THRUST_MINOR_VERSION "${minor}")
endif( THRUST_INCLUDE_DIR )

# Check for required components
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Thrust
  REQUIRED_VARS THRUST_INCLUDE_DIR
  VERSION_VAR THRUST_VERSION
  )

set(THRUST_INCLUDE_DIRS ${THRUST_INCLUDE_DIR})
mark_as_advanced(THRUST_INCLUDE_DIR)

# Create an interface library target
if(THRUST_INCLUDE_DIR)
  add_library(thrust INTERFACE)
  add_library(thrust::thrust ALIAS thrust)
  target_include_directories(thrust INTERFACE ${THRUST_INCLUDE_DIR})
endif()