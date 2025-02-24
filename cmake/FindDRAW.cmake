# FindDRAW
#
# Finds DRAW library available at:
# https://github.com/feltor-dev/draw
# 
# Sets the variable DRAW_INCLUDE_DIR and creates an interface library target
# draw::draw.

find_path(DRAW_INCLUDE_DIR
  HINTS ./  
        /usr/include/
        /usr/local/include
        $ENV{HOME}/include
        $ENV{HOME}/include/draw
  NAMES draw/host_window.h
  DOC "Draw headers"
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DRAW REQUIRED_VARS DRAW_INCLUDE_DIR)
mark_as_advanced(DRAW_INCLUDE_DIR)

# Create an interface library target
if(DRAW_INCLUDE_DIR)
  add_library(draw INTERFACE)
  add_library(draw::draw ALIAS draw)
  target_include_directories(draw INTERFACE ${DRAW_INCLUDE_DIR})
endif()
