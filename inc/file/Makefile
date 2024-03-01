device=cpp
#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../../ # other project libraries
INCLUDE+= -I../    # other project libraries

TARGETS=easy_dims_t easy_dims_mpit\
easy_output_t easy_output_mpit\
easy_input_t easy_input_mpit\
netcdf_t netcdf_mpit\
probes_parser_t probes_parser_mpit\
probes_t probes_mpit\
jsoncpp_utilities_t jsonhpp_utilities_t\
jsoncpp_netcdf_t jsonhpp_netcdf_t\


all: $(TARGETS)

%_t: %_t.cpp %.h
	$(CC) $< -o $@ $(CFLAGS) -g $(INCLUDE) $(LIBS)

%_mpit: %_t.cpp %.h
	$(MPICC) $< -o $@ $(MPICFLAGS) -g $(INCLUDE) $(LIBS) -DWITH_MPI

netcdf_t: netcdf_t.cpp
	$(CC) $< -o $@ $(CFLAGS) -g $(INCLUDE) $(LIBS) $(JSONLIB)

netcdf_mpit: netcdf_t.cpp
	$(MPICC) $< -o $@ $(MPICFLAGS) -g $(INCLUDE) $(LIBS) -DWITH_MPI $(JSONLIB)

probes_t: probes_t.cpp
	$(CC) $< -o $@ $(CFLAGS) -g $(INCLUDE) $(LIBS) $(JSONLIB)

probes_mpit: probes_t.cpp
	$(MPICC) $< -o $@ $(MPICFLAGS) -g $(INCLUDE) $(LIBS) -DWITH_MPI $(JSONLIB)

probes_parser_t: probes_parser_t.cpp probes_params.h json_probes.h
	$(CC) $< -o $@ $(CFLAGS) -g $(INCLUDE) $(JSONLIB)

probes_parser_mpit: probes_parser_t.cpp probes_params.h json_probes.h
	$(MPICC) $< -o $@ $(MPICFLAGS) -g $(INCLUDE) $(JSONLIB) -DWITH_MPI

jsoncpp_utilities_t: json_utilities_t.cpp json_utilities.h
	$(CC) $< -o $@ $(CFLAGS) -g $(INCLUDE) $(JSONLIB)

jsonhpp_utilities_t: json_utilities_t.cpp json_utilities.h
	$(CC) $< -o $@ $(CFLAGS) -g $(INCLUDE) -DDG_USE_JSONHPP

jsoncpp_netcdf_t: json_netcdf_t.cpp json_netcdf.h
	$(CC) $< -o $@ $(CFLAGS) -g $(INCLUDE) $(JSONLIB) $(LIBS) $(VERSION_FLAGS)

jsonhpp_netcdf_t: json_netcdf_t.cpp json_netcdf.h
	$(CC) $< -o $@ $(CFLAGS) -g $(INCLUDE) $(LIBS) $(VERSION_FLAGS) -DDG_USE_JSONHPP

.PHONY: doc clean

doc:
	doxygen Doxyfile

clean:
	rm -f $(TARGETS)
