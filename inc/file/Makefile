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
nc_file_t nc_file_mpit\
netcdf_t netcdf_mpit\
probes_parser_t probes_parser_mpit\
probes_t probes_mpit\
jsoncpp_wrapper_t jsonhpp_wrapper_t\
#easy_atts_t easy_atts_mpit\
#


all: $(TARGETS)

%_t: %_t.cpp %.h
	$(CC) $< -o $@ $(CFLAGS) -g $(INCLUDE) $(LIBS) $(VERSION_FLAGS)

%_mpit: %_t.cpp %.h
	$(MPICC) $< -o $@ $(MPICFLAGS) -g $(INCLUDE) $(LIBS) $(VERSION_FLAGS) -DWITH_MPI

############### CATCH TESTS  - REQUIRES catch2 package #################
%_t.o: %_t.cpp %.h
	$(CC) -c $< $(CFLAGS) -g $(INCLUDE) $(LIBS) $(VERSION_FLAGS)

%_mpit.o: %_t.cpp %.h
	$(MPICC) -c $< -o $@ $(MPICFLAGS) -g $(INCLUDE) $(LIBS) $(VERSION_FLAGS) -DWITH_MPI

catch-tests.o: catch-tests.cpp
	$(CC) $(CFLAGS) -c $<

catch-tests-mpi.o: catch-tests-mpi.cpp
	$(MPICC) $(MPICFLAGS) -c $< -DWITH_MPI

easy_atts_t: catch-tests.o easy_atts_t.o
	$(CC) -o $@ catch-tests.o easy_atts_t.o $(LIBS)

easy_atts_mpit: catch-tests-mpi.o easy_atts_mpit.o
	$(MPICC) -o $@ catch-tests-mpi.o easy_atts_mpit.o $(LIBS)

tests: catch-tests.o easy_atts_t.o
	$(CC) -o tests catch-tests.o easy_atts_t.o $(LIBS)

tests-mpi: catch-tests-mpi.o easy_atts_mpit.o
	$(MPICC) -o tests-mpi catch-tests-mpi.o easy_atts_mpit.o $(LIBS)

#######################################################################

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

jsoncpp_wrapper_t: json_wrapper_t.cpp json_wrapper.h
	$(CC) $< -o $@ $(CFLAGS) -g $(INCLUDE) $(JSONLIB)

jsonhpp_wrapper_t: json_wrapper_t.cpp json_wrapper.h
	$(CC) $< -o $@ $(CFLAGS) -g $(INCLUDE) -DDG_USE_JSONHPP

.PHONY: doc clean

doc:
	doxygen Doxyfile

clean:
	rm -f $(TARGETS) *.o tests tests-mpi
