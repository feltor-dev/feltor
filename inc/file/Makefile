device=cpp
#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../../ # other project libraries
INCLUDE+= -I../    # other project libraries


TARGETS=easy_dims_t\
easy_output_t\
easy_input_t\
nc_file_t\
netcdf_t\
probes_parser_t\
probes_t\
jsoncpp_wrapper_t\
easy_atts_t
#jsonhpp_wrapper_t # think about how to link both

TARGETSMPI=easy_dims_mpit\
easy_output_mpit\
easy_input_mpit\
nc_file_mpit\
netcdf_mpit\
probes_parser_mpit\
probes_mpit

CATCH=../../tests/catch-tests
CATCHMPI=../../tests/catch-tests-mpi


all: $(TARGETS) $(TARGETSMPI)

############### CATCH TESTS  - REQUIRES catch2 package #################
$(CATCH).o: $(CATCH).cpp
	$(CC) $(CFLAGS) -g -c $< -o $@

$(CATCHMPI).o: $(CATCHMPI).cpp
	$(MPICC) $(MPICFLAGS) -g -c $< -o $@ -DWITH_MPI

%_t.o: %_t.cpp %.h
	$(CC) $(CFLAGS) -c $< -g $(INCLUDE) $(VERSION_FLAGS)

%_mpit.o: %_t.cpp %.h
	$(MPICC) $(MPICFLAGS) -c $< -o $@ -g $(INCLUDE) $(VERSION_FLAGS) -DWITH_MPI

%_t: $(CATCH).o %_t.o
	$(CC) -o $@ $^ $(LIBS)

%_mpit: $(CATCHMPI).o %_mpit.o
	$(MPICC) -o $@ $^ $(LIBS)

#######################################################################

probes_t.o: probes_t.cpp probes.h
	$(CC) $(CFLAGS) -c $< -o $@ -g $(INCLUDE)

probes_mpit.o: probes_t.cpp probes.h
	$(MPICC) $(MPICFLAGS) -c $< -o $@ -g $(INCLUDE) -DWITH_MPI

probes_parser_t.o: probes_parser_t.cpp probes_params.h json_probes.h
	$(CC) $(CFLAGS) -c $< -o $@ -g $(INCLUDE)

jsoncpp_wrapper_t.o: json_wrapper_t.cpp json_wrapper.h
	$(CC) $(CFLAGS) -c $< -o $@ -g $(INCLUDE)

jsonhpp_wrapper_t.o: json_wrapper_t.cpp json_wrapper.h
	$(CC) $(CFLAGS) -c $< -o $@ -g $(INCLUDE) -DDG_USE_JSONHPP

probes_parser_mpit.o: probes_parser_t.cpp probes_params.h json_probes.h
	$(MPICC) $(MPICFLAGS) -c $< -o $@ -g $(INCLUDE) -DWITH_MPI

probes_t: probes_t.o $(CATCH).o
	$(CC) -o $@ $^ $(LIBS) $(JSONLIB)

probes_mpit: probes_mpit.o $(CATCHMPI).o
	$(MPICC) -o $@ $^ $(LIBS) $(JSONLIB) -DWITH_MPI

probes_parser_t: probes_parser_t.o $(CATCH).o
	$(CC) -o $@ $^ $(JSONLIB)

probes_parser_mpit: probes_parser_mpit.o $(CATCHMPI).o
	$(MPICC) -o $@ $^ $(JSONLIB) -DWITH_MPI

jsoncpp_wrapper_t: jsoncpp_wrapper_t.o $(CATCH).o
	$(CC) -o $@ $^ $(JSONLIB)

jsonhpp_wrapper_t: jsonhpp_wrapper_t.o $(CATCH).o
	$(CC) -o $@ $^

tests: $(CATCH).o $(addsuffix .o,$(TARGETS))
	$(CC) -o tests $^ $(LIBS) $(JSONLIB)

tests-mpi: $(CATCHMPI).o easy_dims_mpit.o easy_output_mpit.o #  $(addsuffix .o,$(TARGETSMPI))
	$(MPICC) -o tests-mpi $^ $(LIBS) $(JSONLIB)


.PHONY: doc clean

doc:
	doxygen Doxyfile

clean:
	rm -f $(TARGETS) $(TARGETSMPI) *.o tests tests-mpi
