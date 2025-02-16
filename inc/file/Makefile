device=cpu
#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../../ # other project libraries
INCLUDE+= -I../    # other project libraries


TARGETS=easy_dims_t\
easy_output_t\
nc_file_t\
nc_utilities_t\
probes_parser_t\
probes_t\
jsoncpp_wrapper_t\
easy_atts_t
#jsonhpp_wrapper_t # think about how to link both

TARGETSMPI=easy_dims_mpit\
easy_output_mpit\
nc_file_mpit\
nc_utilities_mpit\
probes_parser_mpit\
probes_mpit

CATCH=../../tests/catch-tests
CATCHMPI=../../tests/catch-tests-mpi


all: $(TARGETS) $(TARGETSMPI)

############### CATCH TESTS  - REQUIRES catch2 package #################

%_t.$(device).o: %_t.cpp %.h
	$(CC) $(OPT) $(CFLAGS) -c $< -o $@ -g $(INCLUDE) $(VERSION_FLAGS)

%_mpit.$(device).o: %_t.cpp %.h
	$(MPICC) $(OPT) $(MPICFLAGS) -c $< -o $@ -g $(INCLUDE) $(VERSION_FLAGS) -DWITH_MPI

%_t: $(CATCH).$(device).o %_t.$(device).o
	$(CC) $(OPT) -o $@ $^ -lCatch2 $(LIBS)

%_mpit: $(CATCHMPI).$(device).o %_mpit.$(device).o
	$(MPICC) $(OPT) -o $@ $^ -lCatch2 $(LIBS)

#######################################################################

probes_t.$(device).o: probes_t.cpp probes.h
	$(CC) $(OPT) $(CFLAGS) -c $< -o $@ -g $(INCLUDE)

probes_mpit.$(device).o: probes_t.cpp probes.h
	$(MPICC) $(OPT) $(MPICFLAGS) -c $< -o $@ -g $(INCLUDE) -DWITH_MPI

probes_parser_t.$(device).o: probes_parser_t.cpp probes_params.h json_probes.h
	$(CC) $(OPT) $(CFLAGS) -c $< -o $@ -g $(INCLUDE)

jsoncpp_wrapper_t.$(device).o: json_wrapper_t.cpp json_wrapper.h
	$(CC) $(OPT) $(CFLAGS) -c $< -o $@ -g $(INCLUDE)

jsonhpp_wrapper_t.$(device).o: json_wrapper_t.cpp json_wrapper.h
	$(CC) $(OPT) $(CFLAGS) -c $< -o $@ -g $(INCLUDE) -DDG_USE_JSONHPP

probes_parser_mpit.$(device).o: probes_parser_t.cpp probes_params.h json_probes.h
	$(MPICC) $(OPT) $(MPICFLAGS) -c $< -o $@ -g $(INCLUDE) -DWITH_MPI

probes_t: probes_t.$(device).o $(CATCH).$(device).o
	$(CC) $(OPT) -o $@ $^ $(LIBS) $(JSONLIB) -lCatch2

probes_mpit: probes_mpit.$(device).o $(CATCHMPI).$(device).o
	$(MPICC) $(OPT) -o $@ $^ $(LIBS) $(JSONLIB) -DWITH_MPI -lCatch2

probes_parser_t: probes_parser_t.$(device).o $(CATCH).$(device).o
	$(CC) $(OPT) -o $@ $^ $(JSONLIB) -lCatch2

probes_parser_mpit: probes_parser_mpit.$(device).o $(CATCHMPI).$(device).o
	$(MPICC) $(OPT) -o $@ $^ $(JSONLIB) -DWITH_MPI -lCatch2

jsoncpp_wrapper_t: jsoncpp_wrapper_t.$(device).o $(CATCH).$(device).o
	$(CC) $(OPT) -o $@ $^ $(JSONLIB) -lCatch2

# TODO Find a way to use both jsoncpp and jsonhpp in tests
jsonhpp_wrapper_t: jsonhpp_wrapper_t.$(device).o $(CATCH).$(device).o
	$(CC) $(OPT) -o $@ $^ -lCatch2

$(CATCH).$(device).o: $(CATCH).cpp
	$(CC) $(OPT) $(CFLAGS) $(INCLUDE) -g -c $< -o $@

$(CATCHMPI).$(device).o: $(CATCHMPI).cpp
	$(MPICC) $(OPT) $(MPICFLAGS) $(INCLUDE) -g -c $< -o $@

tests: $(CATCH).$(device).o $(addsuffix .$(device).o,$(TARGETS))
	$(CC) -o $@ $^ -lCatch2 $(LIBS) $(JSONLIB)

mpi-tests: $(CATCHMPI).$(device).o $(addsuffix .$(device).o,$(TARGETSMPI))
	$(MPICC) -o $@ $^ -lCatch2 $(LIBS) $(JSONLIB)


.PHONY: doc clean

doc:
	doxygen Doxyfile

clean:
	rm -f $(TARGETS) $(TARGETSMPI) *.o tests mpi-tests
