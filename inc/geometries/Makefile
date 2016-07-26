device=gpu

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../../src   # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: guenther_ds_b geometry_diag

ds_geom_t: ds_geom_t.cu solovev.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) $(LIBS)  -DDG_BENCHMARK 

conformal_t: conformal_t.cu conformal.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -g
	
boozer_t: boozer_t.cu boozer.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -g
	
hamada_t: hamada_t.cu hamada.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -g
	
equalarc_t: equalarc_t.cu equalarc.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -g
	
flux_t: flux_t.cu flux.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -g
	
flux2_t: flux2_t.cu flux2.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -g
	
orthogonal_t: orthogonal_t.cu orthogonal.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -g

conformalX_t: conformalX_t.cu conformalX.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -g

orthogonalX_t: orthogonalX_t.cu orthogonalX.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -g
	
geometry_diag: geometry_diag.cu solovev.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(LIBS) $(INCLUDE) 

%_mpit: %_mpit.cu 
	$(MPICC) $(INCLUDE) $(MPICFLAGS)  $< -o $@ -g $(LIBS)

%_mpib: %_mpib.cu
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ -g $(INCLUDE) $(LIBS) 

%_b: %_b.cu 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE)  $(LIBS)

.PHONY: clean doc

doc: 
	doxygen Doxyfile
	
clean:
	rm -f geometry_diag *_t *_b *_mpit *_mpib
