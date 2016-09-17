device=gpu

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../../src   # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: guenther_ds_b geometry_diag

ds_geom_t: ds_geom_t.cu solovev.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) $(LIBS) $(JSONLIB)  -DDG_BENCHMARK 

solovev_t: solovev_t.cu solovev.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) $(LIBS) $(JSONLIB)  -DDG_BENCHMARK 

conformal_t: conformal_t.cu conformal.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g

ribeiro_t: ribeiro_t.cu ribeiro.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
	
boozer_t: boozer_t.cu boozer.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
	
hamada_t: hamada_t.cu hamada.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
	
equalarc_t: equalarc_t.cu equalarc.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
	
flux_t: flux_t.cu flux.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
	
flux2_t: flux2_t.cu flux2.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
	
orthogonal_t: orthogonal_t.cu orthogonal.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g

conformalX_t: conformalX_t.cu conformalX.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g

orthogonalX_t: orthogonalX_t.cu orthogonalX.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
	
geometry_diag: geometry_diag.cu solovev.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(LIBS) $(INCLUDE) $(JSONLIB) 

%_mpit: %_mpit.cu 
	$(MPICC) $(INCLUDE) $(MPICFLAGS)  $< -o $@ -g $(LIBS) $(JSONLIB)

%_mpib: %_mpib.cu
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ -g $(INCLUDE) $(LIBS) $(JSONLIB)

%_b: %_b.cu 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE)  $(LIBS) $(JSONLIB) -g 

.PHONY: clean doc

doc: 
	doxygen Doxyfile
	
clean:
	rm -f geometry_diag *_t *_b *_mpit *_mpib
