# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
We do not (yet) follow semantic versioning.
Only changes in code are reported here, we do not track changes in the
doxygen documentation, READMEs or tex writeups.
## [v5.2]
### Added
 - M100 config file
 - json utility functions get, get_idx in json_utilities.h which adds a small abstraction layer that gives a user more control over what happens if a variable is not found
 - json utility functions file2Json, and string2Json in json_utilities.h which adds a small abstraction layer that gives a user more control over what happens if an error happens during the parsing of a file
 - "easy output" netcdf utility functions that are particularly useful for MPI output: either write data in parallel or funnel through the master thread
 - new include files dg/file/json_utilities.h and dg/exblas/exblas.h
 - new class dg::Gradient for gradient and variation
 - new tensor functions tensor::
 - new class dg::Advection for the upwind scheme
 - dg::blas1::dot and dg::blas2::dot and corresponding exblas functions now detect NaN and Inf errors
### Changed
 - namespace file changed to **dg::file** and exblas changed to **dg::exblas** (for consistency reasons, everything should go into the dg namespace, which in particular reduces the chance for name-clashes to just one, namely 'dg')
 - changed file paths **dg/file/file.h**, **dg/geometries/geometries.h** , **dg/file/nc_utilities.h**
 - std=c++14 We use the C++-14 standard now (previously 11)
 - vectorclass dependency changed to vectorclass/version1 (previously we used a custom upload on feltor-dev repository)
 - default cuda compute capability bumped to sm-61 (previously sm-35)
 - Marconi config now uses jsoncpp module (previously manually installed)
 - Moved variation memeber function into new class Gradient (previously in ArakawaX and Poisson)
 - blas1::dot and blas2::dot now both do not accumulate rest of multiplication
### Deprecated
### Removed
 - removed diag/feltordiag.cu
### Fixed
 - Fix bug: race condition in dg::blas1::dot and dg::blas2::dot on GPUs that led to hard to reproduce and seemingly unreasonable crashes
 - Fix bugs: std namespace in diag/probes.h
 - Fix bug: const in exblas::cpu::get_element 
 - Fix bug: correct  indices in exblas::cpu::make_vcl_vec8d

## [v5.1]
### Added
- dg::Elliptic3d: a three-dimensional version of dg::Elliptic
- Add 4 parameter symv member to dg::Elliptic class
- dg::Helmholtz3d: a three-dimensional version of dg::Helmholtz
- a 2nd set\_chi function in Elliptic allowing a SparseTensor argument
- new member functions for DS
- Adaptive timesteppers through the dg::Adaptive and dg::ERKStep and dg::ARKStep classes in inc/dg/adaptive.h and tests in inc/dg/adaptive\_t.cu
- dg::ButcherTableau and dg::ConvertsToButcherTableau in inc/dg/tableau.h
- Add interface to Extrapolation class so that it can take a time variable
- dg::View vector view class in inc/dg/view.h usable in dg::blas1 and dg::blas2 functions (tested in inc/dg/view\_t.cu)
- introduce dg::assign and dg::construct that replace the dg::transfer functions and have increased flexibility through an additional template parameter pack
- dg::tensor::inv\_multiply3d and dg::tensor::inv\_multiply2d functions
- Add unified typedefs for geometry, matrix, container and value types in dg library classes
- Add more ContainerType template parameters to ArakawaX, CG, Invert, Elliptic and Helmholtz class members
- Add set\_chi member to ArakawaX to allow for an optional prefactor (like the magnetic field)
- Additional constructors in BijectiveComm consistent with SujectiveComm
- Add consistency test to mpi\_collective\_mpit.cu
- copyable member function in CG, Invert, RungeKutta, AdamsBashforth and Karniadakis
- float overloads of dg::one and dg::zero functions
- Add trivial 2d and 3d operator() for most 1d functors in functors.h
- New subroutines in subroutines.h
- Add shift\_topologic member to RealTopology overriding boundary conditions

### Changed
- Redesign: dg::AB now named dg::AdamsBashforth and stages not a template parameter any more (changed src/lamb\_dipole/shu\_time.cu accordingly)
- Redesign: dg::RK now named dg::RungeKutta and stages not a template parameter any more
- stepperRK does not need a stage template parameter any more
- Redesign dg::SparseTensor and all associated dg::tensor functions; (Not allowing unset values significantly reduces complexity and especially allows easy and fast implementation of dg::tensor::multiply2d and dg::tensor::multiply3d in terms of dg::blas1::subroutine)
- Change construction of Metric and Jacobian in grid geometries and implementation of dg::pushForward transformation functions
- Change grids() method interface of Multigrid class to grid(unsigned) and adapt this in all programs using it (now we completely avoid exposing ClonePtr to users)
- Reimplement and merge src/feltor with src/asela
- Reimplement src/heat project
- experimental use of adaptive timestepper in src/toeflR.cu and src/toeflR.cuh
- change folder name from inc/dg/geometry to inc/dg/topology (mainly to avoid clashes with the new inc/dg/geometries) and adapt include directives accordingly
- dg::split now creates vector views instead of hard copies
- default constructor of MPI\_Vector constructs empty communicator instead of MPI\_COMM\_WORLD
- set\_communicator in MPI\_Vector takes three arguments now to avoid group
  creation
- Configure cuda-aware mpi test a warning instead of an error
- rewrite feltordiag.cu merging ncdiag and filamentdiag
- Remove container argument from dg::geo::SafetyFactor constructor (Since it always takes Alpha)
- More general interface for geometries/init.h functors including Nprofile and ZonalFlow (Old Psi functors are now regular functors)
- Rename BinaryFunctors to CylindricalFunctors (they are not only binary after all) and change files in inc/geometries accordingly
- Rewrite dg::ClonePtr as a true pointer class and adapt memory\_t.cu
- dg::pushForwardPerp for tensors uses SparseTensor as output argument
- interface for DS
- boundary condition implementation of Fieldaligned (reflected in parallel.pdf)
- boundary conditions in interpolation functions due to Fieldaligned
- better test programs ds\_t.cu, ds\_curv\_t.cu and ds\_guenther\_t.cu
  including corresponding \*mpit.cu programs
- FieldR, FieldZ and FieldP are now named BFieldR, BFieldZ and BFieldP
- change default optimization to -O2 due to gcc-7 bug
- replace use of dg::transfer with either dg::assign or dg::construct throughout the dg library
- improve 3d benchmark functionality in cluster\_mpib.cu
- add hasXpoint() and default parameters to solovev::parameter
- add map\_index to NearestNeighborComm to signify responsibility of mapping matrix indices to buffer indices
- Optimization: implement fast EllSparseBlockMat kernel for z derivative
- Optimization: change buffer layout in dg::NearestNeighborComm and CooSparseBlockMat kernels to avoid slow scatter/gather operations in mpi matrix-vector multiplication
- Optimization: implement faster kernels for CooSparseBlockMat symv kernel to accelerate mpi symv with low computation to communication ratio

### Deprecated
- dg::blas1::transfer (replaced by the more general dg::assign and dg::construct)
- the header geometries/geometries.h (now dg/geometries/geometries.h is
  preferred for unified access and easier recognition as a dg header file)
- the header file/nc\_utilities.h ( now dg/file/nc\_utilities.h is preferred for
  unified access and easier recognition as a dg header file)
- the Helmholtz2 class (it's much faster to use Helmholtz twice)

### Removed
- diag/filamentdiag.cu and diag/ncdiag.cpp (merged into feltordiag.cu)
- src/asela and src/asela2d (now merged into src/feltor and src/feltor2d)
- src/feltor2D (now merged in src/feltor project as an input parameter)
- src/feltor/feltor.cu (not really useful any more with our adatpive
  Timesteppers as we don't need to find suitable timesteps anymore)
- src/feltor/feltor\_mpi.cu (now merged into feltor\_hpc.cu)
- BinaryFunctorAdapter and aCloneableBinaryFunctor ( we now use the better
  CylindricalFunctor based on std::function to do the type erasure)
- TensorElliptic, GeneralElliptic and GeneralEllipticSym classes ( superceded by Elliptic3d)
- inc/geometries/ds\_geom\_t.cu and ds\_guenther\_b.cu (superceded by better test programs)
- dg::SparseElement (it's more intuitive and practical to use a container directly and the performance gain is negligeable)
- src/heat/init.h
- dg::join (no longer necessary because split works on views)

### Fixed
- fix bug in diag/reco2Ddiag.cu regarding DVec in interpolation
- fix bug in src/reco2D/reconnection\_mpi.cu regarding output
- fix bug in src/polar/polar.cu regarding DVec in interpolation function
- fix bug of too many constructor parameters in BathRZ constructor in src/feltorShw/feltur.cu, feltorShw/feltor\_hpc.cu, feltorShw/feltor\_mpi.cu, inc/dg/bathRZ\_t.cu
- fix bug in src/feltorSHp/feltor\_hpc.cu regarding step function in timestepper
- fix bug in inc/dg/backend/exblas/mylibm.hpp regarding msvc preprocessor
- fix bug in dg::Histogramm: remove DG\_DEVICE from operator()
- change parameter type in members of aCommunicator and NearesNeighborComm to pointer types to allow different container types in blas2 functions for mpi vectors too
- fix bug in GeneralComm global\_scatter\_reduce member initializing values to zero and adding local\_size to constructor
- geometries functors can now be used as std::function (by removing the clone method)
- geometries/geometries.h and file/nc\_utilities.h headers are now also includable by multiple translation units

