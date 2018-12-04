# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
We do not (yet) follow semantic versioning.
Only code changes are reported here, we do not track changes in the
doxygen documentation or tex writeups.

## [Unreleased]
### Added
- dg::Elliptic3d: a three-dimensional version of dg::Elliptic
- dg::Helmholtz3d: a three-dimensional version of dg::Helmholtz
- a 2nd set\_chi function in Elliptic allowing a SparseTensor argument
- new member functions for DS
- adaptive timesteppers, notably through the dg::Adaptive and dg::ERKStep and dg::ARKStep classes in inc/dg/adaptive.h and tests in inc/dg/adaptive\_t.cu
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
- copyable member function in CG, Invert, RungeKutta and Karniadakis

### Changed
- change folder name from inc/dg/geometry to inc/dg/topology (mainly to avoid clashes with the new inc/dg/geometries) and adapt include directives accordingly
- dg::split now creates vector views instead of hard copies
- default constructor of MPI\_Vector constructs empty communicator
- set\_communicator in MPI\_Vector takes three arguments now to avoid group
  creation
- Configure cuda-aware mpi test a warning instead of an error
- rewrite feltordiag.cu merging ncdiag and filamentdiag
- Remove container argument from dg::geo::SafetyFactor constructor (Since it
  always takes Alpha)
- Reimplement and merge src/feltor with src/asela
- More general interface for geometries/init.h functors including Nprofile and
  ZonalFlow
- Rename BinaryFunctors to CylindricalFunctors (they are not only binary after all)
- Rewrite dg::ClonePtr as a true pointer class and adapt memory\_t.cu
- Change grids() method interface of Multigrid class to grid(unsigned) and adapt this in all programs using it (now we completely avoid exposing ClonePtr to users)
- pushForwardPerp for tensors uses SparseTensor as output argument
- interface for DS
- boundary condition implementation of Fieldaligned (reflected in parallel.pdf)
- boundary conditions in interpolation functions due to Fieldaligned
- better test programs ds\_t.cu, ds\_curv\_t.cu and ds\_guenther\_t.cu
  including corresponding \*mpit.cu programs
- FieldR, FieldZ and FieldP are now named BFieldR, BFieldZ and BFieldP
- dg::RK now named dg::RungeKutta and stages not a template parameter any more
- dg::AB now named dg::AdamsBashforth and stages not a template parameter any more
- Redesign dg::SparseTensor and all associated dg::tensor functions;
  necessarily change implementation of grid geometries
- change default optimization to -O2 due to gcc-7 bug
- replace use of dg::transfer with either dg::assign or dg::construct throughout the dg library
- improve 3d benchmark functionality in cluster\_mpib.cu

### Deprecated
- dg::blas1::transfer
- the header geometries/geometries.h; now dg/geometries/geometries.h is
  preferred for unified access and easier recognition as a dg header file
- the header file/nc\_utilities.h; now dg/file/nc\_utilities.h is preferred for
  unified access and easier recognition as a dg header file
- the Helmholtz2 class (it's much faster to use Helmholtz twice)

### Removed
- dg::join (no longer necessary because split works on views)
- diag/filamentdiag.cu and diag/ncdiag.cpp (merged into feltordiag.cu)
- src/asela and src/asela2d (now merged into src/feltor and src/feltor2d)
- src/feltor2D (now merged in src/feltor project as an input parameter)
- src/feltor/feltor.cu (not really useful any more with our adatpive
  Timesteppers as we don't need to find suitable timesteps anymore)
- src/feltor/feltor\_mpi.cu (now merged into feltor\_hpc.cu)
- BinaryFunctorAdapter and aCloneableBinaryFunctor ( we now use the better
  CylindricalFunctor based on std::function to do the type erasure)
- TensorElliptic, GeneralElliptic and GeneralEllipticSym classes ( superceded by Elliptic3d)
- ds\_geom\_t.cu, ds\_guenther\_b.cu (superceded by better test programs)
- dg::SparseElement (it's more intuitive to use a container directly)

### Fixed
- fix bug in reconnection\_mpi.cu regarding output
- fix bug in diag/reco2Ddiag.cu regarding DVec in interpolation
- fix bug of too many constructor parameters in BathRZ constructor in feltorShw/feltur.cu, feltorShw/feltor\_hpc.cu, feltorShw/feltor\_mpi.cu, bathRZ\_t.cu
- fix bug in feltorSHp/feltor\_hpc.cu regarding step function in timestepper
- fix bug in polar/polar.cu regarding DVec in interpolation function
- fix bug in exblas/mylibm.hpp regarding msvc preprocessor
- change parameter type in members of aCommunicator and NearesNeighborComm to pointer types to allow different container types in blas2 functions for mpi vectors too
- fix bug in GeneralComm global\_scatter\_reduce member initializing values to zero and adding local\_size to constructor

