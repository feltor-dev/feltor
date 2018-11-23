# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
We do not (yet) follow semantic versioning.

## [Unreleased]
### Added
- dg::Elliptic3d: a three-dimensional version of dg::Elliptic
- dg::Helmholtz3d: a three-dimensional version of dg::Helmholtz
- a 2nd set\_chi function in Elliptic allowing a SparseTensor argument
- new member functions for DS
- adaptive timesteppers, notably through the dg::Adaptive and dg::ERKStep and
  dg::ARKStep classes
- dg::ButcherTableau and dg::ConvertsToButcherTableau
- Extrapolation class now can take a time variable
- dg::View vector view class usable in dg::blas1 and dg::blas2
- introduce dg::assign and dg::construct that replace the dg::blas1::transfer
  functions and take an additional optional parameter pack
- dg::tensor::inv\_multiply3d and dg::tensor::inv\_multiply2d functions

### Changed
- Reimplement and merge src/feltor with src/asela
- More general interface for geometries/init.h functors including Nprofile and
  ZonalFlow
- Rename BinaryFunctors to CylindricalFunctors (they are not only binary after all)
- Rewrite dg::ClonePtr as a pointer class
- Change grids() method interface of Multigrid class to grid(unsigned) (now we
  completely avoid exposing ClonePtr to users)
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

### Deprecated
- dg::blas1::transfer
- the header geometries/geometries.h; now dg/geometries/geometries.h is
  preferred for unified access and easier recognition as a dg header file
- the header file/nc\_utilities.h; now dg/file/nc\_utilities.h is preferred for
  unified access and easier recognition as a dg header file
- the Helmholtz2 class (it's much faster to use Helmholtz twice)

### Removed
- src/feltor/feltor.cu (not really useful any more with our adatpive
  Timesteppers as we don't need to find suitable timesteps anymore)
- src/feltor/feltor\_mpi.cu (now merged into feltor\_hpc.cu)
- BinaryFunctorAdapter and aCloneableBinaryFunctor ( we now use the better
  CylindricalFunctor based on std::function to do the type erasure)
- TensorElliptic, GeneralElliptic and GeneralEllipticSym classes ( superceded
  by Elliptic3d)
- ds\_geom\_t.cu, ds\_guenther\_b.cu (superceded by better test programs)
- dg::SparseElement (it's more intuitive to use a container directly)
- src/asela and src/asela2d (now merged into src/feltor and src/feltor2d)

### Fixed
    Fixed for any bug fixes.

### Security
    Security in case of vulnerabilities.

