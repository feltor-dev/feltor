# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
We do not (yet) follow semantic versioning.

## [Unreleased]
### Added
- Elliptic3d analogous to Elliptic
- a 2nd set\_chi function in Elliptic allowing a SparseTensor argument
- new member functions for DS
- adaptive timesteppers, notably through the dg::Adaptive and dg::ERKStep and dg::ARKStep classes
- dg::ButcherTableau and dg::ConvertsToButcherTableau
- Extrapolation class now can take a time variable
- dg::View vector view class usable in dg::blas1 and dg::blas2
- introduce dg::assign and dg::construct that replace the dg::blas1::transfer functions and take an additional optional parameter pack

### Changed
- Rewrite dg::ClonePtr as a pointer class
- Change grids method interface of Multigrid to grid(unsigned)
- pushForwardPerp for tensors uses SparseTensor as output argument
- interface for DS
- better test programs ds\_t.cu, ds\_curv\_t.cu and ds\_guenther\_t.cu
  including corresponding \*mpit.cu programs
- FieldR, FieldZ and FieldP are now named BFieldR, BFieldZ and BFieldP
- dg::RK now named dg::RungeKutta and stages not a template parameter any more
- dg::AB now named dg::AdamsBashforth and stages not a template parameter any more

### Deprecated
- dg::blas1::transfer
- the header geometries/geometries.h; now dg/geometries/geometries.h is preferred for unified access and easier recognition as a dg header file
- the header file/nc\_utilities.h; now dg/file/nc\_utilities.h is preferred for unified access and easier recognition as a dg header file

### Removed
- TensorElliptic, GeneralElliptic and GeneralEllipticSym classes
- ds\_geom\_t.cu, ds\_guenther\_b.cu (superceded by better test programs)

### Fixed
    Fixed for any bug fixes.

### Security
    Security in case of vulnerabilities.

