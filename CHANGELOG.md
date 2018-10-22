# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
We do not (yet) follow semantic versioning.

## [Unreleased]
### Added
- new member functions for DS
- adaptive timesteppers, notably through the dg::Adaptive and dg::ERKStep and dg::ARKStep classes
- dg::ButcherTableau and dg::ConvertsToButcherTableau
- dg::View vector view class usable in dg::blas1 and dg::blas2
- introduce dg::assign and dg::construct that replace the dg::blas1::transfer functions and take an additional optional parameter pack

### Changed
- interface for DS
- better test programs ds\_t.cu, ds\_curv\_t.cu and ds\_guenther\_t.cu
  including corresponding \*mpit.cu programs
- dg::RK now named dg::RungeKutta and stages not a template parameter any more
- dg::AB now named dg::AdamsBashforth and stages not a template parameter any more

### Deprecated
- dg::blas1::transfer
- the header geometries/geometries.h, now dg/geometries/geometries.h is preferred for unified access and easier recognition as a dg header file
- the header file/nc\_utilities.h, now dg/file/nc\_utilities.h is preferred for unified access and easier recognition as a dg header file

### Removed
    Removed for now removed features.

### Fixed
    Fixed for any bug fixes.

### Security
    Security in case of vulnerabilities.

    Added for new features.
    Changed for changes in existing functionality.

