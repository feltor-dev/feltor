# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
We do not (yet) follow semantic versioning.

## [Unreleased]
### Added
- new member functions for DS
- adaptive timesteppers, notably through the dg::Adaptive and dg::ERKStep and dg::ARKStep classes
- dg::View vector view class usable in dg::blas1 and dg::blas2

### Changed
- interface for DS
- better test programs ds\_t.cu, ds\_curv\_t.cu and ds\_guenther\_t.cu
  including corresponding \*mpit.cu programs
- RK now named RungeKutta and order parameter not a template any more
- AB now named AdamsBashforth and order parameter not a template any more

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

