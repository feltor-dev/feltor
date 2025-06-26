# Changelog
All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

We mark updates with a (pre-)release tag on
github.
> Creating a release on github will create an associated archived version
  on zenodo.

As of v8.0 we try to follow [Semantic versioning](https://semver.org/) i.e. major version number changes break backwards compatibility and minor version number changes indicate new features. Before that we kind of made up our own version numbers. A new major version number is often associated with a journal publication, but other than that there is no defined mapping from the amount or kind of change to a version number.

> Only changes in code are reported here, we do not track changes in the
> doxygen documentation, READMEs or tex writeups.
> As of v7.0 we also stop reporting changes in test and benchmark programs.
## [v8.2.1] Fix Clang compilation
### Changed
 - Update cccl dependency version from 2.8.0 to 2.8.4
### Fixed
 - Fix all warnings and errors from clang-19
 - Fix some remaining warnings from msvc
 - `FP_FAST_FMA` macro no longer used for fma support test except for GCC
### Removed
 - Revmove `DG_BENCHMARK` macro everywhere

## [v8.2] CMake integration
### Added
 - Complete cmake build system for all programs and libraries
 - Macro `VCL_NO_INCLUDE_PREFIX` can be used to include `vectorclass.h` directly
### Changed
 - matrix now depends on `lapack` directly, not `lapacke`
 - The timestamp in the file history now follows ISO8601 formatting in UTC
 - probes_params uses `std::vector` instead of `thrust::host_vector`
 - All remaining `*.cu` files are renamed to `*.cpp` 
 - Moved all diag programs to their respective src project
### Fixed
 - everything compiles without warnings from `-Wextra -Wpedantic` and `\W4` on Windows
 - Remove the use of variable length arrays (VLAs)
 - Change `uint` to `unsigned` in `mpi_kron.h`
 - Change iterator type in `dg::View` to given pointer type (else undefined construction of unkown iterator type from pointer)
 - `dg::is_divisable` ignored `eps` parameter
 - Avoid zero sized array warnings in `aRealMPITopology` for `Nd=0`
 - filesystem paths are given in `wchar` on windows; fix ability to read wchar
 - Fixed MPI file write error when get does not set value type 
 - Fix probes parser when using nlohmann-json
### Removed
 - `diag/compare.cpp` Removed because it can be trivially implemented in python
 - Boost dependency of geometries header completely removed using C++17 special functions, in particular in `taylor.h`
## [v8.1.1] Hotfix aliases
### Fixed
 -  Aliases in `dg::blas1::pointwiseDot` and `dg::blas1::pointwiseDivide` lead to wrong results
## [v8.1] Remove cusp dependency
### Added
 - New `dg::SparseMatrix` class to replace our previous `cusp::csr_matrix` and `cusp::coo_matrix`
 - The `SPARSELIB` Makefile variables expands to `-lcusparse` when compiling with cuda
 - resize method in `dg::TriDiagonal`
 - Writeable data method in `dg::SquareMatrix`
### Changed
 - The `LIBS` Makefile variable includes `-lcusparse` dependency when compiling with cuda. This is rather trivial because cuda ships with cusparse.
 - Default build should now use `https://github.com/nvidia/cccl` repository
 - Default build should now use `https://github.com/vectorclass/version2` (instead of version1)
 - All dg functions/classes/typedefs returning or depending on any cusp coo or csr matrix now use `dg::SparseMatrix`
 - All dg functions/classes returning or depending on `cusp::dia_matrix` now use `dg::TriDiagonal`
 - All dg functions/classes returning or depending on `cusp::array1d` now use `thrust::host_vector`
 - All dg functions/classes returning or depending on `cusp::array2d` now use `dg::SquareMatrix`
 - The inverse type of tridiagonal matrices in `dg/matrix/matrix.h` is `dg::SquareMatrix`
### Deprecated
 - `dg::InverseTriDiagonal` because untested and unused over `dg::mat::compute_Tinv_y`
### Removed
 - All cusp header files are removed from `dg/algorithm.h`, `dg/geometries/geometries.h` and `dg/matrix/matrix.h` (The relevant tensor traits for cusp can still manually be included though)
### Fixed
 - Thrust library can now be used at newest version

## [v8.0] Improved foundations
### Added
 - A novel C++17 style `dg::file::NcFile` class together with its MPI counterpart and supporting infrastructure that greatly simplifies netCDF-4 output
 - Add modular `dg::file::Probes` class with default `dg::file::parse_probes` that can be added to any project
 - Simplify output design with `dg::file::Record`
 - Add possibility to use `nlohmann::json` instead of `jsoncpp` in our WrappedJsonValue using the macro `DG_USE_JSONHPP`
 - New blas1 function `dg::blas1::kronecker` for Nd dimensional product space evaluations
 - New blas1 function `dg::blas1::vdot` that can handle general and in particular complex scalar products
 - Novel N-dimensional Grid class `dg::aRealTopology` and `dg::RealGrid` (and its MPI counterparts) that generalize the previous 1d, 2d and 3d grids. Previous grid classes still fully function using typedefs.
 - Use catch2 as a unit test framework for `dg`, `dg/file` and `dg/topology`. We introduce `tests/run-all-tests` script
 - A novel MPI communication backend allowing for consistent overlapping communication and computation in all sparse matrices
 - `dg::mpi_cart_xxx` with `xxx` in `create`, `kron`, `split` for more consistent and easier Cartesian communicator handling in MPI grids
 - `dg::mpi_read*` functions for more consistent and easier MPI initialisation
 - Improved handling of cuda-aware MPI detection for non-OpenMPI libraries
 - Add default constructors to most classes notably Grid classes
 - Add optional weights parameter to `dg::least_squares`
 - Add experimental `extrapolate_least_squares` method to `dg::Extrapolate` (unfortunately not very successful)
 - Template version of `dg::is_same`
 - Add more verbose output to `SeparatrixOrthogonal`
 - Add `dg::mat::BesselJ` and `dg::mat::LaguerreL` and `dg::mat::GyroLagK` functors
 - Add `dg::exblas::fpedot_cpu`, `dg::exblas::fpedot_gpu` and `dg::exblas::fpedot_omp`
 - Add `dg::exblas::ufloat`
 - Probes module in feltor 3d code
 - More general interfaces working with new N dimensional grid class to all topology functions like evaluate, interpolate, derivatives etc.
 - `dg::cHVec`, `dg::cDVec`, `dg::cMHVec` and `dg::cMDVec` typedefs for complex vector types
 - Expose `dg::is_scalar_v`, `dg::is_vector_v`, `dg::has_policy_v` and `dg::is_matrix_v` type predicates
 - Extended Tag system on scalar types in particular complex types (`std::complex` and `thrust::complex`)
 - Add `dg::geo::SOLRegion` and `dg::geo::ClosedFieldlineRegion` as predicates
 - Add predicate parameter to `dg::geo::WallFieldlineDistance` and `dg::geo::WallFieldlineCoordinate`, which may accelerate sheath generation by  a factor 4
 - Add double X-point switch in `dg::geo::createModifiedField` as "sol_pfr_2X"
### Changed
 - **std=c++17** Change C++ standard to C++-17
 - `dg::Average` has **one additional template parameter** (the interpolation matrix). Changed in all occurences in src programs.
 - `dg::EllSparseBlockMat` and `dg::CooSparseBlockMat` got one additional template parameter (but does not matter if used through `dg::HMatrix` typedef)
 - Bump Doxyfile(s) to 1.9.8
 - `dg::mpi_init` now also sets GPU
 - `dg/geometries/geometries.h` now **always includes `dg/file/json_utilities.h`**
 - The main documentation page is changed to `topics.html` from `modules.html`
 - Complete rework of MPI sparse matrix communication backend
 - Most `*.cu` file endings were changed to `*.cpp` (Nicer for C++ automatic syntax highlighting)
 - Most `inc/*_mpi*.cpp` programs were merged into its corresponding `inc/*.cpp` program (reduces future maintenance load)
 - Changed `static inline` function declarations to `inline`
 - MPI grid classes can handle arbitrary parallelization (even those that do not evenly divide cell numbers)
 - `dg::geo::DS` class no longer adds jump terms in its symv method. Thus **removed one template parameter** from `dg::geo::DS` class. Adapted in all src prorams.
 - `SeparatrixOrthogonal` class only integrates with `1e-11` default accuracy (prior `1e-13`)
 - `dg/geometries/geometries.h` now automatically incurs `json` dependency
 - The `tridiag` method of `dg::mat::UniversalLanczos` now returns the tridagonal matrix T
 - Restructure feltor, feltorSH, feltorSHp, feltorSesol, feltorShw, toefl, lamb-dipole, geometry-diag programs using new I/O design
 - Use WrappedJsonValue in poet, esol, hasegawa and heat code
 - Use macro `WITH_MPI` in most executable programs for easy compiler switch
 - Move `ping_mpit.cu` program to `src/ping/mpi-ping.cpp`
 - Changed x,y,z coordinate to R,Z,P in geometry-diag
 - Add capability for 2nd X-point in geometry-diag
 - Add ClosedFieldlineRegion and SOL output in geometry-diag
 - Major code refactoring of 3d feltor code in preparation of thermal code
 - Member functions of `dg::DLT` are now static
 - `dg::Operator` is now named `dg::SquareMatrix` and can be used in `dg::blas2::symv` functions together with `std::vector` (`dg::Operator` typedef still exists for backward compatibility)
 - `dg::lu_solve` is now exposed in main dg namespace
 - Allow general value types in all blas1 functions
 - `dg::zero` and `dg::one` have more general template interface
 - `dg::blas1::dot` and `dg::blas2::dot` allow complex input vectors (in fact they allow any addable type)
 - `dg::MPI_Vector::set_communicator` member only needs one `MPI_Comm` parameter
### Deprecated
 - All previous netcdf utilitiy functions are now deprecated
 - `dg::mpi_init1d`, `dg::mpi_init2d` and `dg::mpi_init3d` and all overloads
### Removed
 - `dg::EllSparseBlockMatDevice` and `dg::CooSparseBlockMatDevice` now merged into `dg::EllSparseBlockMat` and `dg::CooSparseBlockMat`
 - `dg::sainv_precond` as it is not used
 - `device=cpp` is no longer supported. Use `device=cpu`.
 - Last remains of `DG_DEBUG` and `SILENT` macros completely removed
 - doxygen-awesome css and js files. We now use doxygen-awesome as external library instead of keeping copies in `feltor/doc`
 - `dg::average` and `dg::mpi_average` (now replaced by `dg::create::reduction`)
 - `dg::extend_line` and `dg::extend_column (now replaced by `dg::create::prolongation`)
 - `diag/probes.h` is now replaced by dg/file/probes.h
 - Some low level operator generators for dg derivatives
 - fehlberg-4-2-3, billington-3-3-2, trbdf2-3-3-2 Butcher tableaus as they performed very badly and should not be used
 - `dg::transpose`
 - `communicator_mod` and `communicator_mod_reduce` members of `dg::MPI_Vector`
 - All previous MPI communication objects
 - `dg::Buffer` memory class
### Fixed
 - Fix unnecessary copy in Adaptive and SingleStep timeloop
 - Fix Broadcase in `mpi_init3d` goes to comm not to world
 - Fix value init in default constructor of `dg::MPI_Vector`
 - Fix bug in double version of `dg::is_same`
 - Fix sign in `SafetyFactor`
 - Fix grid generation close to O-point
 - Fix verbose output in SeparatrixOrthogonal
 - Performance of serial sparseblock matrix now equals that of OpenMP with 1 thread

## [v7.0] Three-dimensional
### Added
 - A left looking sparse inverse preconditioner `dg::sainv_precond` (sadly did not yield any benefits over `dg::nested_iterations` in our tests so we did not implement an MPI version)
 - `dg::blas2::parallel_for` for general parallelization tasks on shared memory containers (very close in functionality to `kokkos::parallel_for`
 - `dg::blas2::stencil` A generalization of `dg::blas2::parallel_for` that works for MPI parallelization at the cost of having to encode the communication stencil in a sparse matrix and only one vector argument
 - `dg::create::window_stencil` and `dg::create::limiter_stencil` overloaded for various topologies for use as communication stencils in `dg::blas2::stencil`
 - Functors `dg::CSRSlopeLimiter`, `dg::CSRMedianFilter`, `dg::CSRSWMFilter`, `dg::CSRAverageFilter`, `dg::CSRSymvFilter` for use as functors in `dg::blas2::stencil`
 - `isLocalBijective` method in `dg::SurjectiveComm`
 - write access to `inner_matrix` and `outer_matrix` in `dg::MPI_Matrix`
 - `MPI_Comm` parameter in constructor of `dg::NearestNeighborComm`
 - `set_right_size` and `set_left_size` methods in `dg::EllSarseBlockMat`
 - `dg::Elliptic1d` (since we have 2d and 3d versions)
 - `dg::Sign` a 1d functor that returns the sign of its argument
 - `dg::GeneralHelmholtz` generalizes all the Helmholtz classes into one class (and specializes to `dg::Helmholtz1d`, `dg::Helmholtz`, `dg::Helmholtz2d`, `dg::Helmholtz3d`
 - Experimental `dg::FilteredERKStep` class (sadly does not work so well)
 - `n` method to access number of polynomial coefficients in `dg::DLT`
 - `dg::create::fast_transform` for creating block diagonal matrices on various topologies
 - overloads for `dg::create::fast_interpolation` and `dg::create::fast_projection` to select directions
 - new fem functionality: `dg::create::fem_linear2const`, `dg::create::inv_fem_mass` `dg::create::inv_fem_linear2const`, `dg::create::fem_mass2d`, `dg::create::inv_fem_mass2d`, `dg::create::fem_linear2const2d`, `dg::create::inv_fem_linear2const2d`
 -  `dg::create::interpolation` and `dg::create::projection` functions now all have a `method` parameter (that defaults to "dg")
 - `dg::invert` overload acting on `dg::Operator` as an alias to `dg::create::inverse`
 - `dg::create::backproject` and `dg::create::inv_backproject` for projecting a dG grid vector to an equidistant grid and back
 - `dg::create::inv_backscatter` for inverting an interpolation to an equidistant grid
 - `dg::convertGlobal2LocalRows` and `dg::convertLocal2GlobalCols` helper functions to enable mpi conversion of column distributed matrices
 - `access_string` method in `dg::file::WrappedJsonValue` to access creation history of an object
 - Restructure of `dg::geo::Fieldaligned` to allow for different interpolation and backprojection from dG to fine grid and back (to a possibly equidistant grid and with possibly cubic, linear or nearest methods)
 - in `dg::geo::DS` the derivative methods now have the general zero transform
 - `dg::geo::createCircularField` has now two radii parameter to allow for the creation of an ellipsis and changed the sign
 - `dg::geo::SeparatrixOrthogonalAdaptor` to create Separatrix grids without XGrid topology
 - Utilities for exponential integration: `dg::mat::ExponentialStep`,  `dg::mat::ExponentialERKStep`, `dg::mat::phi1`, `dg::mat::phi2`, `dg::mat::phi3`, `dg::mat::phi4`, `dg::mat::FuncionalButcherTableau`, `dg::mat::ConvertsToFunctionalButcherTableau` with the four methods "Euler", "Midpoint-2-2", "Runge-Kutta-4-4", and "Hochbruck-3-3-4"
 - Utility function `dg::mat::MatrixFunction` for easy general purpose use
 - Test of filters and Limiters and implicit timesteppers in the `src/lamb_dipole` project
 - Test of new schemes in `src/navier_stokes` project
 - File `src/toefl/input/default_hpc.json` to be consistent with Quick Start guide
### Changed
 - explicit use of "default" constructors in `dg::Adaptive`, `dg::Advection`, `dg::AndersonAcceleration`, `dg::ArakawaX`, `dg::BijectiveComm`, `dg::GeneralComm`, `dg::BICGSTABl`, `dg::ChebyshevIteration`, `dg::EVE`, `dg::LGMRES`, `dg::MultigridCG2d`, `dg::PCG`, `dg::Poisson`, `dg::ERKStep`, `dg::SinglestepTimeloop`, `dg::ButcherTableau`, `dg::ShuOSherTableau`, `dg::RealGrid1d`, `dg::Operator`
 - `dg::create::modal_filter` no longer returns a `dg::MultiMatrix` directly. Instead, a `dg::Operator` is returned that can be converted to `dg::MultiMatrix` using `dg::create::fast_transform`
 - rename `getLocalGatherMap` to `getLocalIndexMap` in `dg::SurjectiveComm` and `dg::GeneralComm`
 - rename `getPidGatherMap` to `getPidIndexMap` in `dg::SurjectiveComm` and `dg::GeneralComm`
 - rename `getSortedGatherMap` to `getSortedIndexMap` in `dg::SurjectiveComm`
 - Constructors of `dg::Helmholtz`, and `dg::Helmholtz3d` change the orders of parameter `alpha` and reverts to constructor of respective `dg::Elliptic` class
 - 1d overloads for `dg::create::fast_interpolation` and `dg::create::fast_projection` return `dg::HMatrix_t` instead of `dg::MultiMatrix`
 - `dg::create::fem_mass` now returns `dg::TriDiagonal` instead of `cusp::coo_matrix`
 - `dg::create::precond` overload for `dg::DLT` renamed to `dg::create::inv_weights`
 - parameters to `dg::tensorproduct` changed from `cusp::coo_matrix` to `cusp::csr_matrix`
 - all appearances of `guenther` or `Guenther` in `dg::geo` are corrected to `guenter` respectively `Guenter` (because that is how her name is spelled)
 - `clone` and other virtual methods in `dg::geo::RealCurvilinearGridX2d`, `dg::geo::RealCurvilinearProductGridX3d`, `dg::geo::DSPGenerator`, `dg::geo::FluxGenerator`, `dg::geo::Hector`, `dg::geo::RealCurvilinearProductMPIGrid3d`, `dg::geo::PolarGenerator`, `dg::geo::LogPolarGenerator`, `dg::geo::Ribeiro`, `dg::geo::RibeiroX`, `dg::geo::SimpleOrthogonalX`, `dg::geo::SeparatrixOrthogonal`, `dg::geo::SimpleOrthogonal` have `override final` keywords
 - default parameters changed in `dg::geo::Fieldaligned` constructor
 - restructure of the `src/feltor` project (including renaming `*.cu` files to `*.cpp`)
### Deprecated
 - `dg::geo::createModifiedSolovevField` is superseded by `dg::geo::make_field`
### Removed
 - `dg::ModalFilter` is removed in favor of generating and using `dg::MultiMatrix` directly
 - `dg::transpose_dispatch` (replaced by `dg::blas2::parallel_for`)
 - `dg::extend_line` (replaced by `dg::blas2::parallel_for`)
 - `dg::extend_column` (replaced by `dg::blas2::parallel_for`)
 - `dg::create::interpolationT` (is not used anywhere)
 - file `src/toefl/toefl_hpc.cu` (was previously merged into `toefl.cpp`
 - `dg::integrate` replaced by `dg::aTimeloop` functionality
 - `dg::integrateERK` replaced by `dg::aTimeloop` functionality
 - `dg::stepperRK` replaced by `dg::aTimeloop` functionality
### Fixed
 - Fix left shift in exblas::AccumulateWord by -1 by explicit conversion to unsigned
 - Fix `isCommunicating` in `dg::BijectiveComm`
 - Fix `global_gather` and `global_scatter_reduce` in `dg::aCommunicator` in the case that a map is not communicating but still needs to locally gather values. This makes the two functions truly the transpose of each other.
 - Fix `dg::ShuOsher` copyable method segmentation fault in case of default constructed
 - Fix size method in `dg::Operator` for integer value types
 - Fix convergence criterion in fieldline integrator for `dg::geo::SimpleOrthogonal`
 - Fix constructor of `dg::mat::PolCharge` and `dg::mat::PolChargeN`


## [v6.0] More modular
### Added
 - Add ability to use lambdas and functors without `dg::TensorTraits` in `dg::apply`, `dg::blas2::symv` and `dg::blas2::gemv` (Extremely useful!)
 - Add `--extended-lambda` flag in default `NVCCFLAGS`
 - New class `dg::file::WrappedJsonValue` plus test file; allows easy bug-tracing on jsoncpp file input
 - Allow arbitrary `nx, ny, nz` polynomial number in all grids; adapt all weights, derivatives, etc. creation accordingly
 - New sub-project `dg/matrix/matrix.h` written by Markus and optimized by Matthias. Allows matrix-function computations in particular square roots
 - `CONTRIBUTING.md` file (moved from homepage)
 - `LAPACKLIB` variable in Makefile configuration
 - new `device=cpu` configuration (enable to compile without parallelization)
 - new file `config/version.mk` provides git hash and branch information for file output
 - new wall and sheath generation capabilities in `inc/geometries`
 - new `dg::aTimeloop` family of classes to help create time-stepping
 - new explicit timesteppers Cavaglieri-3-1-3 (explicit and implicit), Fehlberg-3-2-3, Fehlberg-4-2-3, Cavaglieri-4-2-3 (explicit and implicit), Tsitouras09/11-7-4-5, Verner-9-5-6, Verner-10-6-7, Dormand-Prince-13-7-8,
 - new implicit timesteppers SDIRK-4-2-3, Sanchez-3-3, Sanchez-3-4, Sanchez-6-5, Sanchez-7-6
 - Add vector support for `std::map` in `blas1` functions
 - Add UnaryOp parameter to `dg::blas1::reduce`
 - Add DenseMatrix-Vector multiplication suppoert in `dg::blas2::symv`
 - Add append method to `dg::Message`
 - Add `dg::abort_program` function
 - Add more cuda Error management in backend functions
 - Add convenience `dg::mpi_init` function
 - Matrix conversion from `dg::EllSparseBlockMat` to `cusp::coo_matrix` and associated `dg::tensorproduct` and `dg::create::diagonal` function to re-create elliptic operators
 - Add convenience mpi Macro `DG_RANK0` helping to write shared/mpi programs more easily
 - Add new multigrid implementation `dg::nested_iterations` for maximum modularity, sovlers can be separately chosen at each stage
 - Add experimental and preliminary FEM support
 - Add `dg::least_squares` function
 - project `esol` Edge-SOL turbulence in Full-F models
 - project `poet` Full-F Full-K blob simulations
 - project `navier_stokes` 3d field-aligned Navier-Stokes equations

### Changed
 - Redesign **implicit and semi-implicit timestepper interfaces** for more modularity and the ability to use lambdas
 - Redesign `dg::Adaptive`
 - Unify stopping criterions for Matrix inversion classes and use of weighted scalar product
 - `dg::apply`, `dg::blas2::symv` and `dg::blas2::gemv` are now formally the same
 - `dg::PCG` works with **self-adjoint matrices** in weights instead of symmetric ones
 - Renamed `dg::lr_pivot` to `dg::lu_pivot` and make it use binary reproducible dot functions, same for `dg::lu_solve`
 - Rename `dg::create::invert` to `dg::creat::inverse`
 - Redesign project `src/toefl`; follows standard design on user-guide
 - Redesign project `src/reco2D`; follows standard design on user-guide
 - Redesign project `src/lamb_dipole`; adapt to `dg::file::WrappedJsonValue`
 - Adapt all `src` projects to new semi-implicit timestepper interface
 - Redesign DS and all related projects
 - Rename `inc/geometries/magnetic_field_t.cu` to `inc/geometries/solovev_t.cu`

### Deprecated
 - `dg::integrateERK` replaced by `dg::aTimeloop` functionality
 - `dg::stepperRK` replaced by `dg::aTimeloop` functionality

### Removed
 - `dg::FilteredImplicitMultistep`; barely any theory behind that
 - `dg::Karniadakis` was marked deprecated before
 - "Verner-8-5-6" Timestepper tableau
 - `dg::norm` ; **Everything is normed now**;  adapt all codes to it; specifically also makes `inv_weights` and `inv_volume` functions unnecessary; matrix inversion works with weighted scalar products
 - class `dg::Invert`; replaced by lambda functionality
 - file `diag/impRdiag.cu`; replaced by python diagnostics
 - file `diag/reco2Ddiag.cu`; replaced by python diagnostics
 - file `diag/toeflEPdiag.cu`; replaced by python diagnostics
 - file `diag/toeflRdiag.cu`; replaced by python diagnostics
 - file `diag/vmaxnc.cu`; replaced by python diagnostics
 - file `diag/window_params.txt`; Useless
 - project `src/ep`; merged and oursourced with `impurities` project
 - project `src/polar`; Useless
 - project `src/impurities`; outsourced to `https://github.com/mwiesenberger/impurities`
 - `nc_enddef` and `nc_redef` calls; they are unnecessary according to netcdf documentation
 - all files `inc/geometries/[*magnetic-geometry.js*]` These are now part of a separate magnetic field repository
 - `hector_t.cu`, `ribeiro_t.cu`, `simple_orthogonal_t.cu` merged into `flux_t.cu`

### Fixed
 - fix bug from linking `-lhdf5` by linking `-lhdf5_serial` in default.mk
 - fix bug nvcc did not compile `.cpp` programs now it does
 - fix bugs in `dg::AndersonAcceleration`
 - fix bugs in `dg::LGMRES`
 - fix bugs in `dg::BICGSTABl`
 - fix bug in `exblas/accumulate.cuh` and `exblas/accumulate.h`
 - fix bug in `dg::ClonePtr` copy-constructor
 - fix bug symv call to CooMatrix in `mpi_matrix.h`
 - fix NaN bugs in `dg::blas2::symv` even though `beta == 0`
 - fix nvcc compiler warnings when using host lambdas in `dg::blas1` functions
 - fix tableau Tsitouras11-7-4-5

## [v5.2] More Multistep
### Added
 - M100 config file
 - json utility functions `dg::file::get, dg::file::get_idx` in `dg/file/json_utilities.h` which adds a small abstraction layer that gives a user more control over what happens if a variable is not found
 - json utility functions `dg::file::file2Json`, and `dg::file::string2Json` in json_utilities.h which adds a small abstraction layer that gives a user more control over what happens if an error happens during the parsing of a file
 - "easy output" netcdf utility functions that are particularly useful for MPI output: either write data in parallel or funnel through the master thread
 - new include files `dg/file/file.h`, `dg/file/json_utilities.h` and `dg/exblas/exblas.h`
 - new class `dg::Gradient` for gradient
 - new class `dg::Advection` for the upwind advection scheme
 - new `dg::blas1::reduce` function for custom reductions
 - new "exchangeable" `dg::x::DVec`, `dg::x::HVec`, ..., `dg::x::CartesianGrid2d`, ..., `dg::x::IHMatrix`, ... typedefs. The idea is that these resolve to either shared memory or mpi distributed memory versions depending on the MPI_VERSION macro. This helps merging shared and mpi programs into single ones.
 - added "simple" mode to `dg::Average` computation, which is beneficial for GPU computing
 - add `dg::integrate` that computes an indefinite integral of a function (essentially the opposite of the derivative)
 - add` dg::ModalFilter` and tests thereof
 - new `dg::compose` function that concatenates two or more functors to one
 - add `dg::cooRZP2X` coordinate transformation functions to easily transform between Cylindrical and Cartesian coordinates
 - interpolate function has an additional `dg::space` parameter to indicate nodal or modal values
 - Grid classes now have `host_vector` and `host_grid` member typedefs
 - new tensor functions `dg::tensor::scalar_product2d` and `dg::scalar_product3d` that can compute uE2 in one go
 - new extended tensor functions `dg::tensor::multiply2d` and `dg::multiply3d` that can compute uE2 in one go
 - new single step timestepper `dg::ShuOsher` including new `dg::ShuOsherTableau` and `dg::ConvertsToShuOsherTableau` classes to hold corresponding coefficients
 - new ShuOsher tableaus SSPRK
 - new Runge Kutta embedded tableaus tsitouras, the default timesteppers in Julia
 - new implicit RK tableau trapezoidal, implicit midpoint and sdirk-2-1-2
 - new class Simpsons that implements Simpsons rule for (time) integration
 - new implicit timesteppers `dg::DIRKStep` and `dg::ImplicitRungeKutta`
 - Redesign of multistep time steppers consistent with Runge-Kutta ones in terms of `dg::MultistepTableau` and `dg::ConvertsToMultistepTableau`
 - a host of new explicit, implicit and semi-implicit multistep tableaus
 - experimental "filtered" multistep time-steppers that allow modal filtering (first tests are not promising though)
 - new experimental multigrid solvers involving Chebyshev iterations as smoother (but none are better than nested iterations so they remain experimental)
 - new class `dg::EVE` that computes the largest Eigenvalue of a matrix (courtesy of Eduard Reiter)
 - new class `dg::ChebyshevIterations` and `dg::ChebyshevPreconditioner` (for chebyshev iterations)
 - new solvers `dg::LGMRES`, `dg::BICGSTABL`, and d`g::AndersonAcceleration` (courtesy of Aslak Poulsen)
 - new `dg::FixedPointSolver` and `dg::AndersonSolver` for nonlinear problems in time
 - a host of new functors for the evaluate and pullback functions
 - `dg::geo::FluxSurfaceIntegral`, `dg::geo::FluxVolumeIntegral` and `dg::geo::SafetyFactorAverage` classes
 - new implementation: `dg::geo::ds_centered_bc_along_field` and `dg::geo::dss_centered_bc_along_field` that implement boundary condition "Stegmeir" style along the magnetic field lines
 - new Fieldaligned member functions `integrate_between_coarse_grid` and `interpolate_from_coarse_grid` that allow field-aligned interpolations
 - `dg::geo::Periodify` class and `dg::geo::periodify` function to extend flux-functions periodically beyond grid boundaries
 - new `dg::geo::findCriticalPoint` function that generalizes X-point and O-point identification
 - new classes `dg::geo::SquareNorm` and `dg::geo::ScalarProduct` that work on cylindrical vector fields
 - new set utility functors `dg::geo::SetUnion`, `dg::geo::SetIntersection`, and `dg::geo::SetNot` that help construct damping regions
 - `dg::geo::createMagneticField` and `dg::geo::createModifiedField` with associated utility functions and classes that generalize the creation of magnetic flux functions and wall and sheath regions
 - new polynomial expansion and associated `dg::Horner2d` functor for magnetic flux functions that can in particular approximate any experimental equilibrium
 - new equilibrium, modifier and description fields for tokamak magnetic fields
 - Sign reversal of magnetic field and associated flux functions is now possible
 - new utility `dg::geo::createAlignmentTensor` and `dg::geo::createProjectionTensor` with respect to the magnetic unit vector
### Changed
 - namespace file changed to **dg::file** and exblas changed to **dg::exblas** (for consistency reasons, everything should go into the dg namespace, which in particular reduces the chance for name-clashes to just one, namely 'dg')
 - Moved **variation** member function into **dg::Elliptic** (previously in ArakawaX and Poisson)
 - **std=c++14** We use the C++-14 standard now (previously 11)
 - vectorclass dependency changed to vectorclass/version1 (previously we used a custom upload on feltor-dev repository)
 - default cuda compute capability bumped to sm-61 (previously sm-35)
 - marconi config now uses jsoncpp module (previously manually installed)
 - `dg::blas1::dot` and `dg::blas2::dot` and corresponding exblas functions now detect NaN and Inf errors
 - `dg::blas1::dot` and `dg::blas2::dot` now both do not accumulate rest of multiplication (inconsistent before)
 - All blas1 functions that do not read or alias their result vector now overwrite NaN and Inf
 - all our mpi communications on GPUs now fall-back to host2host communication for cuda-unaware mpi-installations
 - swapped input and output parameters in `dg::blas1::evaluate` first subroutine
 - the fast_interpolation and fast_projection functions now can also double / divide the polynomial coefficient consistent with the grids
 - change `shift_topologic()` shared RealTopology member functions to `shift()` and have an additional `negative` parameter that indicates sign swaps
 - clarify and unify the behaviour of the interpolation functions when points lie outside the grid boundaries
 - split and join functions have an additional real_type template parameter
 - improved stopping criterion for `dg::bisection1d` function
 - implicit time steppers give access to solver
 - multistep time-stepper now initialize with Runge-Kutta timesteppers of corresponding order
 - Multigrid nested iteration algorithm now allows accuracies for each stage separately (which can give a significant speed-up)
 - `dg::inverse( bc)` function is now a free-standing function to invert a boundary condition
 - `dg::Elliptic` classes now have `jump_weighting` member function
 - `dg::CG` operator now has a `test-frequency` parameter to control the number of times the error condition is evaluated
 - `dg::Extrapolation` class now has a `derive` member function to interpolate the derivative of the interpolating polynomial
 - Adapt all src and diag projects to changed file and json utilities and the moved variation member
 - Rename all input files with correct json file-ending
 - Complete redesign of src/feltor and src/lamb_dipole
 - Merge toefl_hpc with old toefl_mpi program
 - bump Doxygen version to 1.8.17
 - DS forward, backward, centered and dss functions are now free-standing, only requiring a fielaligned object, plus, and minus applications (this allows to reduce the number of times the plus and minus interpolation has to be applied)
 - changed Fieldaligned members hp_inv to hbp
 - changed name `dg::forward_transform` function (previously `dg::create::forward_transform`)
 - new `dg::geo::MagneticFieldParameters` struct to unify the representation of Meta-data in the `dg::geo::TokamakMagneticField` class (simplifies construction)

### Deprecated
 - `dg::Karniadakis` time-stepper is now superceded by the `dg::ImExMultistep` class
### Removed
 - remove diag/feltordiag.cu
 - remove dg::MemoryTraits and associated dimensionality and memory_category traits in favor of direct host_vector and host_grid typedefs in topology classes
 - old txt input files
 - `dg::geo::DeltaFunction` and `dg::geo::Alpha` for the computation of flux-surface averages no longer needed
 - dg::blas1::transfer (previously marked deprecated)
### Fixed
 - Fix bug: race condition in `dg::blas1::dot` and `dg::blas2::dot` on GPUs that led to hard to reproduce and seemingly unreasonable crashes
 - Fix bug: std namespace in diag/probes.h
 - Fix bug: const in `exblas::cpu::get_element`
 - Fix bug: correct  indices in `exblas::cpu::make_vcl_vec8d`
 - Fix bug: infinite creation of MPI communicators in `exblas::mpi_reduce_communicator` . Lead to MPI crashes due to memory overflow.
 - Fix bug: correct capture of cuda-aware mpi in configuration
 - Fix bug: test for no-communication in mpi_communicator.h (indicated false positives)
 - Fix bug: coefficient and initialization in `dg::Extrpolate`
 - Fix bug: Fpsi safety-factor in case nan is encountered still works
 - Fix bug: Fpsi safety-factor works up to the O-point
 - Fix bug: `dg::pushForwardPerp` on functors computed wrong result (only affects `dg::geo::Hector`)
 - Fix bug(s): several bugs in `dg::geo::Hector` which computed wrong grid (happened probably when we changed the grid design to polymorphic)
 - Fix bug: in perpendicular grid of MPI Curvlinear grid
 - Fix bug: missing direction initialization in mpi fieldaligned class
 - Fix bug: host mpi code compiled with nvcc
 - Fix bug: non-zero parallel boundary condition in mpi fieldaligned
 - Fix bug: GPU symv on X-point grids

## [v5.1] Adaptive Timesteppers
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
- cuda-aware mpi no longer a requirement, fall-back to traditional mpi implemented
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
- separate modification of fluxfunctions into mod namespace that works on flux functions in general (previously only solovev)

### Deprecated
- dg::blas1::transfer (replaced by the more general dg::assign and dg::construct)
- the header geometries/geometries.h (now **dg/geometries/geometries.h** is
  preferred for unified access and easier recognition as a dg header file)
- the header file/nc\_utilities.h ( now **dg/file/nc_utilities.h** is preferred for
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

