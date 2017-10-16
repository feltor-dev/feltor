#error Documentation only
/*! @mainpage
 * @section pdf PDF writeups
 * DON'T PANIC!
 *  - <a href="./dg_introduction.pdf" target="_blank">Introduction to dg methods</a>
 */
/*! @namespace dg 
 * @brief This is the namespace for all functions and 
 * classes defined and used by the discontinuous Galerkin solvers.
 */
/*! 
 * @defgroup backend Level 1: Vectors, Matrices and basic operations
 * @{
 *     @defgroup blas Basic Linear Algebra Subroutines
 *
 *         These routines form the heart of our container free numerical algorithms. 
 *         They are called by all our numerical algorithms like conjugate gradient or 
 *         time integrators.
 *     @{
 *         @defgroup blas1 BLAS level 1 routines
 *             This group contains Vector-Vector operations.
 *             Successive calls to blas routines are executed sequentially.
 *             A manual synchronization of threads or devices is never needed in an application 
 *             using these functions. All functions returning a value block until the value is ready.
 *         @defgroup blas2 BLAS level 2 routines
 *             This group contains Matrix-Vector operations.
 *             Successive calls to blas routines are executed sequentially.
 *             A manual synchronization of threads or devices is never needed in an application 
 *             using these functions. All functions returning a value block until the value is ready.
 *     @}
 *     @defgroup sparsematrix Sparse matrix formats
 *     @defgroup mpi_structures MPI backend functionality
 *             In this section the blas functions are implemented for the MPI+X hardware architectures, where X 
 *             is e.g. CPU, GPU, accelerator cards...
 *             The general idea to achieve this is to separate global communication from local computations and thus 
 *             readily reuse the existing, optimized library for the local part.
 *     @defgroup typedefs Typedefs
 *          Useful type definitions for easy programming
 * @}
 * @defgroup numerical0 Level 2: Basic numerical algorithms
 * These algorithms make use only of blas level 1 and 2 functions
 * @{
 *     @defgroup time Time integrators
 *     @defgroup invert Matrix inversion
 *     @defgroup root Root finding
 * @}
 * @defgroup geo Level 3: Topology and Geometry
 * @{
 *     @defgroup grid Topological grids and operations
 *
 *     Objects that store topological information (which point is neighbour of which other point) 
 *     about the grid. 
 *     @{
 *         @defgroup basictopology Topology base classes
 *         @defgroup evaluation evaluate
 *             
 *             The function discretisation routines compute the DG discretisation
 *             of analytic functions on a given grid. In 1D the discretisation
 *             simply consists of n function values per grid cell ( where n is the number
 *             of Legendre coefficients used; currently 1 <= n <= 20 ) evaluated at
 *             the Gaussian abscissas in the respective cell. In 2D and 3D we simply 
 *             use the product space. We choose x to be the contiguous direction.
 *             The first elements of the resulting vector lie in the cell at (x0,y0) and the last
 *             in (x1, y1).
 *         @defgroup highlevel create weights 
 *              overloads for the dg::create::weights and dg::create::inv_weights functions for all
 *              available topologies
 *         @defgroup creation create derivatives 
 *
 *             High level matrix creation functions
 *         @defgroup interpolation Interpolation and projection
 *         @defgroup scatter Scatter and Gather
 *     @}
 *     @defgroup geometry Geometric grids and operations
 *
 *        These routines form the heart of our geometry free numerical algorithms. 
 *        They are called by our geometric operators like the Poisson bracket. 
 *    @{
 *        @defgroup basicgeometry Geometry base classes
 *        @defgroup pullback pullback and pushforward
 *        @defgroup metric create volume
 *        @defgroup utilities Averaging
 *        @defgroup generators Grid Generator classes
 *            The classes to perform field line integration for DS and averaging classes
 *    @}
 * @}
 * @defgroup numerical1 Level 4: Advanced numerical schemes
 *
 * These routines make use of both the basic operations as well as the interfaces defined in the Geometry section.
 * @{
 *     @defgroup arakawa Discretization of Poisson bracket
 *     @defgroup matrixoperators Elliptic and Helmholtz operators
 * @}
 * @defgroup misc Level 00: Miscellaneous additions
 * @{
 *     @defgroup timer Timer class
 *     @defgroup functions Functions and Functors
 * 
 *         The functions are useful mainly in the constructor of Operator objects. 
 *         The functors are useful for either vector transformations or
 *         as init functions in the evaluate routines.
 *     @defgroup lowlevel Lowlevel helper functions and classes
 *         Low level helper routines.
 * @}
 * 
 */

/** @class hide_binary
  * @tparam BinaryOp A class or function type with a member/signature equivalent to
  *  - double operator()(double, double) const
  */
/** @class hide_ternary
  * @tparam TernaryOp A class or function type with a member/signature equivalent to
  *  - double operator()(double, double, double) const
  */

 /** @class hide_container
  * @tparam container 
  * A data container class for which the blas1 functionality is overloaded.
  * We assume that container is copyable/assignable and has a swap member function. 
  * Currently this is one of 
  *  - dg::HVec, dg::DVec, dg::MHVec or dg::MDVec  
  *  - std::vector<dg::HVec>, std::vector<dg::DVec>, std::vector<dg::MHVec> or std::vector<dg::MDVec> . 
  *
  */
 /** @class hide_matrix
  * @tparam Matrix 
  * A class for which the blas2 functions are callable in connection with the container class. 
  * The Matrix type can be one of:
  *  - container: A container acts as a  diagonal matrix. 
  *  - dg::HMatrix and dg::IHMatrix with dg::HVec or std::vector<dg::HVec>
  *  - dg::DMatrix and dg::IDMatrix with dg::DVec or std::vector<dg::DVec>
  *  - dg::MHMatrix with dg::MHVec or std::vector<dg::MHVec>
  *  - dg::MDMatrix with dg::MDVec or std::vector<dg::MDVec>
  *  - Any type that has the SelfMadeMatrixTag specified in a corresponding 
  *  MatrixTraits class (e.g. Elliptic). In this case only those blas2 functions 
  *  that have a corresponding member function in the Matrix class (e.g. symv( const container&, container&); ) can be called.
  *  If the container is a std::vector, then the Matrix is applied to each of the elements.
  */
  /** @class hide_geometry
  * @tparam Geometry 
  A type that is or derives from one of the abstract geometry base classes ( aGeometry2d, aGeometry3d, aMPIGeometry2d, ...). 
  */

  /** @class hide_container_geometry
  * @tparam container 
  * A data container class for which the blas1 functionality is overloaded and to which the return type of blas1::evaluate() can be converted. 
  * We assume that container is copyable/assignable and has a swap member function. 
  * In connection with Geometry this is one of 
  *  - dg::HVec, dg::DVec when Geometry is a shared memory geometry
  *  - dg::MHVec or dg::MDVec when Geometry is one of the MPI geometries
  * @tparam Geometry 
  A type that is or derives from one of the abstract geometry base classes ( aGeometry2d, aGeometry3d, aMPIGeometry2d, ...). Geometry determines which container type can be used.
  */

  /** @class hide_geometry_matrix_container
  * @tparam Geometry 
  A type that is or derives from one of the abstract geometry base classes ( aGeometry2d, aGeometry3d, aMPIGeometry2d, ...). Geometry determines which Matrix and container types can be used:
  * @tparam Matrix 
  * A class for which the blas2 functions are callable in connection with the container class and to which the return type of create::dx() can be converted. 
  * The Matrix type can be one of:
  *  - dg::HMatrix with dg::HVec and one of the shared memory geometries
  *  - dg::DMatrix with dg::DVec and one of the shared memory geometries
  *  - dg::MHMatrix with dg::MHVec and one of the MPI geometries
  *  - dg::MDMatrix with dg::MDVec and one of the MPI geometries
  * @tparam container 
  * A data container class for which the blas1 functionality is overloaded and to which the return type of blas1::evaluate() can be converted. 
  * We assume that container is copyable/assignable and has a swap member function. 
  * In connection with Geometry this is one of 
  *  - dg::HVec, dg::DVec when Geometry is a shared memory geometry
  *  - dg::MHVec or dg::MDVec when Geometry is one of the MPI geometries
  */

 /** @class hide_symmetric_op
 * @tparam SymmetricOp 
 A class for which the blas2::symv(Matrix&, Vector1&, Vector2&) function is callable 
 with the container type as argument. Also, The functions %inv_weights() and %precond() 
 need to be callable and return inverse weights and the preconditioner for the conjugate 
 gradient method. The Operator is assumed to be linear and symmetric!
 @note you can make your own SymmetricOp by providing the member function void symv(const container&, container&);
  and specializing MatrixTraits with the SelfMadeMatrixTag as the matrix_category
  */

/*!@addtogroup mpi_structures
@{
@page mpi_matrix MPI Vectors and the blas1 functions

In Feltor each mpi process gets an equally sized chunk of a vector.
The corresponding structure in FELTOR is the dg::MPI_Vector, which is 
nothing but a wrapper around any container type object and a MPI_Comm. 
With this the dg::blas1 functions can readily implemented by just redirecting to the
implementation for the container type. The only functions that need
communication are the dg::blas1::dot functions (MPI_Allreduce).

@page mpi_vector Row and column distributed matrices

Contrary to a vector
a matrix can be distributed in two ways, row-wise and column wise. 
The structure dg::MPIDistMat is a wrapper around a LocalMatrix type object 
and an instance of dg::aCommunicator
In a row-distributed matrix each process gets the complete 
rows of the matrix that correspond to the indices in the 
vector it holds. 
In a column-distributed matrix each process gets the complete 
columns of the matrix corresponding to the indices in the 
vector it holds. 
When we implement a matrix-vector multiplication the order 
of communication and computation depends on the distribution 
of the matrix.
For the row-distributed matrix each process first has to gather 
all elements of the input vector it needs to be able to compute the elements of the output. In general this requires MPI communication.
(read the documentation of dg::aCommunicator for more info of how global scatter/gather operations work).
Formally, the gather operation can be written as a matrix \f$G\f$
of \f$1'\f$s and \f$0'\f$s.
After the elements have been gathered into a buffer the local matrix-vector
multiplications can be executed.
\f[
M = R\cdot G
\f]
where \f$R\f$ is the row-distributed matrix with modified indices 
and \f$G\f$ is the gather matrix, in which the MPI-communication takes place.
The dg::RowColDistMat goes one step further and separates the matrix \f$ R\f$ into 
a part that can be computed entirely on the local process and a part that needs communication.

\section column 

In a column distributed matrix the local matrix-vector multiplication can be executed first because each processor already
has all vector elements it needs. 
However the resuling elements have to be communicated back to 
the process they belong to. Furthermore, a process has to sum
all elements it receives from other processes on the same
index. This is a scatter and reduce operation and
it can be written as a scatter matrix \f$S\f$ (s.a. dg::aCommunicator). The transpose
of the scatter matrix is a gather matrix and vice-versa.
\f[
M = S\cdot C
\f]
where \f$S\f$ is the scatter matrix and \f$C\f$ is the column distributed
matrix with modified indices. 

It turns out that a row-distributed matrix can be transposed
by transposition of the local matrices and the gather matrix (s.a. dg::transpose).
The result is then a column distributed matrix.
The transpose of a column distributed matrix is a row-distributed matrix and vice-versa.
@}
*/
