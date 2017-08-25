#error Documentation only
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
 *             The general idea is to separate global communication from local parallelization and thus 
 *             readily reuse the existing, optimized library for the local part
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
 *             of Legendre coefficients used; currently 1, 2, 3, 4 or 5) evaluated at
 *             the Gaussian abscissas in the respective cell. In 2D and 3D we simply 
 *             use the product space. We choose x to be the contiguous direction.
 *             The first elements of the resulting vector lie in the cell at (x0,y0) and the last
 *             in (x1, y1).
 *         @defgroup highlevel create weights 
 *              overloads for the create::weights and create::inv_weights functions for all
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
 *     @defgroup fieldaligned Fieldaligned derivatives
 * @}
 * @defgroup templates Level 99: Template models
 * Documentation for template models
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
 /** @class hide_geometry
  * @tparam Geometry A type that is or derives from one of the abstract geometry base classes (e.g. aGeometry2d, aGeometry3d, aMPIGeometry2d, ...)
  * The functions dg::create::dx() and dg::create::dy() must be callable and return an instance convertible to the Matrix class. 
  * Furthermore dg::evaluate() must return an instance of the container class.
     as do calls to dg::create::weights() and dg::create::inv_weights()
  */

 /** @class hide_container_lvl1
  * @tparam container A data container class for which the blas1 functionality is overloaded. Also we assume that the type is copyable/assignable and has a swap member function. Currently this is one of 
  *   dg::HVec, dg::DVec, dg::MHVec or dg::MDVec
  */
 /** @class hide_matrix_container
  * @tparam Matrix A class for which the blas2 functions are callable in connection with the container class
  *  - dg::HMatrix with dg::HVec
  *  - dg::DMatrix with dg::DVec
  *  - dg::MHMatrix with dg::MHVec
  *  - dg::MDMatrix with dg::MDVec
  *
  * @tparam container A data container class for which the blas1 functionality is overloaded. Also we assume that the type is copyable/assignable and has a swap member function. Currently this is one of 
  *   dg::HVec, dg::DVec, dg::MHVec or dg::MDVec
  */

 /** @class hide_symmetric_op
 * @tparam SymmetricOp A class for which the blas2::symv(Matrix&, Vector1&, Vector2&) function is callable 
 with the container type as argument. Also, The functions %weights() and %precond() 
 need to be callable and return weights and the preconditioner for the conjugate 
 gradient method. The Operator is assumed to be linear and symmetric!
 @note you can make your own SymmetricOp by providing the member function void symv(const container&, container&);
  and specializing MatrixTraits with the SelfMadeMatrixTag as the matrix_category
  */

/**
 * @brief Struct that performs collective scatter and gather operations across processes
 * on distributed vectors using mpi
 *
 * In order to understand the issue you must first really(!) understand what 
 gather and scatter operations are, so grab pen and paper: 

 First we note that gather and scatter are most often used in the context
 of memory buffers. The buffer needs to be filled wih values (gather) or these
 values need to be written back into the original place (scatter).

 Gather: imagine a buffer vector w and a map that gives to every element in this vector w
 an index into a source vector v where the value of this element should be taken from
 i.e. \f$ w[i] = v[\text{idx}[i]] \f$ 
 Note that an index in the source vector can appear several times or not at all. 
 This is why the source vector w can have any size and even be smaller than w. 

 Scatter: imagine a buffer vector w and a map that gives to every element in the buffer w an
 index in a target vector v where this element should go to, 
 i.e. \f$ v[\text{idx}[i]] = w[i] \f$. This is ill-defined.
 Note again that an index in v can appear several times or never at all. 
 Then in our case we perform a reduction operation (we sum up all elements) beginning
 with 0 which remedies the defintion. 

Since it is a vector operation the gather and scatter operation can 
also be represented/defined by a matrix. The gather matrix is just a 
(permutation) matrix of 1's and 0's with exactly one "1" in each line.
In a "coo" formatted sparse matrix format the values array would consist only of "1"s, 
row array is just the index and column array is the gather map.
We uniquely define the corresponding scatter matrix as the transpose of the gather matrix. 
The scatter matrix can have zero, one or more "1"s in each line.
\f[ w = G v \\
    v = S w \f]

The scatter matrix S is the actual inverse of G if and only if the gather map is bijective.
In this case the buffer and the vector can swap their roles. 

Finally note that when v is filled with its indices, i.e. \f$ v[i] = i \f$, then
the gather operation will reproduce the index map in the buffer w \f$ w[i] = \text{idx}[i]\f$ .

 * @ingroup templates
 @attention this is not a real class it's there for documentation only
 */
struct aCommunicator
{

    /**
     * @brief Gather data across processes
     *
     * @param values data to gather from
     * @tparam LocalContainer a container on a shared memory system
     *
     * @return the buffer vector of size size()
     */
    template< class LocalContainer>
    LocalContainer global_gather( const LocalContainer& values)const;
    //actually the return type in NNC is const LocalContainer& 

    /**
     * @brief Scatters data accross processes and reduces on double indices
     *
     * @tparam LocalContainer a container on a shared memory system
     * @param toScatter buffer vector (has to be of size given by size())
     * @param values contains values from other processes sent back to the origin 
     */
    template< class LocalContainer>
    void global_scatter_reduce( const LocalContainer& toScatter, LocalContainer& values) const;

    /**
    * @brief The size of the local buffer = local map size
    *
 Consider that both the vector v and the buffer w are distributed across processes.
 In Feltor the vector v is distributed equally among processes and the local size
 of v is the same for all processes. However the buffer size might be different for each process. 
    * @note may return 0 to indicate identity between v and w and that no MPI communication is needed 
    * @note we assume that the vector size is always the local size of a dg::MPI_Vector
    * @return buffer size
    */
    unsigned size() const;
    /**
    * @brief The internal mpi communicator used 
    *
    * used to assert that communicators of matrix and vector are the same
    * @return MPI Communicator
    */
    MPI_Comm communicator() const;
};
