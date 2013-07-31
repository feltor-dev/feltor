#error Documentation only
/*! @namespace dg 
 * @brief This is the namespace for all functions and 
 * classes defined and used by the discontinuous galerkin solvers.
 */
/*! 
 * @defgroup DG The Discontinuous Galerkin library
 *
 *  
 * @{
 *      @defgroup grid Grid objects
 *
 *          Objects that store topological information about the grid. Currently
 *          we only use equidistant grid in 1D and 2D. 
 *      @defgroup evaluation Function discretization
 *          
 *          The function discretisation routines compute the DG discretisation
 *          of analytic functions on a given grid. In 1D the discretisation
 *          simply consists of n function values per grid cell ( where n is the number
 *          of Legendre coefficients used; currently 1, 2, 3, 4 or 5) evaluated at
 *          the gaussian abscissas in the respective cell. In 2D we store all nxn 
 *          values per cell contiguously in memory. We choose x to be the contiguous direction.
 *          The first elements of the resulting vector lie in the cell at (x0,y0) and the last
 *          in (x1, y1).
 *          The expand functions are equivalent to a call to the respective evaluate
 *          function followed by a forward dlt transformation in each cell. (i.e. a multiplication 
 *          with DLT<n>::forward)
 *      @defgroup functions Functions and Functors
 *
 *          The functions are useful in the constructor of Operator objects. 
 *          The functors are useful for either vector transformations or
 *          as init functions in the evaluate or expand routines.
 *
 *      @defgroup creation Discrete derivatives and diagonal weight matrices
 *      @{
 *          @defgroup lowlevel Helper functions and classes
 *              Low level helper routines.
 *          @defgroup highlevel Creation functions
 *              High level matrix creation functions
 *      @}
 *      @defgroup blas Basic Linear Algebra Subprograms
 *
 *          These routines form the heart of our container free numerical algorithms. 
 *          They are called by all our numerical algorithms like conjugate gradient or 
 *          time integrators.
 *      @{
 *          @defgroup blas1 BLAS level 1 routines
 *              This group contains Vector-Vector operations.
 *          @defgroup blas2 BLAS level 2 routines
 *              This group contains Matrix-Vector operations.
 *      @}
 *      @defgroup algorithms Numerical schemes
 *          Numerical time integration and a conjugate gradient method.
 *      @defgroup utilities Utilities
 *          Utilities that might come in handy at some place or the other.
 * @}
 * 
 */
/*! @mainpage
 * Welcome to the DG library. 
 *
 * @par Design principles
 *
 * The DG library is built on top of the thrust and cusp libraries. 
 * Its intention is to provide easy to use
 * functions and objects needed for the integration of the 2D gyrofluid system with a
 * discontinuous galerkin method.  
 * Since it is build on top of thrust, code can run on a CPU as well as a GPU by simply 
 * switching between thrust's host_vector and device_vector. 
 * The DG library uses a design pattern also employed in the cusp library and other modern C++ codes. 
 * It might be referred to as container-free numerical algorithms. 
 *
 * @par Typical usage
 *
 * The typical usage of the library is as follows:
 * First you generate a grid object, which so far can only be a grid of equisized rectangles. 
 * It also contains information about the number of Legendre coefficients you want to use
 * per cell per grid dimension. 
 * Then you evaluate self-written functions on that grid to get a discretization of your 
 * initial conditions.
 * In the create namespace there are utility functions to create matrices which, when multiplied
 * with your previously generated vector, compute derivatives, etc. 
 * Multiplication, addition, etc. can be done with blas routines. 
 * There are several explicit Runge-Kutta and Adams-Bashforth methods implemented for time-integration. Moreover there is a conjugate - gradient method for the iterative solution of symmetric matrix 
 * equations.
 *
 *
 */
