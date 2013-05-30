#error Documentation only
/*! @namespace dg 
 * @brief This is the namespace for all functions and 
 * classes defined and used by the discontinuous galerkin solvers.
 */
/*! 
 * @defgroup DG The Discontinuous Galerkin library
 * @{
 *      @defgroup grid Grid objects
 *      @defgroup evaluation Function discretization
 *      @defgroup creation Creation of discrete derivatives
 *      @defgroup blas Basic Linear Algebra Subprograms
 *      @{
 *          @defgroup blas1 BLAS level 1 routines
 *          @defgroup blas2 BLAS level 1 routines
 *      @}
 *      @defgroup algorithms Numerical schemes
 *      @defgroup utilities Utilities
 * @}
 * 
 */
/*! @mainpage
 * Welcome to the DG library. 
 *
 * The DG library is built on top of thrust and cusp. Its intention is to provide easy to use
 * functions and objects needed for the integration of the 2D gyrofluid system with a
 * discontinuous galerkin method.  
 * Since it is build on top of thrust, code can run on a CPU as well as a GPU by simply 
 * switching between thrust's host_vector and device_vector. 
 * The DG library uses a design pattern also employed in the cusp library and other modern C++ codes. 
 * It might be referred to as container-free numerical algorithms. This means that 
 * the actual numerical algorithm is written in terms of a reduced set of operations (vector addition
 * matrix-vector multiplication, etc.) which are templates of matrix and vector classes. 
 * The algorithm doesn't actually know how exaclty a vector lies in memory or how its elements should 
 * be added. 
 * In order to use an algorithm for a given Vector class, you simply have to specialize these
 * operations for this specific vector class. 
 *
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
 * equations. Thus far only diagonal preconditioners are usable which is enough if 
 * extrapolation of solutions from previous timesteps is used.
 *
 *
 */
