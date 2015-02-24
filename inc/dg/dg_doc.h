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
 *          the gaussian abscissas in the respective cell. In 2D and 3D we simply 
 *          use the product space. We choose x to be the contiguous direction.
 *          The first elements of the resulting vector lie in the cell at (x0,y0) and the last
 *          in (x1, y1).
 *      @defgroup functions Functions and Functors
 *
 *          The functions are useful mainly in the constructor of Operator objects. 
 *          The functors are useful for either vector transformations or
 *          as init functions in the evaluate routines.
 *
 *      @defgroup creation Discrete derivatives 
 *      @{
 *          @defgroup lowlevel Lowlevel helper functions and classes
 *              Low level helper routines.
 *          @defgroup highlevel Matrix creation functions and classes
 *              High level matrix creation functions
 *          @defgroup arakawa Arakawas scheme
 *          @defgroup matrixoperators Classes that act as matrices in blas2 routines
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
 *          Numerical time integration and a conjugate gradient method based
 *          solely on the use of blas routines
 *      @defgroup utilities Utilities
 *          Utilities that might come in handy at some place or the other.
 *      @{
 *          @defgroup scatter Utility functions for reorder operations on DG-formatted vectors
 *          @defgroup polarization Utility functions for C-style bindings of polarization solver
 *      @defgroup mpi_structures MPI backend functions
 *
 To use these funcions use code like:
@code
#include "dg.h"

int main()
{
    //allocate a workspace
    dg_workspace* w = dg_create_workspace( Nx, Ny, hx, hy, dg::DIR, dg::PER);
    //allocate chi, x and b
    double* chi = new double[Nx*Ny];
    double* x = new double[Nx*Ny];
    double* b = new double[Nx*Ny];
    ...//compute useful values for chi
    //assemble polarization matrix
    dg_update_polarizability( w, chi);
    ...//compute useful values for b and an initial guess for x
    //solve A(chi)*x = b to a precision of 1e-4
    dg_solve( w, x, b, 1e-4);
    //release resources
    dg_free_workspace( w);
    ...
    
    return 0;
}
@endcode
 *
 *      @}
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
 * functions and objects needed for the integration of the 2D and 3D gyrofluid system with a
 * discontinuous galerkin method.  
 * Since it is build on top of thrust and cusp, code can run on a CPU as well as a GPU by simply 
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
