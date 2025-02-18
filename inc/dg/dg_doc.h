#error Documentation only
/*! @namespace dg
 * @brief This is the namespace for all functions and
 * classes defined and used by the discontinuous Galerkin library.
 */
/*!
 * @defgroup level1 Level 1: Vectors, Matrices and basic operations
 * @{
 *    @defgroup blas1 BLAS level 1 routines: Vector-Vector
 *    @brief \f$ f( x_{0i}, x_{1i}, x_{2i}, ...) \f$ and \f$ x^T y\f$
 *
 *      @note Successive calls to blas routines are executed sequentially
 *      @note A manual synchronization of threads or devices is never needed in
 *      an application using these functions. All functions returning a value
 *      block until the value is ready.
 *    @defgroup blas2 BLAS level 2 routines: Matrix-Vector
 *    @brief \f$ \alpha M \cdot x + \beta y\f$ and \f$ x^T M y \f$
 *
 *      @note Successive calls to blas routines are executed sequentially
 *      @note A manual synchronization of threads or devices is never needed in an application
 *      using these functions. All functions returning a value block until the value is ready.
 *    @defgroup mpi_structures MPI backend
 *    @{
 *        @defgroup mpi_matvec Distributed Matrix and Vector
 *        @defgroup mpi_comm MPI Distributed Gather and Scatter
 *    @}
 *    @defgroup utility Utility
 *    @{
 *        @defgroup dispatch The tag dispatch system
 *        @{
 *           @defgroup traits All TensorTraits specialisations
 *        @}
 *        @defgroup sparsematrix Sparse matrix formats
 *        @defgroup densematrix Dense matrix formats
 *        @defgroup view Vector view
 *        @defgroup typedefs Typedefs for Vectors and Matrices
 *    @}
 *    @brief \#include <mpi.h> before any other dg header file
 * @}
 * @defgroup lvevl2 Level 2: Basic numerical algorithms
 * @{
 *     @defgroup time ODE integration
 *     @brief \f$ \dot y = f(y,t) \f$
 *     @{
 *       @defgroup time_utils Utilities for ODE integrators
 *     @}
 *     @defgroup integration Quadrature
 *     @brief \f$ \int_{t_0}^T u(t) dt \f$
 *     @defgroup extrapolation Extrapolation
 *     @defgroup invert Linear and nonlinear solvers
 *     @brief Linear \f$ Ax = b\f$ and non-linear \f$ f(x) = b\f$
 *
 * @}
 * @defgroup level3 Level 3: Topology and Geometry
 * @{
 *     @defgroup grid Topological grids and operations
 *
 * Objects that store topological information (which point is neighbour of
 * which other point) about the grid.
 *     @{
 *         @defgroup basictopology Discontinuous Galerkin Nd grid
 *         @defgroup basicXtopology Discontinuous Galerkin grids with X-point
 *         @defgroup evaluation Evaluate, integrate, weights
 *         @brief \f$ f_i = f(\vec x_i) \f$
 *         @defgroup creation Create derivatives
 *         @brief  \f$ D_x\f$, \f$ D_y\f$ and \f$ D_z \f$
 *
 * High level matrix creation functions
 *         @defgroup stencil Create stencils
 *         @defgroup interpolation Interpolation, Projection, Transformation
 *         @{
 *              @defgroup fast_interpolation Fast interpolation
 *         @}
 *         @brief \f$ I \f$ and \f$ P = I^\dagger\f$
 *         @defgroup average Averaging and partial reductions
 *         @defgroup scatter Scatter and Gather
 *     @}
 *     @defgroup geometry Geometric grids and tensor operations
 *
 * These routines form the heart of our geometry free numerical algorithms.
 * They are called by our geometric operators like the Poisson bracket.
 *     @{
 *         @defgroup basicgeometry Geometry classes
 *         @defgroup basicXgeometry Geometry classes with X-point
 *         @defgroup generators Grid refinement
 *
 *         @defgroup tensor Tensor-Vector operations
 *         @brief \f$ v^i = T^{ij} w_j\f$
 *
 *         @defgroup pullback Pullback, pushforward, volume
 *         @brief \f$ f_i = f( x (\zeta_i,\eta_i), y(\zeta_i,\eta_i)) \f$, \f$
 *         \sqrt{g} \f$
 *     @}
 *     @defgroup fem Finite Element Methods
 *     @defgroup gridtypes Grid Typedefs
 * @}
 * @defgroup lvel4 Level 4: Advanced numerical schemes
 *
 * These routines make use of both the basic operations as well as the
 * interfaces defined in the Geometry section.
 * @{
 *     @defgroup arakawa Advection terms
 *     @brief \f$ \vec v \cdot \nabla u\f$ and \f$ \{ f,g\} \f$
 *     @defgroup matrixoperators Elliptic operators
 *     @brief Elliptic \f$ -\nabla\cdot (\chi \nabla f)\f$ and Helmholtz \f$ (\chi + \alpha \Delta) f\f$
 *     @defgroup multigrid Multigrid matrix inversion
 *     @brief \f$ A x = b\f$
 * @}
 * @defgroup misc Level 99: Miscellaneous additions
 * @{
 *     @defgroup timer Timer class
 *     @brief  t.tic(); t.toc(); t.diff();
 *     @defgroup blas1_helpers Functions and functors
 *     @{
 *          @defgroup basics Simple
 *          @brief   For the dg::evaluate and dg::blas1::evaluate functions
 *
 *          @defgroup functions A large collection
 *          @brief   For the dg::evaluate and dg::blas1::evaluate functions
 *
 *          @defgroup composition Composition of two or more functors
 *
 *          @defgroup binary_operators dg::blas1::evaluate binary operators
 *          @brief   Binary subroutines for the dg::blas1::evaluate function
 *
 *          @defgroup variadic_evaluates dg::blas1::evaluate variadic functors
 *          @brief   Functors to use in the dg::blas1::evaluate function
 *
 *          @defgroup variadic_subroutines dg::blas1::subroutine subroutines
 *          @brief   Functors to use in the dg::blas1::subroutine functions
 *
 *          @defgroup filters dg::blas2::stencil subroutines
 *          @brief Functors to use in dg::blas2::stencil function
 *     @}
 *     @defgroup mpi_utility MPI utility functions
 *     @{
 *          @defgroup mpi_legacy Legacy MPI functions
 *     @}
 *     @defgroup lowlevel Lowlevel helper functions and classes
 *
 * @}
 *
 */

/** @class hide_binary
  * @tparam BinaryOp A class or function type with a member/signature equivalent to
  *  - <tt>real_type operator()(real_type, real_type) const </tt>
  */
/** @class hide_ternary
  * @tparam TernaryOp A class or function type with a member/signature equivalent to
  *  - <tt>real_type operator()(real_type, real_type, real_type) const</tt>
  */

/** @class hide_construct
* @brief Perfect forward parameters to one of the constructors
*
* @tparam Params deduced by the compiler
* @param ps parameters forwarded to constructors
*/
/** @class hide_copyable
* @brief Return an object of same size as the object used for construction
* @return A copyable object; what it contains is undefined, its size is important
*/

 /**
  * @class hide_value_type>
  * @tparam value_type
  * Any type that can be used in an arithmetic operation with
  * <tt>dg::get_value_type<ContainerType></tt>
  */
 /** @class hide_ContainerType
  * @tparam ContainerType
  * Any class for which a specialization of \c TensorTraits exists and which
  * fulfills the requirements of the there defined data and execution policies derived from \c AnyVectorTag and \c AnyPolicyTag.
  * Among others
  *  - <tt> dg::HVec (serial), dg::DVec (cuda / omp), dg::MHVec (mpi + serial) or dg::MDVec (mpi + cuda / omp) </tt>
  *  - <tt> std::vector<dg::DVec> </tt> (vector of shared device vectors), <tt> std::array<double, 4> </tt> (array of 4 doubles) or <tt> std::map < std::string, dg::DVec> </tt>  ( a map of named vectors)
  *  - <tt> double (scalar)</tt> and other primitive types ...
  *  .
  * If there are several \c ContainerTypes in the argument list, then \c TensorTraits must exist for all of them
  * \see See \ref dispatch for a detailed explanation of our type %dispatch system
  */
 /** @class hide_matrix
  * @tparam MatrixType
  * Any class for which a specialization of \c TensorTraits exists and defines
  * a tensor_category derived from \c AnyMatrixTag.
  * Furthermore, any functor/lambda type with signature
  *  <tt> void operator()( const ContainerType0&, ContainerType1&) </tt>.
  * For example
  *  - \c scalar or \c Container: Scalars and containers act as diagonal matrices.
  *  - <tt> std::function<void( const ContainerType0&, ContainerType1&)> </tt>
  *  - \c dg::HMatrix and \c dg::IHMatrix with \c dg::HVec or \c std::vector<dg::HVec>
  *  - \c dg::DMatrix and \c dg::IDMatrix with \c dg::DVec or \c std::vector<dg::DVec>
  *  - \c dg::MHMatrix with \c dg::MHVec or \c std::vector<dg::MHVec>
  *  - \c dg::MDMatrix with \c dg::MDVec or \c std::vector<dg::MDVec>
  *  .
  * In case of \c SelfMadeMatrixTag only those \c blas2 functions that have a
  * corresponding member function in the Matrix class (e.g.
  * <tt> symv( const ContainerType0&, ContainerType1&); </tt> )
  * can be called.  If a \c Container has the \c RecursiveVectorTag, then the
  * matrix is applied to each of the elements unless the type has the
  * \c SelfMadeMatrixTag or is a Functor type.
  */
  /** @class hide_geometry
  * @tparam Geometry
  * A type that is or derives from one of the abstract geometry base classes ( \c aGeometry2d, \c aGeometry3d, \c aMPIGeometry2d, ...).
  */

  /** @class hide_container_geometry
  * @tparam Container
  * A data container class for which the \c blas1 functionality is overloaded and to which the return type of \c blas1::subroutine() can be converted using \c dg::construct.
  * We assume that \c Container is copyable/assignable and has a swap member function.
  * In connection with \c Geometry this is one of
  *  - \c dg::HVec, \c dg::DVec when \c Geometry is a shared memory geometry
  *  - \c dg::MHVec or \c dg::MDVec when \c Geometry is one of the MPI geometries
  * @tparam Geometry
  A type that is or derives from one of the abstract geometry base classes ( \c aGeometry2d, \c aGeometry3d, \c aMPIGeometry2d, ...). \c Geometry determines which \c Container type can be used.
  */

  /** @class hide_geometry_matrix_container
  * @tparam Geometry
  * A type that is or derives from one of the abstract geometry base classes ( \c aGeometry2d, \c aGeometry3d, \c aMPIGeometry2d, ...). \c Geometry determines which \c Matrix and \c Container types can be used:
  * @tparam Matrix
  * A class for which the \c dg::blas2::symv functions are callable in connection with the
  * \c Container class and to which the return type of \c dg::create::dx() can be
  * converted using \c dg::blas2::transfer.
  * The \c Matrix type can be one of:
  *  - \c dg::HMatrix with \c dg::HVec and one of the shared memory geometries
  *  - \c dg::DMatrix with \c dg::DVec and one of the shared memory geometries
  *  - \c dg::MHMatrix with \c dg::MHVec and one of the MPI geometries
  *  - \c dg::MDMatrix with \c dg::MDVec and one of the MPI geometries
  * @tparam Container
  * A data container class for which the \c blas1 functionality is overloaded and to which the return type of \c blas1::subroutine() can be converted using \c dg::assign.
  * We assume that \c Container is copyable/assignable and has a swap member function.
  * In connection with \c Geometry this is one of
  *  - \c dg::HVec, \c dg::DVec when \c Geometry is a shared memory geometry
  *  - \c dg::MHVec or \c dg::MDVec when \c Geometry is one of the MPI geometries
  */

/*! @mainpage
 * Welcome to the dg library
 */

/*!
 * @addtogroup dispatch
 * @section dispatch The dg dispatch system
 *
 * At the heart of the dispatch system lies the <tt>dg::TensorTraits</tt> class that specifies how
 * a type \c T behaves in the respective functions though a tag system defined by the \c tensor_category tag and the \c execution_policy tag. Per default it looks like
@code{.cpp}
template< class T, class Enable=void>
struct TensorTraits
{
    using value_type = void;
    using tensor_category = NotATensorTag;
    using execution_policy = NoPolicyTag;
};
@endcode
 * The values for \c tensor_category are either \c dg::NotATensorTag or (a class derived from) \c dg::AnyMatrixTag,
 * while the values for \c execution_policy are either \c dg::NoPolicyTag or (a class derived from) \c dg::AnyPolicyTag.
 * Any type \c T assumes the above default values for \c value_type, \c tensor_category and \c execution_policy unless a specialisation exists for that type (see \ref traits for a full list).
 * We define the following nomenclature
 *    - \e Scalar: A template parameter T is a Scalar if <tt> typename dg::TensorTraits<T>::tensor_category </tt>  derives from \c dg::AnyScalarTag
 *    - \e Vector: A template parameter T is a Vector if it is not a Scalar and if <tt> typename dg::TensorTraits<T>::tensor_category </tt> derives from \c dg::AnyVectorTag
 *    - \e Matrix: A template parameter T is a Matrix if it is not a Scalar or Vector and if <tt> typename  dg::TensorTraits<T>::tensor_category </tt> derives from \c dg::AnyMatrixTag
 *    - \e execution \e policy: A template parameter T has an execution policy if
 *      <tt> typename dg::TensorTraits<T>::execution_policy </tt> derives from \c dg::AnyPolicyTag, The execution policy is \e trivial if it is \c dg::AnyPolicyTag
 *    - \e value \e type : A template parameter T has a value type if
 *      <tt> typename dg::TensorTraits<T>::value_type </tt> is non-void
 *    - \e compatible: Two vectors are compatible if their tensor_categories both derive from the same base class that itself derives from but is not equal to \c dg::AnyVectorTag, Two execution policies are compatible if they are equal or if at least one of them is trivial.
 *    - \e promote: A Scalar can be promoted to a Vector with all elements equal to the value of the Scalar. A Vector can be promoted to a  Matrix with the Vector being the diagonal and all other elements zero.
 *
 * When dispatching level 1 functions we distinguish between three classes of
 * functions: trivially parallel (\c dg::blas1::subroutine and related \c dg::blas1 functions), global communication
 * (\c dg::blas1::dot and \c dg::blas2::dot) and local communication (\c dg::blas2::symv).
 *
 * @subsection dispatch_evaluate The subroutine function
 *
 * The execution of \c dg::blas1::subroutine with a Functor called \c routine
 * (and all related \c dg::blas1 functions) is equivalent to the following:
 *  -# Assert the following prerequisites:
 *
 *      -# All template parameter types must be either Scalars or Vectors and have an execution policy and a value type
 *      -# All Vectors and all execution policies must be mutually compatible
 *      -# All Vectors must contain the same number of elements (equal sizes)
 *      -# The number of template parameters must be equal to the number of parameters in the \c routine
 *      -# The value type of every template parameter must be convertible to the respective parameter type in the \c routine
 *  -# If all types are Scalars, apply the \c routine and return
 *  -# If at least one type is a Vector, then all Scalars
 *     are promoted to this type with the same size, communicator and execution policy
 *  -# Check the base class of the tensor_category:
 *      -# If \c dg::SharedVectorTag, check execution policy and dispatch to respective implementation (The implementation just loops over all elements in the vectors and applies the \c routine)
 *      -# If \c dg::MPIVectorTag, access the underlying data and recursively call \c dg::blas1::subroutine (and start again at 1)
 *      -# If \c dg::RecursiveVectorTag, loop over all elements and recursively call \c dg::blas1::subroutine for all elements (and start again at 1)
 *
 * @subsection dispatch_dot The dot function
 * The execution of \c dg::blas1::dot and \c dg::blas2::dot is equivalent to the following:
 *  -# Assert the following prerequisites:
 *
 *      -# All template parameter types must be either Scalars or Vectors and have an execution policy and a value type
 *      -# All Vectors and all execution policies must be mutually compatible
 *      -# All Vectors must contain the same number of elements (equal sizes)
 *  -# If all types are Scalars, multiply and return
 *  -# If at least one type is a Vector, then all Scalars
 *     are promoted to this type with the same size, communicator and execution policy
 *  -# Check the base class of the tensor_category:
 *      -# If \c dg::SharedVectorTag, check execution policy and dispatch to respective implementation (The implementation multiplies and accumulates all elements in the vectors). Return the result.
 *      -# If \c dg::MPIVectorTag, assert that the vector MPI-communicators are
 *      \c congruent (same process group, different context) or \c ident (same
 *      process group, same context). Then, access the underlying data and
 *      recursively call \c dot (and start again at 1). Accumulate the result
 *      among participating processes and return.
 *      -# If \c dg::RecursiveVectorTag, loop over all elements and recursively call \c dot for all elements (and start again at 1). Accumulate the results
 *      and return.
 *
 * @subsection dispatch_symv The symv function
 * In the execution of the \c dg::blas2::symv (and its aliases \c dg::blas2::gemv and \c dg::apply)
 * each matrix type has individual prerequisites and execution paths depending on its \c tensor_category.
 * We can identify some general rules:
 *   -# If the tensor category is \c dg::NotATensorTag (any type has this tag
 *   by default unless it is overwritten by a specialization of
 *   \c dg::TensorTraits ) then the compiler assumes that the type is a functor
 *   type and immediately forwards all parameters to the \c operator() member
 *   and none of the following applies
 *   -# If the Matrix type has the \c dg::SelfMadeMatrixTag tensor category,
 *   then all parameters are immediately forwarded to the \c symv member
 *   function.  No asserts are performed and none of the following applies.
 *   -# If the Matrix type is either a Scalar or a Vector and the remaining types do
 *   not have the \c dg::RecursiveVectorTag tensor category, then
 *   \c dg::blas2::symv is equivalent to \c dg::blas1::pointwiseDot
 *   -# The container template parameters must be Vectors or Scalars and must
 *   have compatible execution policies. Vectors must be compatible.
 *   -# If the tensor category of the Vectors is \c dg::RecursiveVectorTag and
 *   the tensor category of the Matrix is not, then the \c dg::blas2::symv is
 *   recursively called with the Matrix on all elements of the Vectors.
 */

namespace dg{
/*! @brief Simple tool for performance measuring
 *
 * The implementation of this class is chosen with compile-time MACROS \c THRUST_DEVICE_SYSTEM and \c MPI_VERSION.
 * @code
   dg::Timer t;
   t.tic();
   some_function_to_benchmark();
   t.toc();
   std::cout << "Function took "<<t.diff()<<"s\n";
 * @endcode
 * @ingroup timer
 */
struct Timer
{
    /*! @brief Start timer
    *
    * uses \c omp_get_wtime() if available, else \c gettimeofday.
    * If compiled with nvcc we place \c cudaEvent_t in the gpu stream.
    * The mpi version places an \c MPI_Barrier(MPI_COMM_WORLD)
    * and then uses \c MPI_Wtime.
    * MPI + Cuda adds an additional \c cudaDeviceSynchronize
    */
    void tic();
    /*!
    * @brief Stop timer
    * @copydetails tic
    */
    void toc();
    /*!
     * \brief Return time in seconds elapsed between tic and toc
     * \return Time in seconds between calls of tic and toc*/
    double diff()const;
};
}

/**
 * @brief Expands to \__host__ \__device__ if compiled with nvcc else is empty
 * @ingroup typedefs
 */
#define DG_DEVICE
