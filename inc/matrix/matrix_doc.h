#error Documentation only
/*!
 *
 *
 * @defgroup matrixnumerical0 Level 2: Basic numerical algorithms
 * @{
 *     @defgroup matrixinvert Inversion of tridiagonal matrices
 *      @brief \f$T^{-1}\f$
 *     @defgroup tridiagfunction Tridiagonal Matrix-functions
 *       approximation
 *      @brief \f$ x \approx f(T)b \f$
 *     @defgroup matrixfunctionapproximation Matrix-functions
 *       approximation
 *      @brief \f$ x \approx f(A)b \f$
 *     @defgroup exp_int Exponential integrators
 *      @brief \f$ \dot y = Ay + g(t,y)\f$
 * @}
 *
 * @defgroup matrixnumerical1 Level 4: Advanced numerical schemes
 * @{
 *     @defgroup matrixmatrixoperators Elliptic operators
 *     @brief Advanced elliptic operators
 * @}
 */
/*! @mainpage
 * This extension adds new features to the FELTOR core dg library.
 *
 */

/** @class hide_construct
* @brief Perfect forward parameters to one of the constructors
*
* @tparam Params deduced by the compiler
* @param ps parameters forwarded to constructors
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
