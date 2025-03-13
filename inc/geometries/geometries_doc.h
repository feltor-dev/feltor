#error Documentation only
/*!
 *
 * @defgroup grids 1. General Curvilinear grids
 * @defgroup generators_geo 2. Curvilinear grid generators
 * @brief \f$( x,\ y,\ \zeta_x,\ \zeta_y,\ \eta_x,\ \eta_y)\f$ dg::geo::Hector
 *
 *      All the grids introduced by this extension can be constructed with
 *      generator classes.
 * @defgroup geom_functors 3. New functors surrounding the magnetic field geometry
 * @brief \f$ f(R,Z,\varphi)\f$

 * All functors in this section model two or three-dimensional functions, i.e.
 * they all overload the operator() like \c aCylindricalFunctor
 * @{
      @defgroup geom 3.1 Creating a flux function
      @brief \f$\psi_p(R,Z)\f$ dg::geo::createMagneticField
      @{
        @defgroup solovev The solovev expansion
        @defgroup polynomial The polynomial expansion
        @defgroup taylor The Taylor state expansion
        @defgroup guenter The Guenter expansion
        @defgroup toroidal The Purely Toroidal expansion
        @defgroup circular The Circular expansion
        @defgroup mod  Modification of any expansion
        @brief \f$ H(\psi_p(R,Z))\f$, dg::geo::createModifiedField
        @defgroup wall Wall and Sheath
      @}
      @defgroup magnetic 3.2 Magnetic field, curvatures and associated functors
      @defgroup profiles 3.3 Penalization, weight and monitor metric functors
      @defgroup fluxfunctions 3.4. Utility functor functionality
      @brief \f$ h^{ij}\f$
 * @}
 * @defgroup fieldaligned 4. Fieldaligned derivatives
 * @brief \f$ \nabla_\parallel f\f$
 * @defgroup misc_geo 5. Miscellaneous additions
 * @brief \f$ q(\psi_p)\f$ and \f$ \nabla \psi_p ( R_X, Z_X) = 0\f$
 */
/*! @mainpage
 * This extension adds new features to the FELTOR core dg library.
 *
 * - This header incurs a dependency on \c jsoncpp (or \c nlohmann::json if \c DG_USE_JSONHPP is defined)
 * - several grid generator classes are added, among them our new Hector class
 *   are added to the \c dg::geo namespace
 * - a bunch of new functors implementing various magnetic field geometries
 *   and profiles are added to the \c dg::geo namespace
 * - there are some miscellaneous additions like a flux surface average class
 * and one used to integrate the field lines for parallel derivatives all in the dg::geo namespace.
 *
 * We have a collection of writeups:
 *  - The <a href="https://www.overleaf.com/read/jjvstccqzcjv" target="_blank">parallel derivative</a>
 *  - The <a href="https://www.overleaf.com/read/vvqwffbjvrxm" target="_blank">HECTOR algorithm</a>
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
  *  TensorTraits class (e.g. Elliptic). In this case only those blas2 functions
  *  that have a corresponding member function in the Matrix class (e.g. symv( const container&, container&); ) can be called.
  *  If the container is a std::vector, then the Matrix is applied to each of the elements.
  */
