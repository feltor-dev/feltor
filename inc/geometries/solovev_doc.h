#error Documentation only
/*! 
 * 
 * @defgroup grids 1. New geometric grids
 * @defgroup generators_geo 2. Grid generators
 *
      All the grids introduced by this extension can be constructed with 
      generator classes. 
 * @defgroup fluxfunctions 3. New functors based on the magnetic field geometry

 All functors in this section model two or three-dimensional functions, i.e. they all overload the operator() like aBinaryOperator
 * @{
      @defgroup geom 3.1 new flux functions and derivatives
      @defgroup magnetic 3.2 magnetic field and associated functors
      @defgroup profiles 3.3 miscellaneous functors based on flux functions
 * @}
 * @defgroup fieldaligned 4. Fieldaligned derivatives
 * @defgroup misc_geo 5. Miscellaneous additions
 *
 * Objects that are used to define and integrate the magnetic field lines. 
 * All objects can be used in the evaluation() function.
 * 
 */
/*! @mainpage
 * This extension adds new features to the FELTOR core dg library. 
 *
 * - several grid generator classes are added, among them our new Hector class
 *   are added to the dg::geo namespace
 * - a bunch of new functors implementing various magnetic field geometries
 *   and profiles are added to the dg::geo namespace
 * - there are some miscellaneous additions like a flux surface average class
 * and one used to integrate the field lines for parallel derivatives all in the dg::geo namespace.
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
