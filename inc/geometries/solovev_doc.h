#error Documentation only
/*! 
 * 
 * @defgroup grids 0. New Geometry classes
 * @defgroup generators 1. Grid generators
 *
      All the grids introduced by this extension can be constructed with 
      generator classes. 
 * @defgroup fluxfunctions 2. New functors based on the magnetic field geometry

 All functors in this section model two or three-dimensional functions, i.e. they all overload the operator() like aBinaryOperator
 * @{
      @defgroup geom 2.1 new flux functions and derivatives
      @defgroup magnetic 2.2 magnetic field and associated functors
      @defgroup profiles 2.3 miscellaneous functors based on flux functions
 * @}
 * @defgroup misc 3. Miscellaneous additions
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
  * @tparam container A data container class for which the blas1 functionality is overloaded. Also we assume that the type is copyable/assignable. Currently this is one of 
  *   dg::HVec, dg::DVec, dg::MHVec or dg::MDVec
  */
