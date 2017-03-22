#error Documentation only
/*! @namespace solovev
 * @brief This is the namespace for a solvev magnetic flux
 */
/*! 
 * 
 * @defgroup grids Grids 
 * @defgroup generators Grid generators
 *
      All the grids introduced by this extension can be constructed with 
      generator classes. A generator class must provide the following public member functions.
  @code
struct SomeGenerator
{
    double width() const; //length in zeta
    double height() const; //length in eta
    bool isOrthogonal() const; 
    bool isConformal()  const;
    void operator()( 
         const thrust::host_vector<double>& zeta1d, 
         const thrust::host_vector<double>& eta1d, 
         thrust::host_vector<double>& x, 
         thrust::host_vector<double>& y, 
         thrust::host_vector<double>& zetaX, 
         thrust::host_vector<double>& zetaY, 
         thrust::host_vector<double>& etaX, 
         thrust::host_vector<double>& etaY) ;
};
@endcode
    A generator class for an X-point type topology must have the following
 * @defgroup fluxfunctions New functors based on the magnetic field geometry
 * @{
      @defgroup geom new flux functions and derivatives
      @defgroup magnetic magnetic field and associated functors
      @defgroup profiles miscellaneous functors based on flux functions
 * @}
 * @defgroup misc Miscellaneous additions
 *
 * Objects that are used to define and integrate the magnetic field lines. 
 * All objects can be used in the evaluation() function.
 * 
 */
/*! @mainpage
 * Welcome to the Geometry extension of FELTOR.
 *
 * This extension adds new features to the FELTOR core dg library. Firstly, 
 * the conformal, orthogonal and curvilinear grid classes are added to the
 * dg namespace. 
 * Secondly, several grid generator classes are added, among them our new Hector class. 
 * Thirdly, a bunch of new functors implementing various magnetic field geometries
 * and profiles are added to the dg::geo namespace
 * Lastly, there are some miscellaneous additions like a flux surface average class
 * and one used to integrate the field lines for parallel derivatives.
 */
