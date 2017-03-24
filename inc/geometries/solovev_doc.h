#error Documentation only
/*! 
 * 
 * @defgroup grids 1. Grids 
 * @defgroup generators 2. Grid generators
 *
      All the grids introduced by this extension can be constructed with 
      generator classes. A generator class is a class that models aGenerator. 
 * @defgroup fluxfunctions 3. New functors based on the magnetic field geometry

 All functors in this section model two or three-dimensional functions, i.e. they all overload the operator() like aBinaryOperator
 * @{
      @defgroup geom 3.1 new flux functions and derivatives
      @defgroup magnetic 3.2 magnetic field and associated functors
      @defgroup profiles 3.3 miscellaneous functors based on flux functions
 * @}
 * @defgroup misc 4. Miscellaneous additions
 *
 * Objects that are used to define and integrate the magnetic field lines. 
 * All objects can be used in the evaluation() function.
   @defgroup temp 99. Template models
   Documentation for template models
 * 
 */
/*! @mainpage
 * This extension adds new features to the FELTOR core dg library. 
 *
 * - the conformal, orthogonal and curvilinear grid classes are added to the
 * dg namespace. 
 * - several grid generator classes are added, among them our new Hector class
 *   are added to the dg::geo namespace
 * - a bunch of new functors implementing various magnetic field geometries
 *   and profiles are added to the dg::geo namespace
 * - there are some miscellaneous additions like a flux surface average class
 * and one used to integrate the field lines for parallel derivatives all in the dg::geo namespace.
 */


/**
* @brief The generator template model

A generator is there to construct coordinates from some coordinates
\f$ x,y\f$ to the computational domain \f$\zeta, \eta\f$, which
is a product space. 
 @attention this is not a real class it's there for documentation only
 @attention parameter names can be different
 @ingroup temp
*/
struct aGenerator
{
    double width() const; //!<length in \f$ \zeta\f$ 
    double height() const; //!<length in \f$ \eta\f$
    bool isOrthogonal() const; //!< true if coordinate system is orthogonal
    bool isConformal() const; //!< true if coordinate system is conformal
    /**
    * @brief Generate grid points and elements of the Jacobian 
    *
    * @param zeta1d (input) a list of \f$ N_\zeta\f$ points \f$ 0<\zeta_i<\f$width() 
    * @param eta1d (input) a list of \f$ N_\eta\f$ points \f$ 0<\eta_j<\f$height() 
    * @param x (output) the list of \f$ N_\eta N_\zeta\f$ coordinates \f$ x(\zeta_i, \eta_j)\f$ 
    * @param y (output) the list of \f$ N_\eta N_\zeta\f$ coordinates \f$ y(\zeta_i, \eta_j)\f$ 
    * @param zetaX (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\zeta/\partial x (\zeta_i, \eta_j)\f$ 
    * @param zetaY (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\zeta/\partial y (\zeta_i, \eta_j)\f$ 
    * @param etaX (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\eta/\partial x (\zeta_i, \eta_j)\f$ 
    * @param etaY (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\eta/\partial y (\zeta_i, \eta_j)\f$ 
    @note the \f$ \zeta\f$ coordinate is contiguous in memory
    */
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

/**
* @brief The generatorX  template model

A generator is there to construct coordinates from some coordinates
\f$ x,y\f$ to the computational domain \f$\zeta, \eta\f$, which
is a product space and has an X-point topology. 
 @attention this is not a real class it's there for documentation only
 @attention parameter names can be different
 @ingroup temp
*/
struct aGeneratorX
{
    bool isOrthogonal() const; //!< true if coordinate system is orthogonal
    bool isConformal() const; //!< true if coordinate system is conformal
    double f0() const; //!< the normalization constant  of the \f$\zeta\f$ coordinate i.e. \f$ \zeta_0 = f_0\psi_0 \f$ and \f$ \zeta_1 = -f_\zeta\zeta_0/(1-f_\zeta) \f$ 
    /**
    * @brief Generate grid points and elements of the Jacobian 
    *
    * @param zeta1d (input) a list of \f$ N_\zeta\f$ points \f$ f_0\psi_0<\zeta_i< -f_\zeta\zeta_0/(1-f_\zeta)\f$
    * @param eta1d (input) a list of \f$ N_\eta\f$ points \f$ 0<\eta_j<\f$height() 
    * @param x (output) the list of \f$ N_\eta N_\zeta\f$ coordinates \f$ x(\zeta_i, \eta_j)\f$ 
    * @param y (output) the list of \f$ N_\eta N_\zeta\f$ coordinates \f$ y(\zeta_i, \eta_j)\f$ 
    * @param nodeX0 is the index of the first point in eta1d  after the first jump in topology in \f$ \eta\f$
    * @param nodeX1 is the index of the first point in eta1d  after the second jump in topology in \f$ \eta\f$
    * @param zetaX (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\zeta/\partial x (\zeta_i, \eta_j)\f$ 
    * @param zetaY (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\zeta/\partial y (\zeta_i, \eta_j)\f$ 
    * @param etaX (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\eta/\partial x (\zeta_i, \eta_j)\f$ 
    * @param etaY (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\eta/\partial y (\zeta_i, \eta_j)\f$ 
    @note the \f$ \zeta\f$ coordinate is contiguous in memory
    */
    void operator()( 
         const thrust::host_vector<double>& zeta1d, 
         const thrust::host_vector<double>& eta1d, 
         const unsigned nodeX0, const unsigned nodeX1, 
         thrust::host_vector<double>& x, 
         thrust::host_vector<double>& y, 
         thrust::host_vector<double>& zetaX, 
         thrust::host_vector<double>& zetaY, 
         thrust::host_vector<double>& etaX, 
         thrust::host_vector<double>& etaY) ;
};


/**
* @brief The functor template model

A functor in the dg framework can be used in the evaluate or pullback
functions. The only function to overload is the operator() member.
 @attention this is not a real class it's there for documentation only
 @attention parameter names can be different
 @ingroup temp
*/
struct aBinaryOperator
{
    double operator()(double x, double y); //!< 2d version
};

/**
* @brief The container template model

A container in the dg framework must be usable in the blas functions
of the main dg library. 
 Also it must be compatible to a thrust::host_vector<double> in the blas1::transfer function.
 Can be one of
 - thrust::host_vector<double>
 - thrust::device_vector<double> 
 - cusp::array1d<double, cusp::device_memory>
 - MPI_Vector<thrust::host_vector<double> >
 - MPI_Vector<thrust::device_vector<double> >
 - MPI_Vector<cusp::array1d<double, cusp::device_memory> >


 @attention this is not a real class it's there for documentation only
 @attention parameter names can be different
 @ingroup temp
*/
struct aContainer
{
};

/**
* @brief The tokamak magnetic geometry template model

A tokamak geometry must contain the functors named as follows, all of
which must model aBinaryOperator.
 @attention this is not a real class it's there for documentation only
 @attention parameter names cannot be different
 @ingroup temp
*/
struct aTokamakMagneticField
{
    Psip psip; //!< \f$ \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    PsipR psipR;//!< \f$ \partial_R \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    PsipZ psipZ;//!< \f$ \partial_Z \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    PsipRR psipRR;//!< \f$ \partial_R\partial_R \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    PsipRZ psipRZ;//!< \f$ \partial_R\partial_Z \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    PsipZZ psipZZ;//!< \f$ \partial_Z\partial_Z \psi_p(R,Z)\f$, where R, Z are cylindrical coordinates
    LaplacePsip laplacePsip; //!< 2d Laplacian of \f$ \psi_p\f$ 
    Ipol ipol; //!< \f$ I(\psi_p) \f$ the current
    IpolR ipolR; //!< \f$ \partial_R I(\psi_p) \f$ 
    IpolZ ipolZ; //!< \f$ \partial_Z I(\psi_p) \f$ 
};
