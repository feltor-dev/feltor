#pragma once

#include "blas.h"
#include "enums.h"
#include "backend/memory.h"
#include "topology/evaluation.h"
#include "topology/derivatives.h"
#ifdef MPI_VERSION
#include "topology/mpi_derivatives.h"
#include "topology/mpi_evaluation.h"
#endif
#include "topology/geometry.h"

/*! @file

  @brief General negative elliptic operators
  */
namespace dg
{
// Note that there are many tests for this file : elliptic2d_b,
// elliptic2d_mpib, elliptic_b, elliptic_mpib, ellipticX2d_b
// And don't forget inc/geometries/elliptic3d_t (testing alignment and
// projection tensors as Chi) geometry_elliptic_b, geometry_elliptic_mpib,
// and geometryX_elliptic_b and geometryX_refined_elliptic_b

/**
 * @brief A 2d negative elliptic differential operator \f$ -\nabla \cdot ( \mathbf{\chi}\cdot \nabla ) \f$
 *
 * @ingroup matrixoperators
 *
 * The term discretized is \f[ -\nabla \cdot ( \mathbf{\chi} \cdot \nabla ) \f] where \f$
 * \nabla \f$ is the two-dimensional nabla and \f$\chi = \sigma
 * \mathbf{\tau}\f$ is a (possibly spatially dependent) tensor with scalar part
 * \f$ \sigma\f$ (usually the volume form) and tensor part \f$ \tau\f$ (usually
 * the inverse metric). In general coordinates that means
 * \f[ -\frac{1}{\sqrt{g}}\left(
 * \partial_x\left(\sqrt{g}\left(\chi^{xx}\partial_x + \chi^{xy}\partial_y \right)\right)
 + \partial_y\left(\sqrt{g} \left(\chi^{yx}\partial_x + \chi^{yy}\partial_y \right)\right) \right)\f]
 is discretized.
 Per default, \f$ \chi = \sqrt{g} g^{-1}\f$ but you can set it to any tensor
 you like (in order for the operator to be invertible \f$\chi\f$ should be
 symmetric and positive definite though).

 * @sa Our theory guide <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a> on overleaf holds a detailed derivation

 Note that the local discontinuous Galerkin discretization adds so-called jump terms
 \f[ D^\dagger \chi D + \alpha \chi_{on/off} J \f]
 where \f$\alpha\f$  is a scale factor ( = jfactor), \f$ D \f$ contains the discretizations of the above derivatives, and \f$ J\f$ is a self-adjoint matrix.
 (The symmetric part of \f$J\f$ is added @b before the volume element is divided). The adjoint of a matrix is defined with respect to the volume element including dG weights.
 Usually, the default \f$ \alpha=1 \f$ is a good choice.
 However, in some cases, e.g. when \f$ \chi \f$ exhibits very large variations
 \f$ \alpha=0.1\f$ or \f$ \alpha=0.01\f$ might be better values.
 In a time dependent problem the value of \f$\alpha\f$ determines the
 numerical diffusion, i.e. for too low values numerical oscillations may appear.
 The \f$ \chi_{on/off} \f$ in the jump term serves to weight the jump term with \f$ \chi \f$. This can be switched either on or off with off being the default.
 Also note that a forward discretization has more diffusion than a centered discretization.

 The following code snippet demonstrates the use of \c Elliptic in an inversion problem
 * @snippet elliptic2d_b.cu pcg
 * @copydoc hide_geometry_matrix_container
 * This class has the \c SelfMadeMatrixTag so it can be used in \c blas2::symv functions
 * and thus in a conjugate gradient solver.
 * @note The constructors initialize \f$ \chi=\sqrt{g}g^{-1}\f$ so that a
 * negative laplacian operator results
 * @note The inverse of \f$ \sigma\f$ makes a good general purpose preconditioner
 * @note the jump term \f$ \alpha J\f$  adds artificial numerical diffusion as discussed above
 * @note Since the pattern arises quite often (because of the ExB velocity \f$ u_E^2\f$ in the ion gyro-centre potential)
 * this class also can compute the variation integrand \f$ \lambda^2\nabla \phi\cdot \tau\cdot\nabla\phi\f$
 * @attention Pay attention to the negative sign which is necessary to make the matrix @b positive @b definite
 */
template <class Geometry, class Matrix, class Container>
class Elliptic
{
    public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    Elliptic(){}
    /**
     * @brief Construct from Grid
     *
     * @param g The Grid, boundary conditions are taken from here
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @param chi_weight_jump If true, the Jump terms are multiplied with the Chi matrix, else it is ignored
     * @note chi is assumed the metric per default
     */
    Elliptic( const Geometry& g,
        direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false):
        Elliptic( g, g.bcx(), g.bcy(), dir, jfactor, chi_weight_jump)
    {
    }

    /**
     * @brief Construct from grid and boundary conditions
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @param chi_weight_jump If true, the Jump terms are multiplied with the Chi matrix, else it is ignored
     * @note chi is assumed the metric per default
     */
    Elliptic( const Geometry& g, bc bcx, bc bcy,
        direction dir = forward,
        value_type jfactor=1., bool chi_weight_jump = false)
    {
        m_jfactor=jfactor;
        m_chi_weight_jump = chi_weight_jump;
        dg::blas2::transfer( dg::create::dx( g, inverse( bcx), inverse(dir)), m_leftx);
        dg::blas2::transfer( dg::create::dy( g, inverse( bcy), inverse(dir)), m_lefty);
        dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
        dg::blas2::transfer( dg::create::dy( g, bcy, dir), m_righty);
        dg::blas2::transfer( dg::create::jumpX( g, bcx),   m_jumpX);
        dg::blas2::transfer( dg::create::jumpY( g, bcy),   m_jumpY);

        dg::assign( dg::create::volume(g),        m_weights);
        dg::assign( dg::evaluate( dg::one, g),    m_precond);
        m_temp = m_tempx = m_tempy = m_weights;
        m_chi=g.metric();
        m_sigma = m_vol = dg::tensor::volume(m_chi);
    }

    /**
    * @brief Perfect forward parameters to one of the constructors
    *
    * @tparam Params deduced by the compiler
    * @param ps parameters forwarded to constructors
    */
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = Elliptic( std::forward<Params>( ps)...);
    }

    /**
     * @brief Change scalar part in Chi tensor
     *
     * Internally, we split the tensor \f$\chi = \sigma\mathbf{\tau}\f$ into
     * a scalar part \f$ \sigma\f$ and a tensor part \f$ \tau\f$ and you can
     * set each part seperately. This functions sets the scalar part.
     *
     * @param sigma The new scalar part in \f$\chi\f$
     * @note The class will take care of the volume element in the divergence so do not multiply it to \c sigma yourself
     * @attention If some or all elements of sigma are zero the preconditioner
     * is invalidated and the operator can no longer be inverted. The symv
     * function can still be called however.
     * @tparam ContainerType0 must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0>
    void set_chi( const ContainerType0& sigma)
    {
        dg::blas1::pointwiseDot( sigma, m_vol, m_sigma);
        //update preconditioner
        dg::blas1::pointwiseDivide( 1., sigma, m_precond);
        // sigma is possibly zero, which will invalidate the preconditioner
        // it is important to call this blas1 function because it can
        // overwrite NaN in m_precond in the next update
    }
    /**
     * @brief Change tensor part in Chi tensor
     *
     * We split the tensor \f$\chi = \sigma\mathbf{\tau}\f$ into
     * a scalar part \f$ \sigma\f$ and a tensor part \f$ \tau\f$ and you can
     * set each part seperately. This functions sets the tensor part.
     *
     * @note The class will take care of the volume element in the divergence so do not multiply it to \c tau yourself
     *
     * @param tau The new tensor part in \f$\chi\f$ (must be positive definite)
     * @note the 3d parts in \c tau will be ignored for 2d computations
     * @tparam ContainerType0 must be usable in \c dg::assign to \c Container
     */
    template<class ContainerType0>
    void set_chi( const SparseTensor<ContainerType0>& tau)
    {
        m_chi = SparseTensor<Container>(tau);
    }

    /**
     * @brief Return the weights making the operator self-adjoint
     *
     * i.e. the volume form
     * @return volume form including weights
     */
    const Container& weights()const {
        return m_weights;
    }
    /**
     * @brief Return the default preconditioner to use in conjugate gradient
     *
     * Currently returns 1 divided by the scalar part of \f$ \sigma\f$.
     * This is especially good when \f$ \sigma\f$ exhibits large amplitudes or
     * variations
     * @return the inverse of \f$\sigma\f$.
     */
    const Container& precond()const {
        return m_precond;
    }
    /**
     * @brief Set the currently used jfactor (\f$ \alpha \f$)
     * @param new_jfactor The new scale factor for jump terms
     */
    void set_jfactor( value_type new_jfactor) {m_jfactor = new_jfactor;}
    /**
     * @brief Get the currently used jfactor (\f$ \alpha \f$)
     * @return  The current scale factor for jump terms
     */
    value_type get_jfactor() const {return m_jfactor;}
    /**
     * @brief Set the chi weighting of jump terms
     * @param jump_weighting Switch for weighting the jump factor with chi. Either true or false.
     */
    void set_jump_weighting( bool jump_weighting) {m_chi_weight_jump = jump_weighting;}
    /**
     * @brief Get the current state of chi weighted jump terms.
     * @return Whether the weighting of jump terms with chi is enabled. Either true or false.
     */
    bool get_jump_weighting() const {return m_chi_weight_jump;}
    /**
     * @brief Compute elliptic term and store in output
     *
     * i.e. \c y=M*x
     * @param x left-hand-side
     * @param y result
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void operator()( const ContainerType0& x, ContainerType1& y){
        symv( 1, x, 0, y);
    }

    /**
     * @brief Compute elliptic term and store in output
     *
     * i.e. \c y=M*x
     * @param x left-hand-side
     * @param y result
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y){
        symv( 1, x, 0, y);
    }
    /**
     * @brief Compute elliptic term and add to output
     *
     * i.e. \c y=alpha*M*x+beta*y
     * @param alpha a scalar
     * @param x left-hand-side
     * @param beta a scalar
     * @param y result
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void symv( value_type alpha, const ContainerType0& x, value_type beta, ContainerType1& y)
    {
        //compute gradient
        dg::blas2::gemv( m_rightx, x, m_tempx); //R_x*f
        dg::blas2::gemv( m_righty, x, m_tempy); //R_y*f

        //multiply with tensor (note the alias)
        dg::tensor::multiply2d(m_sigma, m_chi, m_tempx, m_tempy, 0., m_tempx, m_tempy);

        //now take divergence
        dg::blas2::symv( m_lefty, m_tempy, m_temp);
        dg::blas2::symv( -1., m_leftx, m_tempx, -1., m_temp);

        //add jump terms
        if( 0.0 != m_jfactor )
        {
            if(m_chi_weight_jump)
            {
                dg::blas2::symv( m_jfactor, m_jumpX, x, 0., m_tempx);
                dg::blas2::symv( m_jfactor, m_jumpY, x, 0., m_tempy);
                dg::tensor::multiply2d(m_sigma, m_chi, m_tempx, m_tempy, 0., m_tempx, m_tempy);
                dg::blas1::axpbypgz(1.0,m_tempx,1.0,m_tempy,1.0,m_temp);
            }
            else
            {
                dg::blas2::symv( m_jfactor, m_jumpX, x, 1., m_temp);
                dg::blas2::symv( m_jfactor, m_jumpY, x, 1., m_temp);
            }
        }
        dg::blas1::pointwiseDivide( alpha, m_temp, m_vol, beta, y);
    }

    /**
     * @brief \f$ \sigma = (\nabla\phi\cdot\tau\cdot\nabla \phi) \f$
     *
     * where \f$ \tau \f$ is the tensor part of \f$\chi\f$ that is the (inverse) metric by default
     * @param phi the vector to take the variation of
     * @param sigma (inout) the variation
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void variation(const ContainerType0& phi, ContainerType1& sigma){
        variation(1., 1., phi, 0., sigma);
    }
    /**
     * @brief \f$ \sigma = \lambda^2(\nabla\phi\cdot\tau\cdot\nabla \phi) \f$
     *
     * where \f$ \tau \f$ is the tensor part of \f$\chi\f$ that is the (inverse) metric by default
     * @param lambda input prefactor
     * @param phi the vector to take the variation of
     * @param sigma (out) the variation
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerTypeL, class ContainerType0, class ContainerType1>
    void variation(const ContainerTypeL& lambda, const ContainerType0& phi, ContainerType1& sigma){
        variation(1.,lambda, phi, 0., sigma);
    }
    /**
     * @brief \f$ \sigma = \alpha \lambda^2 (\nabla\phi\cdot\tau\cdot\nabla \phi) + \beta \sigma\f$
     *
     * where \f$ \tau \f$ is the tensor part of \f$\chi\f$ that is the (inverse) metric by default
     * @param alpha scalar input prefactor
     * @param lambda input prefactor
     * @param phi the vector to take the variation of
     * @param beta the output prefactor
     * @param sigma (inout) the variation
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerTypeL, class ContainerType0, class ContainerType1>
    void variation(value_type alpha, const ContainerTypeL& lambda, const ContainerType0& phi, value_type beta, ContainerType1& sigma)
    {
        dg::blas2::gemv( m_rightx, phi, m_tempx); //R_x*f
        dg::blas2::gemv( m_righty, phi, m_tempy); //R_y*f
        dg::tensor::scalar_product2d(alpha, lambda, m_tempx, m_tempy, m_chi, lambda, m_tempx, m_tempy, beta, sigma);
    }


    private:
    Matrix m_leftx, m_lefty, m_rightx, m_righty, m_jumpX, m_jumpY;
    Container m_weights, m_precond;
    Container m_tempx, m_tempy, m_temp;
    SparseTensor<Container> m_chi;
    Container m_sigma, m_vol;
    value_type m_jfactor;
    bool m_chi_weight_jump;
};

///@copydoc Elliptic
///@ingroup matrixoperators
template <class Geometry, class Matrix, class Container>
using Elliptic2d = Elliptic<Geometry, Matrix, Container>;

//Elliptic3d is tested in inc/geometries/elliptic3d_t.cu
/**
 * @brief A 3d negative elliptic differential operator \f$ -\nabla \cdot ( \mathbf{\chi}\cdot \nabla ) \f$
 *
 * @ingroup matrixoperators
 *
 * The term discretized is \f[ -\nabla \cdot ( \mathbf{\chi} \cdot \nabla ) \f] where \f$
 * \nabla \f$ is the two-dimensional nabla and \f$\chi = \sigma
 * \mathbf{\tau}\f$ is a (possibly spatially dependent) tensor with scalar part
 * \f$ \sigma\f$ (usually the volume form) and tensor part \f$ \tau\f$ (usually
 * the inverse metric). In general coordinates that means
 * \f[ -\frac{1}{\sqrt{g}}\left(
 * \partial_x\left(\sqrt{g}\left(\chi^{xx}\partial_x + \chi^{xy}\partial_y + \chi^{xz}\partial_z \right)\right)
 + \partial_y\left(\sqrt{g}\left(\chi^{yx}\partial_x + \chi^{yy}\partial_y + \chi^{yz}\partial_z \right)\right)
 + \partial_z\left(\sqrt{g}\left(\chi^{zx}\partial_x + \chi^{zy}\partial_y + \chi^{zz}\partial_z \right)\right)
 \right)\f]
 is discretized.
 Per default, \f$ \chi = \sqrt{g} g^{-1}\f$ but you can set it to any tensor
 you like (in order for the operator to be invertible \f$\chi\f$ should be
 symmetric and positive definite though).

 * @sa Our theory guide <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a> on overleaf holds a detailed derivation

 Note that the local discontinuous Galerkin discretization adds so-called jump terms
 \f[ D^\dagger \chi D + \alpha\chi_{on/off} J \f]
 where \f$\alpha\f$  is a scale factor ( = jfactor), \f$ D \f$ contains the discretizations of the above derivatives, and \f$ J\f$ is a self-adjoint matrix.
 (The symmetric part of \f$J\f$ is added @b before the volume element is divided). The adjoint of a matrix is defined with respect to the volume element including dG weights.
 Usually the default \f$ \alpha=1 \f$ is a good choice.
 However, in some cases, e.g. when \f$ \chi \f$ exhibits very large variations
 \f$ \alpha=0.1\f$ or \f$ \alpha=0.01\f$ might be better values.
 In a time dependent problem the value of \f$\alpha\f$ determines the
 numerical diffusion, i.e. for too low values numerical oscillations may appear.
 The \f$ \chi_{on/off} \f$ in the jump term serves to weight the jump term with \f$ \chi \f$. This can be switched either on or off with off being the default.
 Also note that a forward discretization has more diffusion than a centered discretization.

 The following code snippet demonstrates the use of \c Elliptic3d in an inversion problem
 * @snippet elliptic_b.cu invert
 * @copydoc hide_geometry_matrix_container
 * This class has the \c SelfMadeMatrixTag so it can be used in \c blas2::symv functions
 * and thus in a conjugate gradient solver.
 * @note The constructors initialize \f$ \chi=\sqrt{g}g^{-1}\f$ so that a
 * negative laplacian operator results
 * @note The inverse of \f$ \sigma\f$ makes a good general purpose preconditioner
 * @note the jump term \f$ \alpha J\f$  adds artificial numerical diffusion as discussed above
 * @note Since the pattern arises quite often (because of the ExB velocity \f$ u_E^2\f$ in the ion gyro-centre potential)
 * this class also can compute the variation integrand \f$ \lambda^2\nabla \phi\cdot \tau\cdot\nabla\phi\f$
 * @attention Pay attention to the negative sign which is necessary to make the matrix @b positive @b definite
 */
template <class Geometry, class Matrix, class Container>
class Elliptic3d
{
    public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    Elliptic3d(){}
    /**
     * @brief Construct from Grid
     *
     * @param g The Grid; boundary conditions are taken from here
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * the direction of the z derivative is always \c dg::centered
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @param chi_weight_jump If true, the Jump terms are multiplied with the Chi matrix, else it is ignored
     * @note chi is assumed the metric per default
     */
    Elliptic3d( const Geometry& g, direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false):
        Elliptic3d( g, g.bcx(), g.bcy(), g.bcz(), dir, jfactor, chi_weight_jump)
    {
    }

    /**
     * @brief Construct from grid and boundary conditions
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param bcz boundary contition in z
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * the direction of the z derivative is always \c dg::centered
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @param chi_weight_jump If true, the Jump terms are multiplied with the Chi matrix, else it is ignored
     * @note chi is the metric tensor multiplied by the volume element per default
     */
    Elliptic3d( const Geometry& g, bc bcx, bc bcy, bc bcz, direction dir = forward, value_type jfactor = 1., bool chi_weight_jump = false)
    {
        m_jfactor=jfactor;
        m_chi_weight_jump = chi_weight_jump;
        dg::blas2::transfer( dg::create::dx( g, inverse( bcx), inverse(dir)), m_leftx);
        dg::blas2::transfer( dg::create::dy( g, inverse( bcy), inverse(dir)), m_lefty);
        dg::blas2::transfer( dg::create::dz( g, inverse( bcz), inverse(dg::centered)), m_leftz);
        dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
        dg::blas2::transfer( dg::create::dy( g, bcy, dir), m_righty);
        dg::blas2::transfer( dg::create::dz( g, bcz, dg::centered), m_rightz);
        dg::blas2::transfer( dg::create::jumpX( g, bcx),   m_jumpX);
        dg::blas2::transfer( dg::create::jumpY( g, bcy),   m_jumpY);

        dg::assign( dg::create::volume(g),        m_weights);
        dg::assign( dg::evaluate( dg::one, g),    m_precond);
        m_temp = m_tempx = m_tempy = m_tempz = m_weights;
        m_chi=g.metric();
        m_sigma = m_vol = dg::tensor::volume(m_chi);
    }
    ///@copydoc Elliptic::construct()
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = Elliptic3d( std::forward<Params>( ps)...);
    }

    ///@copydoc Elliptic2d::set_chi(const ContainerType0&)
    template<class ContainerType0>
    void set_chi( const ContainerType0& sigma)
    {
        dg::blas1::pointwiseDot( sigma, m_vol, m_sigma);
        //update preconditioner
        dg::blas1::pointwiseDivide( 1., sigma, m_precond);
        // sigma is possibly zero, which will invalidate the preconditioner
        // it is important to call this blas1 function because it can
        // overwrite NaN in m_precond in the next update
    }

    ///@copydoc Elliptic2d::set_chi(const SparseTensor<ContainerType0>&)
    template<class ContainerType0>
    void set_chi( const SparseTensor<ContainerType0>& tau)
    {
        m_chi = SparseTensor<Container>(tau);
    }

    ///@copydoc Elliptic::weights()
    const Container& weights()const {
        return m_weights;
    }
    ///@copydoc Elliptic::precond()
    const Container& precond()const {
        return m_precond;
    }
    ///@copydoc Elliptic::set_jfactor()
    void set_jfactor( value_type new_jfactor) {m_jfactor = new_jfactor;}
    ///@copydoc Elliptic::get_jfactor()
    value_type get_jfactor() const {return m_jfactor;}
    ///@copydoc Elliptic::set_jump_weighting()
    void set_jump_weighting( bool jump_weighting) {m_chi_weight_jump = jump_weighting;}
    ///@copydoc Elliptic::get_jump_weighting()
    bool get_jump_weighting() const {return m_chi_weight_jump;}

    /**
     * @brief Restrict the problem to the first 2 dimensions
     *
     * This effectively makes the behaviour of dg::Elliptic3d
     * identical to the dg::Elliptic class.
     * @param compute_in_2d if true, the symv function avoids the derivative in z, false reverts to the original behaviour.
     */
    void set_compute_in_2d( bool compute_in_2d ) {
        m_multiplyZ = !compute_in_2d;
    }

    ///@copydoc Elliptic::symv(const ContainerType0&,ContainerType1&)
    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y){
        symv( 1, x, 0, y);
    }
    ///@copydoc Elliptic::symv(value_type,const ContainerType0&,value_type,ContainerType1&)
    template<class ContainerType0, class ContainerType1>
    void symv( value_type alpha, const ContainerType0& x, value_type beta, ContainerType1& y)
    {
        //compute gradient
        dg::blas2::gemv( m_rightx, x, m_tempx); //R_x*f
        dg::blas2::gemv( m_righty, x, m_tempy); //R_y*f
        if( m_multiplyZ )
        {
            dg::blas2::gemv( m_rightz, x, m_tempz); //R_z*f

            //multiply with tensor (note the alias)
            dg::tensor::multiply3d(m_sigma, m_chi, m_tempx, m_tempy, m_tempz, 0., m_tempx, m_tempy, m_tempz);
            //now take divergence
            dg::blas2::symv( -1., m_leftz, m_tempz, 0., m_temp);
            dg::blas2::symv( -1., m_lefty, m_tempy, 1., m_temp);
        }
        else
        {
            dg::tensor::multiply2d(m_sigma, m_chi, m_tempx, m_tempy, 0., m_tempx, m_tempy);
            dg::blas2::symv( -1.,m_lefty, m_tempy, 0., m_temp);
        }
        dg::blas2::symv( -1., m_leftx, m_tempx, 1., m_temp);

        //add jump terms
        if( 0 != m_jfactor )
        {
            if(m_chi_weight_jump)
            {
                dg::blas2::symv( m_jfactor, m_jumpX, x, 0., m_tempx);
                dg::blas2::symv( m_jfactor, m_jumpY, x, 0., m_tempy);
                dg::tensor::multiply2d(m_sigma, m_chi, m_tempx, m_tempy, 0., m_tempx, m_tempy);
                dg::blas1::axpbypgz(1.0,m_tempx,1.0,m_tempy,1.0,m_temp);
            }
            else
            {
                dg::blas2::symv( m_jfactor, m_jumpX, x, 1., m_temp);
                dg::blas2::symv( m_jfactor, m_jumpY, x, 1., m_temp);
            }
        }
        dg::blas1::pointwiseDivide( alpha, m_temp, m_vol, beta, y);
    }

    ///@copydoc Elliptic::variation(const ContainerType0&,ContainerType1&)
    template<class ContainerType0, class ContainerType1>
    void variation(const ContainerType0& phi, ContainerType1& sigma){
        variation(1.,1., phi, 0., sigma);
    }
    ///@copydoc Elliptic::variation(const ContainerTypeL&,const ContainerType0&,ContainerType1&){
    template<class ContainerTypeL, class ContainerType0, class ContainerType1>
    void variation(const ContainerTypeL& lambda, const ContainerType0& phi, ContainerType1& sigma){
        variation(1.,lambda, phi, 0., sigma);
    }
    ///@copydoc Elliptic::variation(value_type,const ContainerTypeL&,const ContainerType0&,value_type,ContainerType1&)
    template<class ContainerTypeL, class ContainerType0, class ContainerType1>
    void variation(value_type alpha, const ContainerTypeL& lambda, const ContainerType0& phi, value_type beta, ContainerType1& sigma)
    {
        dg::blas2::gemv( m_rightx, phi, m_tempx); //R_x*f
        dg::blas2::gemv( m_righty, phi, m_tempy); //R_y*f
        if( m_multiplyZ)
            dg::blas2::gemv( m_rightz, phi, m_tempz); //R_y*f
        else
            dg::blas1::scal( m_tempz, 0.);
        dg::tensor::scalar_product3d(alpha, lambda,  m_tempx, m_tempy, m_tempz, m_chi, lambda, m_tempx, m_tempy, m_tempz, beta, sigma);
    }

    private:
    Matrix m_leftx, m_lefty, m_leftz, m_rightx, m_righty, m_rightz, m_jumpX, m_jumpY;
    Container m_weights, m_precond;
    Container m_tempx, m_tempy, m_tempz, m_temp;
    SparseTensor<Container> m_chi;
    Container m_sigma, m_vol;
    value_type m_jfactor;
    bool m_multiplyZ = true;
    bool m_chi_weight_jump;
};
///@cond
template< class G, class M, class V>
struct TensorTraits< Elliptic<G, M, V> >
{
    using value_type      = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};

template< class G, class M, class V>
struct TensorTraits< Elliptic3d<G, M, V> >
{
    using value_type      = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
///@endcond

} //namespace dg
