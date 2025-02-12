#pragma once

#include "blas.h"
#include "enums.h"
#include "backend/memory.h"
#include "topology/evaluation.h"
#include "topology/derivativesA.h"
#ifdef MPI_VERSION
#include "topology/mpi_evaluation.h"
#endif
#include "topology/geometry.h"

/*! @file

  @brief General negative elliptic operators
  */
namespace dg
{
    //TODO Elliptic can be made complex aware with a 2nd complex ContainerType
// Note that there are many tests for this file : elliptic2d_b,
// elliptic2d_mpib, elliptic_b, elliptic_mpib, ellipticX2d_b
// And don't forget inc/geometries/elliptic3d_t (testing alignment and
// projection tensors as Chi) geometry_elliptic_b, geometry_elliptic_mpib,
// and geometryX_elliptic_b and geometryX_refined_elliptic_b


/*!
 * @class hide_note_jump
 *
 * @sa Our theory guide <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a> on overleaf holds a detailed derivation
 *
 Note that the local discontinuous Galerkin discretization adds so-called jump terms
 \f[ D^\dagger \chi D + \alpha\chi_{on/off} J \f]
 where \f$\alpha\f$  is a scale factor ( = jfactor), \f$ D \f$ contains the discretizations of the above derivatives, and \f$ J\f$ is a self-adjoint matrix.
 (\f$J\f$ is added @b before the volume element is divided). The adjoint of a matrix is defined with respect to the volume element including dG weights.
 Usually the default \f$ \alpha=1 \f$ is a good choice.
 However, in some cases, e.g. when \f$ \sigma \f$ exhibits very large variations
 \f$ \alpha=0.1\f$ or \f$ \alpha=0.01\f$ might be better values.
 In a time dependent problem the value of \f$\alpha\f$ determines the
 numerical diffusion, i.e. for too low values numerical oscillations may appear.
 The \f$ \chi_{on/off} \f$ in the jump term serves to weight the jump term with \f$ \chi \f$. This can be switched either on or off with off being the default.
 Also note that a forward discretization has more diffusion than a centered discretization.
 */
/**
 * @brief A 1d negative elliptic differential operator \f$ -\partial_x ( \chi \partial_x ) \f$
 *
 * @ingroup matrixoperators
 *
 * The term discretized is \f[ -\partial_x ( \chi \partial_x ) \f] where \f$
 * \partial_x \f$ is the one-dimensional derivative and \f$\chi\f$
 * is a scalar function
 * Per default, \f$ \chi = 1\f$ but you can set it to any value
 you like (in order for the operator to be invertible \f$\chi\f$ should be
  strictly positive though).
 * @copydoc hide_note_jump
 * @copydoc hide_geometry_matrix_container
 * This class has the \c SelfMadeMatrixTag so it can be used in \c blas2::symv functions
 * and thus in a conjugate gradient solver.
 * @note The constructors initialize \f$ \chi=1\f$ so that a
 * negative laplacian operator results
 * @note The inverse of \f$ \chi\f$ makes a good general purpose preconditioner
 * @attention Pay attention to the negative sign which is necessary to make the matrix @b positive @b definite
 */
template <class Geometry, class Matrix, class Container>
class Elliptic1d
{
    public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    Elliptic1d() = default;
    /**
     * @brief Construct from Grid
     *
     * @param g The Grid, boundary conditions are taken from here
     * (can be 2d or 3d grid, but the volume form is always 1 and the 2nd and
     * 3rd dimension are trivially parallel)
     * @param dir Direction of the right first derivative in x
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @note chi is one by default
     */
    Elliptic1d( const Geometry& g,
        direction dir = forward, value_type jfactor=1.):
        Elliptic1d( g, g.bcx(), dir, jfactor)
    {
    }

    /**
     * @brief Construct from grid and boundary conditions
     * @param g The Grid (can be 2d or 3d grid, but the volume form is always 1
     * and the 2nd and 3rd dimension are trivially parallel)
     * @param bcx boundary condition in x
     * @param dir Direction of the right first derivative in x
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @note chi is one by default
     */
    Elliptic1d( const Geometry& g, bc bcx,
        direction dir = forward,
        value_type jfactor=1.)
    {
        m_jfactor=jfactor;
        dg::blas2::transfer( dg::create::dx( g, inverse( bcx), inverse(dir)), m_leftx);
        dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
        dg::blas2::transfer( dg::create::jumpX( g, bcx),   m_jumpX);

        dg::assign( dg::create::weights(g),       m_weights);
        dg::assign( dg::evaluate( dg::one, g),    m_precond);
        m_tempx = m_sigma = m_precond;
    }

    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = Elliptic1d( std::forward<Params>( ps)...);
    }

    /**
     * @brief Change scalar part Chi
     *
     * @param sigma The new scalar part \f$\chi\f$
     * @attention If some or all elements of sigma are zero the preconditioner
     * is invalidated and the operator can no longer be inverted (due to divide by zero)
     * until \c set_chi is called with a positive sigma again.
     * The symv function can always be called, however, if sigma is zero you likely also want to set
     * the \c jfactor to 0 because the jumps in phi may not vanish and then pollute the result.
     * @tparam ContainerType0 must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0>
    void set_chi( const ContainerType0& sigma)
    {
        dg::blas1::copy( sigma, m_sigma);
        //update preconditioner
        dg::blas1::pointwiseDivide( 1., sigma, m_precond);
        // sigma is possibly zero, which will invalidate the preconditioner
        // it is important to call this blas1 function because it can
        // overwrite NaN in m_precond in the next update
    }

    /**
     * @brief Return the weights making the operator self-adjoint
     * @return weights
     */
    const Container& weights()const {
        return m_weights;
    }
    /**
     * @brief Return the default preconditioner to use in conjugate gradient
     *
     * Currently returns 1 divided by the scalar part \f$ \sigma\f$.
     * This is especially good when \f$ \sigma\f$ exhibits large amplitudes or
     * variations
     * @return the inverse of \f$\sigma\f$.
     */
    const Container& precond()const {
        return m_precond;
    }
    ///@copydoc Elliptic2d::set_jfactor()
    void set_jfactor( value_type new_jfactor) {m_jfactor = new_jfactor;}
    ///@copydoc Elliptic2d::get_jfactor()
    value_type get_jfactor() const {return m_jfactor;}
    ///@copydoc Elliptic2d::operator()(const ContainerType0&,ContainerType1&)
    template<class ContainerType0, class ContainerType1>
    void operator()( const ContainerType0& x, ContainerType1& y){
        symv( 1, x, 0, y);
    }

    ///@copydoc Elliptic2d::symv(const ContainerType0&,ContainerType1&)
    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y){
        symv( 1, x, 0, y);
    }
    ///@copydoc Elliptic2d::symv(value_type,const ContainerType0&,value_type,ContainerType1&)
    template<class ContainerType0, class ContainerType1>
    void symv( value_type alpha, const ContainerType0& x, value_type beta, ContainerType1& y)
    {
        dg::blas2::gemv( m_rightx, x, m_tempx);
        dg::blas1::pointwiseDot( m_tempx, m_sigma, m_tempx);
        dg::blas2::symv( -alpha, m_leftx, m_tempx, beta, y);
        //add jump terms
        if( 0.0 != m_jfactor )
        {
            dg::blas2::symv( m_jfactor*alpha, m_jumpX, x, 1., y);
        }
    }

    private:
    Matrix m_leftx, m_rightx, m_jumpX;
    Container m_weights, m_precond;
    Container m_tempx;
    Container m_sigma;
    value_type m_jfactor;
};

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

 @copydoc hide_note_jump

 The following code snippet demonstrates the use of \c Elliptic in an inversion problem
 * @snippet elliptic2d_b.cpp pcg
 * @copydoc hide_geometry_matrix_container
 * This class has the \c SelfMadeMatrixTag so it can be used in \c blas2::symv functions
 * and thus in a conjugate gradient solver.
 * @note The constructors initialize \f$ \chi=\sqrt{g}g^{-1}\f$ so that a
 * negative laplacian operator results
 * @note The inverse of \f$ \sigma\f$ makes a good general purpose preconditioner
 * @note Since the pattern arises quite often (because of the ExB velocity \f$ u_E^2\f$ in the ion gyro-centre potential)
 * this class also can compute the variation integrand \f$ \lambda^2\nabla \phi\cdot \tau\cdot\nabla\phi\f$
 * @attention Pay attention to the negative sign which is necessary to make the matrix @b positive @b definite
 */
template <class Geometry, class Matrix, class Container>
class Elliptic2d
{
    public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    Elliptic2d() = default;
    /**
     * @brief Construct from Grid
     *
     * Initialize \f$ \chi=\sqrt{g}g^{-1}\f$ so that a negative laplacian operator results
     *
     * @param g The Grid, boundary conditions are taken from here
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @param chi_weight_jump If true, the Jump terms are multiplied with the Chi matrix, else it is ignored
     * @note The grid can be a 3d grid, then the 3rd row and column of \f$
     * \chi\f$ (and / or the metric) are ignored in the discretization, which
     * makes the 3rd dimension trivially parallel; the volume form will be the
     * full 3d volume form though)
     */
    Elliptic2d( const Geometry& g,
        direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false):
        Elliptic2d( g, g.bcx(), g.bcy(), dir, jfactor, chi_weight_jump)
    {
    }

    /**
     * @brief Construct from grid and boundary conditions
     *
     * Initialize \f$ \chi=\sqrt{g}g^{-1}\f$ so that a negative laplacian operator results
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @param chi_weight_jump If true, the Jump terms are multiplied with the Chi matrix, else it is ignored
     * @note The grid can be a 3d grid, then the 3rd row and column of \f$
     * \chi\f$ (and / or the metric) are ignored in the discretization, which
     * makes the 3rd dimension trivially parallel; the volume form will be the
     * full 3d volume form though)
     */
    Elliptic2d( const Geometry& g, bc bcx, bc bcy,
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

    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = Elliptic2d( std::forward<Params>( ps)...);
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
     * is invalidated and the operator can no longer be inverted (due to divide by zero)
     * until \c set_chi is called with a positive sigma again.
     * The symv function can always be called, however, if sigma is zero you likely also want to set
     * the \c jfactor to 0 because the jumps in phi may not vanish and then pollute the result.
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

///@copydoc Elliptic2d
///@ingroup matrixoperators
template <class Geometry, class Matrix, class Container>
using Elliptic = Elliptic2d<Geometry, Matrix, Container>;

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

 @copydoc hide_note_jump


 The following code snippet demonstrates the use of \c Elliptic3d in an inversion problem
 * @snippet elliptic_b.cpp invert
 * @copydoc hide_geometry_matrix_container
 * This class has the \c SelfMadeMatrixTag so it can be used in \c blas2::symv functions
 * and thus in a conjugate gradient solver.
 * @note The constructors initialize \f$ \chi=\sqrt{g}g^{-1}\f$ so that a
 * negative laplacian operator results
 * @note The inverse of \f$ \sigma\f$ makes a good general purpose preconditioner
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
    Elliptic3d() = default;
    /**
     * @brief Construct from Grid
     *
     * @param g The Grid; boundary conditions are taken from here
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * the direction of the z derivative is \c dg::centered if \c nz = 1
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
        // MW we should create an if guard for nx, ny, or nz = 1 and periodic boundaries
        m_jfactor=jfactor;
        m_chi_weight_jump = chi_weight_jump;
        dg::blas2::transfer( dg::create::dx( g, inverse( bcx), inverse(dir)), m_leftx);
        dg::blas2::transfer( dg::create::dy( g, inverse( bcy), inverse(dir)), m_lefty);
        dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
        dg::blas2::transfer( dg::create::dy( g, bcy, dir), m_righty);
        dg::blas2::transfer( dg::create::jumpX( g, bcx),   m_jumpX);
        dg::blas2::transfer( dg::create::jumpY( g, bcy),   m_jumpY);
        if( g.nz() == 1)
        {
            dg::blas2::transfer( dg::create::dz( g, bcz, dg::centered), m_rightz);
            dg::blas2::transfer( dg::create::dz( g, inverse( bcz), inverse(dg::centered)), m_leftz);
            m_addJumpZ = false;
        }
        else
        {
            dg::blas2::transfer( dg::create::dz( g, bcz, dir), m_rightz);
            dg::blas2::transfer( dg::create::dz( g, inverse( bcz), inverse(dir)), m_leftz);
            dg::blas2::transfer( dg::create::jumpZ( g, bcz),   m_jumpZ);
            m_addJumpZ = true;
        }

        dg::assign( dg::create::volume(g),        m_weights);
        dg::assign( dg::evaluate( dg::one, g),    m_precond);
        m_temp = m_tempx = m_tempy = m_tempz = m_weights;
        m_chi=g.metric();
        m_sigma = m_vol = dg::tensor::volume(m_chi);
    }
    ///@copydoc hide_construct
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

    ///@copydoc Elliptic2d::weights()
    const Container& weights()const {
        return m_weights;
    }
    ///@copydoc Elliptic2d::precond()
    const Container& precond()const {
        return m_precond;
    }
    ///@copydoc Elliptic2d::set_jfactor()
    void set_jfactor( value_type new_jfactor) {m_jfactor = new_jfactor;}
    ///@copydoc Elliptic2d::get_jfactor()
    value_type get_jfactor() const {return m_jfactor;}
    ///@copydoc Elliptic2d::set_jump_weighting()
    void set_jump_weighting( bool jump_weighting) {m_chi_weight_jump = jump_weighting;}
    ///@copydoc Elliptic2d::get_jump_weighting()
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

    ///@copydoc Elliptic2d::symv(const ContainerType0&,ContainerType1&)
    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y){
        symv( 1, x, 0, y);
    }
    ///@copydoc Elliptic2d::symv(value_type,const ContainerType0&,value_type,ContainerType1&)
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
                if( m_addJumpZ)
                {
                    dg::blas2::symv( m_jfactor, m_jumpZ, x, 0., m_tempz);
                    dg::tensor::multiply3d(m_sigma, m_chi, m_tempx, m_tempy,
                            m_tempz, 0., m_tempx, m_tempy, m_tempz);
                }
                else
                    dg::tensor::multiply2d(m_sigma, m_chi, m_tempx, m_tempy,
                            0., m_tempx, m_tempy);

                dg::blas1::axpbypgz(1., m_tempx, 1., m_tempy, 1., m_temp);
                if( m_addJumpZ)
                    dg::blas1::axpby( 1., m_tempz, 1., m_temp);
            }
            else
            {
                dg::blas2::symv( m_jfactor, m_jumpX, x, 1., m_temp);
                dg::blas2::symv( m_jfactor, m_jumpY, x, 1., m_temp);
                if( m_addJumpZ)
                    dg::blas2::symv( m_jfactor, m_jumpZ, x, 1., m_temp);
            }
        }
        dg::blas1::pointwiseDivide( alpha, m_temp, m_vol, beta, y);
    }

    ///@copydoc Elliptic2d::variation(const ContainerType0&,ContainerType1&)
    template<class ContainerType0, class ContainerType1>
    void variation(const ContainerType0& phi, ContainerType1& sigma){
        variation(1.,1., phi, 0., sigma);
    }
    ///@copydoc Elliptic2d::variation(const ContainerTypeL&,const ContainerType0&,ContainerType1&){
    template<class ContainerTypeL, class ContainerType0, class ContainerType1>
    void variation(const ContainerTypeL& lambda, const ContainerType0& phi, ContainerType1& sigma){
        variation(1.,lambda, phi, 0., sigma);
    }
    ///@copydoc Elliptic2d::variation(value_type,const ContainerTypeL&,const ContainerType0&,value_type,ContainerType1&)
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
    Matrix m_leftx, m_lefty, m_leftz, m_rightx, m_righty, m_rightz, m_jumpX, m_jumpY, m_jumpZ;
    Container m_weights, m_precond;
    Container m_tempx, m_tempy, m_tempz, m_temp;
    SparseTensor<Container> m_chi;
    Container m_sigma, m_vol;
    value_type m_jfactor;
    bool m_multiplyZ = true, m_addJumpZ = false;
    bool m_chi_weight_jump;
};
///@cond
template< class G, class M, class V>
struct TensorTraits< Elliptic1d<G, M, V> >
{
    using value_type      = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
template< class G, class M, class V>
struct TensorTraits< Elliptic2d<G, M, V> >
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
