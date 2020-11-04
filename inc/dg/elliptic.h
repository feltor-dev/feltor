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

/**
 * @brief A 2d negative elliptic differential operator
 *
 * @ingroup matrixoperators
 *
 * The term discretized is \f[ -\nabla \cdot ( \chi \nabla ) \f]
 * where \f$ \nabla \f$ is the two-dimensional nabla and \f$\chi\f$ is a
 * (possibly spatially dependent) tensor.
 * In general coordinates that means
 * \f[ -\frac{1}{\sqrt{g}}\left(
 * \partial_x\left(\sqrt{g}\left(\chi^{xx}\partial_x + \chi^{xy}\partial_y \right)\right)
 + \partial_y\left(\sqrt{g} \left(\chi^{yx}\partial_x + \chi^{yy}\partial_y \right)\right) \right)\f]
 is discretized.
 Per default, \f$ \chi\f$ is the metric tensor but you can set it to any tensor
 you like (in order for the operator to be invertible \f$\chi\f$ should be
 symmetric and positive definite though).
 Note that the local discontinuous Galerkin discretization adds so-called jump terms
 \f[ D^\dagger \chi D + \alpha J \f]
 where \f$\alpha\f$  is a scale factor ( = jfactor), \f$ D \f$ contains the discretizations of the above derivatives, and \f$ J\f$ is a self-adjoint matrix.
 (The symmetric part of \f$J\f$ is added @b before the volume element is divided). The adjoint of a matrix is defined with respect to the volume element including dG weights.
 Usually, the default \f$ \alpha=1 \f$ is a good choice.
 However, in some cases, e.g. when \f$ \chi \f$ exhibits very large variations
 \f$ \alpha=0.1\f$ or \f$ \alpha=0.01\f$ might be better values.
 In a time dependent problem the value of \f$\alpha\f$ determines the
 numerical diffusion, i.e. for too low values numerical oscillations may appear.
 Also note that a forward discretization has more diffusion than a centered discretization.

 The following code snippet demonstrates the use of \c Elliptic in an inversion problem
 * @snippet elliptic2d_b.cu invert
 * @copydoc hide_geometry_matrix_container
 * This class has the \c SelfMadeMatrixTag so it can be used in blas2::symv functions
 * and thus in a conjugate gradient solver.
 * @note The constructors initialize \f$ \chi=1\f$ so that a negative laplacian operator
 * results
 * @note The inverse of \f$ \chi\f$ makes a good general purpose preconditioner
 * @note the jump term \f$ \alpha J\f$  adds artificial numerical diffusion as discussed above
 * @attention Pay attention to the negative sign which is necessary to make the matrix @b positive @b definite
 *
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
     * @param no choose \c dg::normed if you want to directly use the object,
     *  \c dg::not_normed if you want to invert the elliptic equation
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @note chi is assumed 1 per default
     */
    Elliptic( const Geometry& g, norm no = not_normed,
        direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false):
        Elliptic( g, g.bcx(), g.bcy(), no, dir, jfactor, chi_weight_jump)
    {
    }

    /**
     * @brief Construct from grid and boundary conditions
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param no choose \c dg::normed if you want to directly use the object,
     *  \c dg::not_normed if you want to invert the elliptic equation
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @note chi is assumed 1 per default
     */
    Elliptic( const Geometry& g, bc bcx, bc bcy,
        norm no = not_normed, direction dir = forward,
        value_type jfactor=1., bool chi_weight_jump = false)
    {
        m_no=no, m_jfactor=jfactor;
        m_chi_weight_jump = chi_weight_jump;
        dg::blas2::transfer( dg::create::dx( g, inverse( bcx), inverse(dir)), m_leftx);
        dg::blas2::transfer( dg::create::dy( g, inverse( bcy), inverse(dir)), m_lefty);
        dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
        dg::blas2::transfer( dg::create::dy( g, bcy, dir), m_righty);
        dg::blas2::transfer( dg::create::jumpX( g, bcx),   m_jumpX);
        dg::blas2::transfer( dg::create::jumpY( g, bcy),   m_jumpY);

        dg::assign( dg::create::inv_volume(g),    m_inv_weights);
        dg::assign( dg::create::volume(g),        m_weights);
        dg::assign( dg::create::inv_weights(g),   m_precond);
        m_temp = m_tempx = m_tempy = m_inv_weights;
        m_chi=g.metric();
        m_vol=dg::tensor::volume(m_chi);
        dg::tensor::scal( m_chi, m_vol);
        dg::assign( dg::create::weights(g), m_weights_wo_vol);
        dg::assign( dg::evaluate(dg::one, g), m_sigma);
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

    ///@copydoc Elliptic3d::set_chi(const ContainerType0&)
    template<class ContainerType0>
    void set_chi( const ContainerType0& sigma)
    {
        dg::blas1::pointwiseDivide( sigma, m_sigma, m_tempx);
        //update preconditioner
        dg::blas1::pointwiseDivide( m_precond, m_tempx, m_precond);
        dg::tensor::scal( m_chi, m_tempx);
        dg::blas1::copy( sigma, m_sigma);
    }
    /**
     * @copydoc Elliptic3d::set_chi(const SparseTensor<ContainerType0>&)
     * @note the 3d parts in \c tau will be ignored
     */
    template<class ContainerType0>
    void set_chi( const SparseTensor<ContainerType0>& tau)
    {
        m_chi = SparseTensor<Container>(tau);
        dg::tensor::scal( m_chi, m_sigma);
        dg::tensor::scal( m_chi, m_vol);
    }

    /**
     * @brief Return the vector missing in the un-normed symmetric matrix
     *
     * i.e. the inverse of the weights() function
     * @return inverse volume form including inverse weights
     */
    const Container& inv_weights()const {
        return m_inv_weights;
    }
    /**
     * @brief Return the vector making the matrix symmetric
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
     * Currently returns the inverse weights without volume elment divided by the scalar part of \f$ \chi\f$.
     * This is especially good when \f$ \chi\f$ exhibits large amplitudes or variations
     * @return the inverse of \f$\chi\f$.
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
        dg::tensor::multiply2d(m_chi, m_tempx, m_tempy, m_tempx, m_tempy);

        //now take divergence
        dg::blas2::symv( m_lefty, m_tempy, m_temp);
        dg::blas2::symv( -1., m_leftx, m_tempx, -1., m_temp);

        //add jump terms
        if(m_chi_weight_jump)
        {
            dg::blas2::symv( m_jfactor, m_jumpX, x, 0., m_tempx);
            dg::blas2::symv( m_jfactor, m_jumpY, x, 0., m_tempy);
            dg::tensor::multiply2d(m_chi, m_tempx, m_tempy, m_tempx, m_tempy);
            dg::blas1::axpbypgz(1.0,m_tempx,1.0,m_tempy,1.0,m_temp);
        } 
        else
        {
            dg::blas2::symv( m_jfactor, m_jumpX, x, 1., m_temp);
            dg::blas2::symv( m_jfactor, m_jumpY, x, 1., m_temp);
        }
        
        if( m_no == normed)
            dg::blas1::pointwiseDivide( alpha, m_temp, m_vol, beta, y);
        if( m_no == not_normed)//multiply weights without volume
            dg::blas1::pointwiseDot( alpha, m_weights_wo_vol, m_temp, beta, y);
    }

    /**
     * @brief Compute elliptic term with a possibly zero prefactor and add to output
     *
     * i.e this function computes \f[ y = -\alpha\nabla \cdot ( \sigma\chi \nabla x )  + \beta y\f]
     * This is in principle possible also with the \c set_chi() and \c symv() functions
     * however sometimes you have a \c sigma with explicit zeros or negative values.
     * Then you need to use this function because \c set_chi() won't allow a \c sigma with zeros
     * @note This function does not change the internal \c chi tensor
     * @param alpha a scalar
     * @param sigma The prefactor for the \c chi tensor
     * @param x left-hand-side
     * @param beta a scalar
     * @param y result
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    void multiply_sigma( value_type alpha, const ContainerType2& sigma, const ContainerType0& x, value_type beta, ContainerType1& y)
    {
        //compute gradient
        dg::blas2::gemv( m_rightx, x, m_tempx); //R_x*f
        dg::blas2::gemv( m_righty, x, m_tempy); //R_y*f

        //multiply with tensor (note the alias)
        dg::tensor::multiply2d(m_chi, m_tempx, m_tempy, m_tempx, m_tempy);
        //sigma is possibly zero so we don't multiply it to m_chi
        dg::blas1::pointwiseDot( m_tempx, sigma, m_tempx); ///////
        dg::blas1::pointwiseDot( m_tempy, sigma, m_tempy); ///////

        //now take divergence
        dg::blas2::symv( m_lefty, m_tempy, m_temp);
        dg::blas2::symv( -1., m_leftx, m_tempx, -1., m_temp);

        //add jump terms
        if( 0 != m_jfactor )
        {
            if(m_chi_weight_jump)
            {
                dg::blas2::symv( m_jfactor, m_jumpX, x, 0., m_tempx);
                dg::blas2::symv( m_jfactor, m_jumpY, x, 0., m_tempy);
                dg::tensor::multiply2d(m_chi, m_tempx, m_tempy, m_tempx, m_tempy);
                dg::blas1::axpbypgz(1.0,m_tempx,1.0,m_tempy,1.0,m_temp);
            } 
            else
            {   
                dg::blas2::symv( m_jfactor, m_jumpX, x, 1., m_temp);
                dg::blas2::symv( m_jfactor, m_jumpY, x, 1., m_temp);
            }
        }
        if( m_no == normed)
            dg::blas1::pointwiseDivide( alpha, m_temp, m_vol, beta, y);
        if( m_no == not_normed)//multiply weights without volume
            dg::blas1::pointwiseDot( alpha, m_weights_wo_vol, m_temp, beta, y);
    }
    /**
     * @brief Determine if weights are multiplied to make operator symmetric or not
     *
     * @param new_norm new setting
     */
    void set_norm( dg::norm new_norm) {
        m_no = new_norm;
    }
    private:
    Matrix m_leftx, m_lefty, m_rightx, m_righty, m_jumpX, m_jumpY;
    Container m_weights, m_inv_weights, m_precond, m_weights_wo_vol;
    Container m_tempx, m_tempy, m_temp;
    norm m_no;
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
 * @brief A 3d negative elliptic differential operator
 *
 * @ingroup matrixoperators
 *
 * The term discretized is \f[ -\nabla \cdot ( \mathbf \chi\cdot \nabla ) \f]
 * where \f$ \mathbf \chi \f$ is a positive semi-definit tensor.
 * In general coordinates that means
 * \f[ -\frac{1}{\sqrt{g}}\left(
 * \partial_x\left(\sqrt{g}\left(\chi^{xx}\partial_x + \chi^{xy}\partial_y + \chi^{xz}\partial_z \right)\right)
 + \partial_y\left(\sqrt{g}\left(\chi^{yx}\partial_x + \chi^{yy}\partial_y + \chi^{yz}\partial_z \right)\right)
 + \partial_z\left(\sqrt{g}\left(\chi^{zx}\partial_x + \chi^{zy}\partial_y + \chi^{zz}\partial_z \right)\right)
 \right)\f]
 is discretized. Note that the local discontinuous Galerkin discretization adds so-called
 jump terms
 \f[ D^\dagger \chi D + \alpha J \f]
 where \f$\alpha\f$  is a scale factor ( = jfactor), \f$ D \f$ contains the discretizations of the above derivatives, and \f$ J\f$ is a self-adjoint matrix.
 (The symmetric part of \f$J\f$ is added @b before the volume element is divided). The adjoint of a matrix is defined with respect to the volume element including dG weights.
 Usually the default \f$ \alpha=1 \f$ is a good choice.
 However, in some cases, e.g. when \f$ \chi \f$ exhibits very large variations
 \f$ \alpha=0.1\f$ or \f$ \alpha=0.01\f$ might be better values.
 In a time dependent problem the value of \f$\alpha\f$ determines the
 numerical diffusion, i.e. for too low values numerical oscillations may appear.
 Also note that a forward discretization has more diffusion than a centered discretization.

 The following code snippet demonstrates the use of \c Elliptic in an inversion problem
 * @snippet elliptic_b.cu invert
 * @copydoc hide_geometry_matrix_container
 * This class has the \c SelfMadeMatrixTag so it can be used in \c blas2::symv functions
 * and thus in a conjugate gradient solver.
 * @note The constructors initialize \f$ \chi=1\f$ so that a negative laplacian operator
 * results
 * @note the jump term \f$ \alpha J\f$  adds artificial numerical diffusion as discussed above
 * @attention Pay attention to the negative sign which is necessary to make the matrix @b positive @b definite
 *
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
     * @param no choose \c dg::normed if you want to directly use the object,
     *  \c dg::not_normed if you want to invert the elliptic equation
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * the direction of the z derivative is always \c dg::centered
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @note chi is assumed 1 per default
     */
    Elliptic3d( const Geometry& g, norm no = not_normed, direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false):
        Elliptic3d( g, g.bcx(), g.bcy(), g.bcz(), no, dir, jfactor, chi_weight_jump)
    {
    }

    /**
     * @brief Construct from grid and boundary conditions
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param bcz boundary contition in z
     * @param no choose \c dg::normed if you want to directly use the object,
     *  \c dg::not_normed if you want to invert the elliptic equation
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * the direction of the z derivative is always \c dg::centered
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @note chi is the metric tensor multiplied by the volume element per default
     */
    Elliptic3d( const Geometry& g, bc bcx, bc bcy, bc bcz, norm no = not_normed, direction dir = forward, value_type jfactor = 1., bool chi_weight_jump = false)
    {
        m_no=no, m_jfactor=jfactor;
        m_chi_weight_jump = chi_weight_jump;
        dg::blas2::transfer( dg::create::dx( g, inverse( bcx), inverse(dir)), m_leftx);
        dg::blas2::transfer( dg::create::dy( g, inverse( bcy), inverse(dir)), m_lefty);
        dg::blas2::transfer( dg::create::dz( g, inverse( bcz), inverse(dg::centered)), m_leftz);
        dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
        dg::blas2::transfer( dg::create::dy( g, bcy, dir), m_righty);
        dg::blas2::transfer( dg::create::dz( g, bcz, dg::centered), m_rightz);
        dg::blas2::transfer( dg::create::jumpX( g, bcx),   m_jumpX);
        dg::blas2::transfer( dg::create::jumpY( g, bcy),   m_jumpY);

        dg::assign( dg::create::inv_volume(g),    m_inv_weights);
        dg::assign( dg::create::volume(g),        m_weights);
        dg::assign( dg::create::inv_weights(g),   m_precond);
        m_temp = m_tempx = m_tempy = m_tempz = m_inv_weights;
        m_chi=g.metric();
        m_vol=dg::tensor::volume(m_chi);
        dg::tensor::scal( m_chi, m_vol);
        dg::assign( dg::create::weights(g), m_weights_wo_vol);
        dg::assign( dg::evaluate(dg::one, g), m_sigma);
    }
    ///@copydoc Elliptic::construct()
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = Elliptic3d( std::forward<Params>( ps)...);
    }

    /**
     * @brief Change scalar part in Chi tensor
     *
     * Internally, we split the tensor \f$\chi = \sigma\mathbf{\tau}\f$ into
     * a scalar part \f$ \sigma\f$ and a tensor part \f$ \tau\f$ and you can
     * set each part seperately. This functions sets the scalar part.
     *
     * @param sigma The new scalar part in \f$\chi\f$ (all elements must be >0)
     * @tparam ContainerType0 must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0>
    void set_chi( const ContainerType0& sigma)
    {
        dg::blas1::pointwiseDivide( sigma, m_sigma, m_tempx);
        //update preconditioner
        dg::blas1::pointwiseDivide( m_precond, m_tempx, m_precond);
        dg::tensor::scal( m_chi, m_tempx);
        dg::blas1::copy( sigma, m_sigma);
    }
    /**
     * @brief Change tensor part in Chi tensor
     *
     * Internally, we split the tensor \f$\chi = \sigma\mathbf{\tau}\f$ into
     * a scalar part \f$ \sigma\f$ and a tensor part \f$ \tau\f$ and you can
     * set each part seperately. This functions sets the tensor part.
     *
     * @note The class will take care of the volume element in the divergence so do not multiply it to \c tau yourself
     *
     * @param tau The new tensor part in \f$\chi\f$ (must be positive definite)
     * @tparam ContainerType0 must be usable in \c dg::assign to \c Container
     */
    template<class ContainerType0>
    void set_chi( const SparseTensor<ContainerType0>& tau)
    {
        m_chi = SparseTensor<Container>(tau);
        dg::tensor::scal( m_chi, m_sigma);
        dg::tensor::scal( m_chi, m_vol);
    }

    ///@copydoc Elliptic::inv_weights()
    const Container& inv_weights()const {
        return m_inv_weights;
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
            dg::tensor::multiply3d(m_chi, m_tempx, m_tempy, m_tempz, m_tempx, m_tempy, m_tempz);
            //now take divergence
            dg::blas2::symv( -1., m_leftz, m_tempz, 0., m_temp);
            dg::blas2::symv( -1., m_lefty, m_tempy, 1., m_temp);
        }
        else
        {
            dg::tensor::multiply2d(m_chi, m_tempx, m_tempy, m_tempx, m_tempy);
            dg::blas2::symv( -1.,m_lefty, m_tempy, 0., m_temp);
        }
        dg::blas2::symv( -1., m_leftx, m_tempx, 1., m_temp);

        //add jump terms
        if(m_chi_weight_jump)
        {
            dg::blas2::symv( m_jfactor, m_jumpX, x, 0., m_tempx);
            dg::blas2::symv( m_jfactor, m_jumpY, x, 0., m_tempy);
            dg::tensor::multiply2d(m_chi, m_tempx, m_tempy, m_tempx, m_tempy);
            dg::blas1::axpbypgz(1.0,m_tempx,1.0,m_tempy,1.0,m_temp);
        } 
        else
        {
            dg::blas2::symv( m_jfactor, m_jumpX, x, 1., m_temp);
            dg::blas2::symv( m_jfactor, m_jumpY, x, 1., m_temp);
        }
        if( m_no == normed)
            dg::blas1::pointwiseDivide( alpha, m_temp, m_vol, beta, y);
        if( m_no == not_normed)//multiply weights without volume
            dg::blas1::pointwiseDot( alpha, m_weights_wo_vol, m_temp, beta, y);
    }
    ///@copydoc Elliptic::multiply_sigma(value_type,const ContainerType2&,const ContainerType0&,value_type,ContainerType1&)
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    void multiply_sigma( value_type alpha, const ContainerType2& sigma, const ContainerType0& x, value_type beta, ContainerType1& y)
    {
        //compute gradient
        dg::blas2::gemv( m_rightx, x, m_tempx); //R_x*f
        dg::blas2::gemv( m_righty, x, m_tempy); //R_y*f
        if( m_multiplyZ )
        {
            dg::blas2::gemv( m_rightz, x, m_tempz); //R_z*f

            //multiply with tensor (note the alias)
            dg::tensor::multiply3d(m_chi, m_tempx, m_tempy, m_tempz, m_tempx, m_tempy, m_tempz);
            //sigma is possibly zero so we don't multiply it to m_chi
            dg::blas1::pointwiseDot( m_tempx, sigma, m_tempx); ///////
            dg::blas1::pointwiseDot( m_tempy, sigma, m_tempy); ///////
            dg::blas1::pointwiseDot( m_tempz, sigma, m_tempz); ///////
            //now take divergence
            dg::blas2::symv( -1., m_leftz, m_tempz, 0., m_temp);
            dg::blas2::symv( -1., m_lefty, m_tempy, 1., m_temp);
        }
        else
        {
            dg::tensor::multiply2d(m_chi, m_tempx, m_tempy, m_tempx, m_tempy);
            dg::blas1::pointwiseDot( m_tempx, sigma, m_tempx); ///////
            dg::blas1::pointwiseDot( m_tempy, sigma, m_tempy); ///////
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
                dg::tensor::multiply2d(m_chi, m_tempx, m_tempy, m_tempx, m_tempy);
                dg::blas1::axpbypgz(1.0,m_tempx,1.0,m_tempy,1.0,m_temp);
            } 
            else
            {   
                dg::blas2::symv( m_jfactor, m_jumpX, x, 1., m_temp);
                dg::blas2::symv( m_jfactor, m_jumpY, x, 1., m_temp);
            }
        }
        if( m_no == normed)
            dg::blas1::pointwiseDivide( alpha, m_temp, m_vol, beta, y);
        if( m_no == not_normed)//multiply weights without volume
            dg::blas1::pointwiseDot( alpha, m_weights_wo_vol, m_temp, beta, y);
    }

    /**
     * @brief Determine if weights are multiplied to make operator symmetric or not
     *
     * @param new_norm new setting
     */
    void set_norm( dg::norm new_norm) {
        m_no = new_norm;
    }
    
    private:
    Matrix m_leftx, m_lefty, m_leftz, m_rightx, m_righty, m_rightz, m_jumpX, m_jumpY;
    Container m_weights, m_inv_weights, m_precond, m_weights_wo_vol;
    Container m_tempx, m_tempy, m_tempz, m_temp;
    norm m_no;
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
