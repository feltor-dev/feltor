#pragma once

#include "elliptic.h"
#include "helmholtz.h"
#include "multigrid.h"
#include "matrixsqrt.h"

namespace dg
{
 
template <class Geometry, class Matrix, class DiaMatrix, class CooMatrix, class Container, class SubContainer>
class Polarization
{
    public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    Polarization(){}

    Polarization(value_type alpha, std::vector<value_type> eps_gamma, const Geometry& g, norm no = not_normed, direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false, std::string mode = "df")
    {
        construct(alpha, eps_gamma, g,  g.bcx(), g.bcy(), no, dir, jfactor, chi_weight_jump, mode);
    }

    Polarization( value_type alpha, std::vector<value_type> eps_gamma, const Geometry& g, bc bcx, bc bcy, norm no = not_normed, direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false, std::string mode = "df")
    { 
         construct(alpha, eps_gamma, g,  bcx, bcy, no, dir, jfactor, chi_weight_jump, mode);
    }

    void construct(value_type alpha, std::vector<value_type> eps_gamma, const Geometry& g, bc bcx, bc bcy, norm no = not_normed, direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false, std::string mode = "df")
    {
        m_alpha = alpha;
        m_eps_gamma = eps_gamma;
        m_mode = mode;
        m_multi_gamma.resize(3);
        m_multi_g.construct(g,3);
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
        m_temp = m_tempx = m_tempy = m_gamma_x = m_inv_weights;
        m_chi=g.metric();
        m_metric=g.metric();
        m_vol=dg::tensor::volume(m_chi);
        dg::tensor::scal( m_chi, m_vol);
        dg::assign( dg::create::weights(g), m_weights_wo_vol);
        dg::assign( dg::evaluate(dg::one, g), m_sigma);
        for( unsigned u=0; u<3; u++)
        {
            m_multi_gamma[u].construct( m_multi_g.grid(u),bcx, bcy, m_alpha, dir, m_jfactor);
        }
    }
    ///@copydoc PolarizationN3d::set_chi(const ContainerType0&)
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
     * @copydoc PolarizationN3d::set_chi(const SparseTensor<ContainerType0>&)
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
     * @brief Compute the total variation integrand
     *
     * Computes \f[ (\nabla\phi)^2 = \partial_i \phi g^{ij}\partial_j \phi \f]
     * in the plane of a 2x1 product space
     * @param phi function
     * @param varphi may equal phi, contains result on output
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void variation( const ContainerType0& phi, ContainerType1& varphi)
    {
        blas2::symv( m_rightx, phi, m_tempx);
        blas2::symv( m_righty, phi, m_tempy);
        tensor::multiply2d( m_metric, m_tempx, m_tempy, varphi, m_temp);
        blas1::pointwiseDot( 1., varphi, m_tempx, 1., m_temp, m_tempy, 0., varphi);
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
//         dg::blas1::axpby(1.0,x, 0.0, m_gamma_x);
        if (m_mode == "df")
        {
            //Invert first
            if (m_alpha == 0) {
                //compute gradient
                dg::blas2::gemv( m_rightx, x, m_tempx); //R_x*f
                dg::blas2::gemv( m_righty, x, m_tempy); //R_y*f
            }
            else {
                std::vector<unsigned> number = m_multi_g.direct_solve( m_multi_gamma, m_gamma_x, x, m_eps_gamma);
                if(  number[0] == m_multi_g.max_iter())
                    throw dg::Fail( m_eps_gamma[0]);
                //compute gradient
                dg::blas2::gemv( m_rightx, m_gamma_x, m_tempx); //R_x*f
                dg::blas2::gemv( m_righty, m_gamma_x, m_tempy); //R_y*f
            }
        }
//         else if (m_mode == "ff")
//         {
//         }

        //multiply with tensor (note the alias)
        dg::tensor::multiply2d(m_chi, m_tempx, m_tempy, m_tempx, m_tempy);

        //now take divergence
        dg::blas2::symv( m_lefty, m_tempy, m_temp);
        dg::blas2::symv( -1., m_leftx, m_tempx, -1., m_temp);

        //add jump terms
        if (m_alpha == 0) { 
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
        else {
            if(m_chi_weight_jump)
            {
                dg::blas2::symv( m_jfactor, m_jumpX, m_gamma_x, 0., m_tempx);
                dg::blas2::symv( m_jfactor, m_jumpY, m_gamma_x, 0., m_tempy);
                dg::tensor::multiply2d(m_chi, m_tempx, m_tempy, m_tempx, m_tempy);
                dg::blas1::axpbypgz(1.0,m_tempx,1.0,m_tempy,1.0,m_temp);
            } 
            else
            {
                dg::blas2::symv( m_jfactor, m_jumpX, m_gamma_x, 1., m_temp);
                dg::blas2::symv( m_jfactor, m_jumpY, m_gamma_x, 1., m_temp);
            }
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
        //Invert first
        if (m_alpha == 0) {
            //compute gradient
            dg::blas2::gemv( m_rightx, x, m_tempx); //R_x*f
            dg::blas2::gemv( m_righty, x, m_tempy); //R_y*f
        }
        else {
            
            std::vector<unsigned> number = m_multi_g.direct_solve( m_multi_gamma, m_temp, x, m_eps_gamma);
            if(  number[0] == m_multi_g.max_iter())
                throw dg::Fail( m_eps_gamma[0]);
            //compute gradient
            dg::blas2::gemv( m_rightx, m_temp, m_tempx); //R_x*f
            dg::blas2::gemv( m_righty, m_temp, m_tempy); //R_y*f
        }

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
            if (m_alpha == 0) { 
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
            else {
                if(m_chi_weight_jump)
                {
                    dg::blas2::symv( m_jfactor, m_jumpX, m_gamma_x, 0., m_tempx);
                    dg::blas2::symv( m_jfactor, m_jumpY, m_gamma_x, 0., m_tempy);
                    dg::tensor::multiply2d(m_chi, m_tempx, m_tempy, m_tempx, m_tempy);
                    dg::blas1::axpbypgz(1.0, m_tempx, 1.0, m_tempy, 1.0, m_temp);
                } 
                else
                {
                    dg::blas2::symv( m_jfactor, m_jumpX, m_gamma_x, 1., m_temp);
                    dg::blas2::symv( m_jfactor, m_jumpY, m_gamma_x, 1., m_temp);
                }
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
    std::vector< dg::Helmholtz<Geometry,  Matrix, Container> > m_multi_gamma;
    dg::MultigridCG2d<Geometry, Matrix, Container> m_multi_g;
    KrylovSqrtCauchyinvert<Geometry, Matrix, DiaMatrix, CooMatrix, Container, SubContainer> sqrtinvert;

    Matrix m_leftx, m_lefty, m_rightx, m_righty, m_jumpX, m_jumpY;
    Container m_weights, m_inv_weights, m_precond, m_weights_wo_vol;
    Container m_tempx, m_tempy, m_temp, m_gamma_x;
    norm m_no;
    SparseTensor<Container> m_chi, m_metric;
    Container m_sigma, m_vol;
    value_type m_jfactor, m_alpha;
    std::vector<value_type> m_eps_gamma;
    bool m_chi_weight_jump;
    std::string m_mode;
};
    
///@cond
template< class G, class M, class DM, class CM, class V, class SV>
struct TensorTraits< Polarization<G, M, DM, CM, V, SV> >
{
    using value_type      = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};


template <class Geometry, class Matrix, class Container>
class PolarizationN
{
    public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    PolarizationN(){}
    /**
     * @brief Construct from Grid
     *
     * @param g The Grid, boundary conditions are taken from here
     * @param no choose \c dg::normed if you want to directly use the object,
     *  \c dg::not_normed if you want to invert the elliptic equation
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @param chi_weight_jump If true, the Jump terms are multiplied with the Chi matrix, else it is ignored
     * @note chi is assumed 1 per default
     */
    PolarizationN( const Geometry& g, norm no = not_normed,
        direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false):
        PolarizationN( g, g.bcx(), g.bcy(), no, dir, jfactor, chi_weight_jump)
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
     * @param chi_weight_jump If true, the Jump terms are multiplied with the Chi matrix, else it is ignored
     * @note chi is assumed 1 per default
     */
    PolarizationN( const Geometry& g, bc bcx, bc bcy,
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
        m_temp = m_phi= m_tempx = m_tempy = m_inv_weights ;
        m_chi=g.metric();
        m_metric=g.metric();
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
        *this = PolarizationN( std::forward<Params>( ps)...);
    }

    template<class ContainerType0>
    void set_phi( const ContainerType0& phi)
    {
      m_phi = phi;
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
     * @brief Compute the total variation integrand
     *
     * Computes \f[ (\nabla\phi)^2 = \partial_i \phi g^{ij}\partial_j \phi \f]
     * in the plane of a 2x1 product space
     * @param phi function
     * @param varphi may equal phi, contains result on output
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void variation( const ContainerType0& phi, ContainerType1& varphi)
    {
        blas2::symv( m_rightx, phi, m_tempx);
        blas2::symv( m_righty, phi, m_tempy);
        tensor::multiply2d( m_metric, m_tempx, m_tempy, varphi, m_temp);
        blas1::pointwiseDot( 1., varphi, m_tempx, 1., m_temp, m_tempy, 0., varphi);
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
     * i.e.  \f[ y=alpha*M*f(x)+beta*y \f]
     * @param alpha a scalar
     * @param x the chi term
     * @param beta a scalar
     * @param y result
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void symv( value_type alpha, const ContainerType0& x, value_type beta, ContainerType1& y)
    {
        
//         dg::blas1::pointwiseDivide( x, m_sigma, m_tempx);
        //update preconditioner
//         dg::blas1::pointwiseDivide( m_precond, x, m_precond);
        
        m_chi = m_metric;
        dg::tensor::scal( m_chi, m_vol);
        dg::tensor::scal( m_chi, x);
//         dg::blas1::copy( x, m_sigma);

        //compute gradient
        dg::blas2::gemv( m_rightx, m_phi, m_tempx); //R_x*f
        dg::blas2::gemv( m_righty, m_phi, m_tempy); //R_y*f

        //multiply with tensor (note the alias)
        dg::tensor::multiply2d(m_chi, m_tempx, m_tempy, m_tempx, m_tempy);

        //now take divergence
        dg::blas2::symv( m_lefty, m_tempy, m_temp);
        dg::blas2::symv( -1., m_leftx, m_tempx, -1., m_temp);

        //add jump terms
        if(m_chi_weight_jump)
        {
            dg::blas2::symv( m_jfactor, m_jumpX, m_phi, 0., m_tempx);
            dg::blas2::symv( m_jfactor, m_jumpY, m_phi, 0., m_tempy);
            dg::tensor::multiply2d(m_chi, m_tempx, m_tempy, m_tempx, m_tempy);
            dg::blas1::axpbypgz(1.0,m_tempx,1.0,m_tempy,1.0,m_temp);
        } 
        else
        {
            dg::blas2::symv( m_jfactor, m_jumpX, m_phi, 1., m_temp);
            dg::blas2::symv( m_jfactor, m_jumpY, m_phi, 1., m_temp);
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
    Container m_weights, m_inv_weights, m_phi, m_precond, m_weights_wo_vol;
    Container m_tempx, m_tempy, m_temp;
    norm m_no;
    SparseTensor<Container> m_chi, m_metric;
    Container m_sigma, m_vol;
    value_type m_jfactor;
    bool m_chi_weight_jump;
};
    
template< class G, class M, class V>
struct TensorTraits< PolarizationN<G, M, V> >
{
    using value_type      = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
  
}  //namespace dg
