#pragma once
#include "dg/algorithm.h"

#include "matrixsqrt.h"
#include "matrixfunction.h"
#include "tensorelliptic.h"

namespace dg
{
/**
 * @brief Various arbitary wavelength polarization charge operators of delta-f (df) and full-f (ff)
 *
 * @ingroup matrixoperators
 * 
 */
template <class Geometry, class Matrix, class Container>
class PolCharge
{
    public:
    using value_type = get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    PolCharge(){}
    /**
     * @brief Construct from Grid
     *
     * @param alpha alpha of the Helmholtz operator
     * @param eps_gamma epsilon (-vector) for the Helmholtz operator inversion or the sqrt Helmholtz operator inversion
     * @param g The Grid, boundary conditions are taken from here
     * @param no choose \c dg::normed if you want to directly use the object,
     *  \c dg::not_normed if you want to invert the elliptic equation
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @param chi_weight_jump If true, the Jump terms are multiplied with the Chi matrix, else it is ignored
     * @param mode arbitrary wavelength polarization charge mode ("df" / "ff" / "ffO4" are implemented)
     * @param commute false if Helmholtz operators (or their square root) are outside the elliptic or tensorelliptic operator and true otherwise
     */
    PolCharge(value_type alpha, std::vector<value_type> eps_gamma, const Geometry& g, norm no = not_normed, direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false, std::string mode = "df", bool commute = false)
    {
        construct(alpha, eps_gamma, g,  g.bcx(), g.bcy(), no, dir, jfactor, chi_weight_jump, mode);
    }
    /**
     * @brief Construct from boundary conditions
     *
     * @param alpha alpha of the Helmholtz operator
     * @param eps_gamma epsilon (-vector) for the Helmholtz operator inversion or the sqrt Helmholtz operator inversion
     * @param g The Grid, boundary conditions are taken from here
    * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param no choose \c dg::normed if you want to directly use the object,
     *  \c dg::not_normed if you want to invert the elliptic equation
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @param chi_weight_jump If true, the Jump terms are multiplied with the Chi matrix, else it is ignored
     * @param mode arbitrary wavelength polarization charge mode ("df" / "ff" / "ffO4" are implemented)
     * @param commute false if Helmholtz operators (or their square root) are outside the elliptic or tensorelliptic operator and true otherwise
    */
    PolCharge( value_type alpha, std::vector<value_type> eps_gamma, const Geometry& g, bc bcx, bc bcy, norm no = not_normed, direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false, std::string mode = "df", bool commute = false)
    { 
         construct(alpha, eps_gamma, g,  bcx, bcy, no, dir, jfactor, chi_weight_jump, mode);
    }
    /**
     * @brief Construct from boundary conditions
     *
     * @param alpha alpha of the Helmholtz operator
     * @param eps_gamma epsilon (-vector) for the Helmholtz operator inversion or the sqrt Helmholtz operator inversion
     * @param g The Grid, boundary conditions are taken from here
    * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param no choose \c dg::normed if you want to directly use the object,
     *  \c dg::not_normed if you want to invert the elliptic equation
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @param chi_weight_jump If true, the Jump terms are multiplied with the Chi matrix, else it is ignored
     * @param mode arbitrary wavelength polarization charge mode ("df" / "ff" / "ffO4" are implemented)
     * @param commute false if Helmholtz operators (or their square root) are outside the elliptic or tensorelliptic operator and true otherwise
    */
    void construct(value_type alpha, std::vector<value_type> eps_gamma, const Geometry& g, bc bcx, bc bcy, norm no = not_normed, direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false, std::string mode = "df", bool commute = false)
    {
        m_alpha = alpha;
        m_eps_gamma = eps_gamma;
        m_mode = mode;
        m_commute = commute;
        m_no = no,
        m_temp2 = dg::evaluate(dg::zero, g);
        m_temp =  m_temp2;        
        m_temp2_ex.set_max(1, m_temp2);
        m_temp_ex.set_max(1, m_temp);
        if (m_mode == "df")
        {
            m_ell.construct(g, bcx, bcy, m_no, dir, jfactor, chi_weight_jump );
            m_multi_gamma.resize(3);
            m_multi_g.construct(g, 3);
            for( unsigned u=0; u<3; u++)
            {
                m_multi_gamma[u].construct( m_multi_g.grid(u), bcx, bcy, m_alpha, dir, jfactor);
            }
        }
        if (m_mode == "ff")
        {
            m_ell.construct(g, bcx, bcy, m_no, dir, jfactor, chi_weight_jump );
            m_multi_gamma.resize(1);
            m_multi_gamma.resize(1);
            m_multi_gamma[0].construct( g, bcx, bcy, m_alpha, dir, jfactor);
            m_sqrtG0inv.construct(m_multi_gamma[0], g,  m_temp,  1e-14, 2000, 40, eps_gamma[0]);
                        
//             m_sqrtG0inv.construct(m_temp, g.size());
            
            value_type hxhy = g.lx()*g.ly()/(g.n()*g.n()*g.Nx()*g.Ny());
            value_type max_weights =   dg::blas1::reduce(m_multi_gamma[0].weights(), 0., dg::AbsMax<double>() );
            value_type min_weights =  -dg::blas1::reduce(m_multi_gamma[0].weights(), max_weights, dg::AbsMin<double>() );
            value_type kappa = sqrt(max_weights/min_weights); //condition number of weight matrix
            value_type EVmin = 1./(1.-m_multi_gamma[0].alpha()*hxhy*(1.0 + 1.0)); //EVs of Helmholtz
            m_res_fac = kappa*sqrt(EVmin);
        }
        if (m_mode == "ffO4")
        {
            m_tensorell.construct(g, bcx, bcy, m_no, dir, jfactor); //not normed by default
            m_multi_gamma.resize(3);
//             m_multi_gamma.resize(3);
            m_multi_g.construct(g, 3);
            for( unsigned u=0; u<3; u++)
            {
                m_multi_gamma[u].construct( m_multi_g.grid(u), bcx, bcy, m_alpha, dir, jfactor);
//                 m_multi_gamma[u].construct( m_multi_g.grid(u), inverse( bcx), inverse( bcy), m_alpha, dir, jfactor);
            }
        }
        

        
    }
    /**
     * @brief Change \f$\chi\f$ in the elliptic or tensor elliptic operator
     *
     * @param sigma The new scalar part in \f$\chi\f$ (all elements must be >0)
     * @tparam ContainerType0 must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0>
    void set_chi( const ContainerType0& sigma)
    {
        if (m_mode == "ff")
            m_ell.set_chi(sigma);
        if (m_mode == "ffO4")
        {
            m_tensorell.set_chi(sigma);
        }
    }
    /**
     * @brief Change \f$\chi\f$ in the elliptic or tensor elliptic operator
     *
     * @param tau The new tensor part in \f$\chi\f$ (all elements must be >0)
     * @tparam ContainerType0 must be usable in \c dg::assign to \c Container
     */
    template<class ContainerType0>
    void set_chi( const SparseTensor<ContainerType0>& tau)
    {
        if (m_mode == "ff")
            m_ell.set_chi(tau);
        if (m_mode == "ffO4")
        {
            m_tensorell.set_chi(tau);
        }
    }
    /**
     * @brief Change \f$\iota\f$ in the tensor elliptic operator
     *
     * @param sigma The new scalar part in \f$\chi\f$ (all elements must be >0)
     * @tparam ContainerType0 must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0>
    void set_iota( const ContainerType0& sigma)
    {
        if (m_mode == "ffO4")
        {
            m_tensorell.set_iota(sigma);
        }
    }
    /**
     * @brief Set the commute
     * @param commute Either true or false.
     */
    void set_commute( bool commute) {m_commute = commute;}
    /**
     * @brief Get the current state of commute
     * @return Either true or false.
     */
    bool get_commute() const {return m_commute;}
    /**
     * @brief Return the vector missing in the un-normed symmetric matrix
     *
     * i.e. the inverse of the weights() function
     * @return inverse volume form including inverse weights
     */
    const Container& inv_weights()const {
        if (m_mode == "ffO4")
            return m_tensorell.inv_weights();
        else return m_ell.inv_weights();
    }
    /**
     * @brief Return the vector making the matrix symmetric
     *
     * i.e. the volume form
     * @return volume form including weights
     */
    const Container& weights()const {
        if (m_mode == "ffO4")
             return  m_tensorell.weights();
        else return  m_ell.weights();
    }
    /**
     * @brief Return the default preconditioner to use in conjugate gradient
     *
     * Currently returns the inverse weights without volume elment divided by the scalar part of \f$ \chi\f$.
     * This is especially good when \f$ \chi\f$ exhibits large amplitudes or variations
     * @return the inverse of \f$\chi\f$.
     */
    const Container& precond()const {
        if (m_mode == "ffO4")
            return m_tensorell.precond();
        else return m_ell.precond();
    }
    
    template<class ContainerType0, class ContainerType1>
    void variation( const ContainerType0& phi, ContainerType1& varphi)
    {
        if (m_mode == "ff")
            m_ell.variation(phi, varphi);
        if (m_mode == "ffO4")
            m_tensorell.variation(phi, varphi);
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

        if (m_alpha == 0)
        {
            if (m_mode == "ffO4")
            {
                m_tensorell.symv(alpha, x, beta, y); //is not normed by default
            }
            if (m_mode == "ff" || m_mode == "df")                
                m_ell.symv(alpha, x, beta, y);            
        }
        else
        {
            if (m_mode == "df")
            {
                if (m_commute == false)
                {
                    m_temp2_ex.extrapolate(m_temp2);
                    std::vector<unsigned> number = m_multi_g.direct_solve( m_multi_gamma, m_temp2, x, m_eps_gamma);
                    if(  number[0] == m_multi_g.max_iter())
                        throw dg::Fail( m_eps_gamma[0]);
                    m_temp2_ex.update(m_temp2);
                    
                    m_ell.symv(alpha, m_temp2, beta, y);
                }
                else
                {
                    m_ell.symv(1.0, x, 0.0, m_temp);
                    if (m_no == not_normed) 
                        dg::blas1::pointwiseDot(m_temp, m_ell.inv_weights(), m_temp);    //temp should be normed
                    
                    m_temp2_ex.extrapolate(m_temp2);
                    std::vector<unsigned> number = m_multi_g.direct_solve( m_multi_gamma, m_temp2, m_temp, m_eps_gamma);
                    if(  number[0] == m_multi_g.max_iter())
                        throw dg::Fail( m_eps_gamma[0]);
                    m_temp2_ex.update(m_temp2);
                   
                    
                    if( m_no == normed)
                        dg::blas1::axpby(alpha, m_temp2, beta, y);  //m_temp2 is normed

                    if( m_no == not_normed)
                        dg::blas1::pointwiseDot( alpha, m_temp2, m_ell.weights(), beta, y);         
                }    
                
            }
            if (m_mode == "ff" ) //assuming constant FLR effects
            {     
                if (m_commute == false)
                {
                    dg::blas1::scal(m_temp2, 0.0);

                    std::array<unsigned,2> number = m_sqrtG0inv( m_temp2, x);  //m_temp2 is normed
                    std::cout << "#number of sqrt iterations: " << number[0] << std::endl;
//                     m_sqrtG0inv( m_temp2, x, dg::SQRT<double>(), m_multi_gamma[0], m_multi_gamma[0].inv_weights(), m_multi_gamma[0].weights(),  m_eps_gamma[0], m_res_fac);  //m_temp2 is normed
        
                    m_ell.symv(1.0, m_temp2, 0.0, m_temp); //m_temp is not normed or not normed
                    
                    //make normed before operator is applied
                    if (m_no == not_normed) 
                        dg::blas1::pointwiseDot(m_temp, m_ell.inv_weights(), m_temp);
                    
                    dg::blas1::scal(m_temp2, 0.0);
                    number =m_sqrtG0inv( m_temp2, m_temp);  //m_temp2 is normed
                    std::cout << "#number of sqrt iterations: " << number[0] << std::endl;
//                     m_sqrtG0inv( m_temp2, m_temp, dg::SQRT<double>(), m_multi_gamma[0], m_multi_gamma[0].inv_weights(), m_multi_gamma[0].weights(),  m_eps_gamma[0], m_res_fac);  //m_temp2 is normed
                    
                    if( m_no == normed)
                        dg::blas1::axpby(alpha, m_temp2, beta, y);  
                    if( m_no == not_normed)
                        dg::blas1::pointwiseDot( alpha, m_temp2, m_ell.weights(), beta, y);   
                    
                }
                else
                {
                    //TODO not implemented so far (relevant thermal models)
                }
            }
            if (m_mode == "ffO4")
            {  
                if (m_commute == false)
                {      
                    m_temp2_ex.extrapolate(m_temp2);
                    std::vector<unsigned> number = m_multi_g.direct_solve( m_multi_gamma, m_temp2, x, m_eps_gamma);
                    if(  number[0] == m_multi_g.max_iter())
                        throw dg::Fail( m_eps_gamma[0]);
                    m_temp2_ex.update(m_temp2);

                    m_tensorell.symv(1.0, m_temp2, 0.0, m_temp); 
                    if (m_no == not_normed) 
                        dg::blas1::pointwiseDot(m_temp, m_tensorell.inv_weights(), m_temp);
                    
                    m_temp_ex.extrapolate(m_temp2);
                    number = m_multi_g.direct_solve( m_multi_gamma, m_temp2, m_temp, m_eps_gamma);
                    if(  number[0] == m_multi_g.max_iter())
                        throw dg::Fail( m_eps_gamma[0]);
                    m_temp_ex.update(m_temp2);
                    
                    if( m_no == normed)
                        dg::blas1::axpby(alpha, m_temp2, beta, y);  
                    if( m_no == not_normed)
                        dg::blas1::pointwiseDot( alpha, m_temp2, m_tensorell.weights(), beta, y); 
                }
                if (m_commute == true)
                {
                    //TODO not implemented so far (relevant thermal models)
                }
            }
        }
    }

    private:
    dg::Elliptic<Geometry,  Matrix, Container> m_ell;
    dg::TensorElliptic<Geometry,  Matrix, Container> m_tensorell;
    
    std::vector< dg::Helmholtz<Geometry,  Matrix, Container> > m_multi_gamma;
    dg::MultigridCG2d<Geometry, Matrix, Container> m_multi_g;
    dg::KrylovSqrtCauchyinvert<Geometry, Matrix, Container> m_sqrtG0inv;
//     dg::KrylovFuncEigenInvert< Container> m_sqrtG0inv;        
    Container m_temp, m_temp2;
    norm m_no;
    value_type  m_alpha,  m_res_fac;
    std::vector<value_type> m_eps_gamma;
    std::string m_mode;
    dg::Extrapolation<Container> m_temp2_ex, m_temp_ex;
    bool m_commute;
    

};
    
///@cond
template< class G, class M, class V>
struct TensorTraits< PolCharge<G, M, V> >
{
    using value_type      = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
///@endcond

  
}  //namespace dg
