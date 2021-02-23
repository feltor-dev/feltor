#pragma once

#include "elliptic.h"
#include "helmholtz.h"

namespace dg
{
/**
 * @brief Various arbitary wavelength polarization charge operators of delta-f (df) and full-f (ff)
 *
 * @ingroup matrixoperators
 * 
 */
//polarization solver class for N
template <class Geometry, class Matrix, class Container>
class PolChargeN
{
    public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    PolChargeN(){}
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
    PolChargeN( const Geometry& g, norm no = not_normed,
        direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false):
        PolChargeN( g, g.bcx(), g.bcy(), no, dir, jfactor, chi_weight_jump)
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
    PolChargeN( const Geometry& g, bc bcx, bc bcy,
        norm no = not_normed, direction dir = forward,
        value_type jfactor=1., bool chi_weight_jump = false)
    {
        m_ell.construct(g, bcx, bcy, no, dir, jfactor, chi_weight_jump );
        m_gamma.construct(g, bcx, bcy, -0.5, dir, jfactor);
        dg::assign(dg::evaluate(dg::zero,g), m_phi);
        dg::assign(dg::evaluate(dg::one,g), m_temp);

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
        *this = PolChargeN( std::forward<Params>( ps)...);
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
        return m_ell.inv_weights();
    }
    /**
     * @brief Return the vector making the matrix symmetric
     *
     * i.e. the volume form
     * @return volume form including weights
     */
    const Container& weights()const {
        return m_ell.weights();
    }
    /**
     * @brief Return the default preconditioner to use in conjugate gradient
     *
     * Currently returns the inverse weights without volume elment divided by the scalar part of \f$ \chi\f$.
     * This is especially good when \f$ \chi\f$ exhibits large amplitudes or variations
     * @return the inverse of \f$\chi\f$.
     */
    const Container& precond()const { 
        return m_ell.precond();
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
//         dg::blas1::transform( x, m_temp, dg::ABS<double>());
//         m_ell.set_chi(x);
//         m_ell.symv(alpha, m_phi, beta, y);
   
        m_ell.set_chi(x); 
        m_ell.symv(1., m_phi, 0.0, y);
        dg::blas2::symv(m_gamma, y, m_temp);
        dg::blas2::symv(m_ell.inv_weights(), m_temp, y);
        
        dg::blas1::axpby(1.0, x, -1.0,y);
        dg::blas1::plus(y, -1.0);

    }

    private:
    dg::Elliptic<Geometry,  Matrix, Container> m_ell;
    dg::Helmholtz<Geometry,  Matrix, Container> m_gamma;
    Container m_phi, m_temp;
};
    
template< class G, class M, class V>
struct TensorTraits< PolChargeN<G, M, V> >
{
    using value_type      = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
  
}  //namespace dg
