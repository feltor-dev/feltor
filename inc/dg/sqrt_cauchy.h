#pragma once
#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
#include <boost/math/special_functions/jacobi_elliptic.hpp>

#include "blas.h"

#include "helmholtz.h"
// #include <cmath>
//! M_PI is non-standard ... so MSVC complains
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
template< class Geometry, class Matrix, class Container>
struct SqrtCauchyIntOp
{
    public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    SqrtCauchyIntOp() {}
    /**
     * @brief Construct operator \f[ (-w^2 I -A) \f] in cauchy formula
     *
     * @param A Helmholtz operator
     * @param g The grid to use
     */
    SqrtCauchyIntOp( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g):   
        m_helper( dg::evaluate( dg::zero, g)),
        m_A(A)
    { 
        dg::assign( dg::create::inv_volume(g),    m_inv_weights);
        dg::assign( dg::create::volume(g),        m_weights);
        dg::assign( dg::create::inv_weights(g),   m_precond);
        m_w=0.0;
    }
    /**
     * @brief Return the weights of the Helmholtz operator
     * 
     * @return the  weights of the Helmholtz operator
     */
    const Container& weights()const { return m_weights; }
    /**
     * @brief Return the inverse weights of the Helmholtz operator
     *
     * @return the inverse weights of the Helmholtz operator
     */
    const Container& inv_weights()const { return m_inv_weights; }
    /**
     * @brief Return the default preconditioner to use in conjugate gradient
     * 
     * @return the preconditioner of the Helmholtz operator
     */
    const Container& precond()const { return m_precond; }
    /**
     * @brief Set the time t
     *
     * @param time the time
     */
    void set_w( const value_type& w) { m_w=w; }   
    /**
     * @brief Compute operator
     *
     * i.e. \f[ y= W (w^2 I +V A)*x \f]
     * @param x left-hand-side
     * @param y result
     */ 
    void symv( const Container& x, Container& y) 
    {

        dg::blas2::symv(m_A, x, m_helper); //m_helper =  A x
        dg::blas1::pointwiseDot(m_inv_weights, m_helper, y); //m_helper = V A x ...normed 
        dg::blas1::axpby(m_w*m_w, x, 1., y); //m_helper = w^2 x + V A x 
        dg::blas1::pointwiseDot(m_weights, y, y); //make  not normed = W w^2 x + A x 
    } 
  private:
    Container m_helper;
    Container m_weights, m_inv_weights, m_precond, m_weights_wo_vol;
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    value_type m_w;
};


template< class Geometry, class Matrix, class Container>
struct dg::TensorTraits< SqrtCauchyIntOp<Geometry,  Matrix, Container> >
{
    using value_type  = dg::get_value_type<Container>;
    using tensor_category = dg::SelfMadeMatrixTag;
};

template<class Geometry, class Matrix, class Container>
struct CauchySqrtInt
{
  public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    /**
     * @brief Construct Rhs operator
     *
     * @param A Helmholtz operator
     * @param g The grid to use
     * @param eps Accuarcy for CG solve
     */
    CauchySqrtInt( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, value_type eps):
         m_helper( dg::evaluate( dg::zero, g)),
         m_helper2( dg::evaluate( dg::zero, g)),
         m_A(A),
         m_op(m_A, g),
         m_invert( m_helper, g.size(), eps, 1)
    {
    }
    /**
     * @brief Compute rhs term (including inversion of lhs) 
     *
     * i.e. \f[ b=  \frac{- 2 K' \sqrt{m}}{\pi iter} V A \sum_{j=1}{iter} (w^2 I -V A)^{-1} cn dn  x \f]
     * @param y  is \f[ y\f]
     * @param b is \f[ b\approx \sqrt{A} x\f]
     * @note The Jacobi elliptic functions are related to the Mathematica functions via jacobi_cn(k,u ) = JacobiCN_(u,k^2), ... and the complete elliptic integral of the first kind via comp_ellint_1(k) = EllipticK(k^2) 
     */
    void operator()(const Container& x, Container& b, const value_type& minEV, const value_type& maxEV, const unsigned& iter)
    {
        dg::blas1::scal(b,0.0);
        value_type sn=0.;
        value_type cn=0.;
        value_type dn=0.;
        value_type w = 0.;
        value_type t=0.;
        const value_type k2 = minEV/maxEV;
        const value_type sqrt1mk2 = sqrt(1.-k2);
        const value_type Ks=std::comp_ellint_1(sqrt1mk2 );
        const value_type fac = -2.* Ks*sqrt(minEV)/(M_PI*iter);
        for (unsigned j=1; j<iter+1; j++)
        {
            t  = (j-0.5)*Ks/iter; //imaginary part .. 1i missing
            cn = 1./boost::math::jacobi_cn(sqrt1mk2, t); 
            sn = boost::math::jacobi_sn(sqrt1mk2, t)*cn;
            dn = boost::math::jacobi_dn(sqrt1mk2, t)*cn;
            w = sqrt(minEV)*sn;
            dg::blas1::axpby(cn*dn, x, 0.0 , m_helper); //m_helper = cn dn x
            m_op.set_w(w);
            m_invert( m_op, m_helper2, m_helper);      // m_helper2 = (w^2 +V A)^(-1) cn dn x
            dg::blas1::axpby(-fac, m_helper2, 1.0, b); // b += -fac A (w^2 +V A)^(-1) cn dn x
        }
        dg::blas2::symv(m_A, b, m_helper); // A (w^2 +V A)^(-1) cn dn x
        dg::blas1::pointwiseDot(m_op.inv_weights(),  m_helper, b);  // fac V A (-w^2 I -V A)^(-1) cn dn x

    }
  private:
    Container m_helper, m_helper2;
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    SqrtCauchyIntOp<Geometry, Matrix, Container> m_op;
    dg::Invert<Container> m_invert;
};




template< class Matrix, class Container>
struct SqrtCauchyIntOpT
{
    public:
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    SqrtCauchyIntOpT() {}
    /**
     * @brief Construct operator \f[ (-w^2 I -T) \f] in cauchy formula
     *
     * @param T Symmetric tridiagonal matrix
     * @param copyable to get the size of the vectors
     */
    SqrtCauchyIntOpT( const Matrix& T, const Container& copyable)
    { 
        construct(T, copyable);
    }
    void construct( const Matrix& T, const Container& copyable)
    {
        m_A = T;
        m_one = copyable;
        dg::blas1::scal(m_one,0.);
        dg::blas1::plus(m_one,1.0);
        m_w=0.0;
        m_size = copyable.size();
    }
    /**
     * @brief Resize matrix T and vectors weights
     *
     * @param T Matrix
     */    
    void new_size( unsigned new_max) { 
        m_one.resize(new_max);
        m_size = new_max;
        dg::blas1::scal(m_one,0.);
        dg::blas1::plus(m_one,1.0);
    }
    /**
     * @brief Return the weights of the Helmholtz operator
     * 
     * @return the  weights r
     */
    const Container& weights()const { return m_one; }
    /**
     * @brief Return the inverse weights of the Helmholtz operator
     *
     * @return the inverse weights 
     */
    const Container& inv_weights()const { return m_one; }
    /**
     * @brief Return the default preconditioner to use in conjugate gradient
     * 
     * @return the preconditioner 
     */
    const Container& precond()const { return m_one; }
    /**
     * @brief Set the Matrix T
     *
     * @param T Matrix
     */
    void set_T( const Matrix& T) { m_A=T; } 
    /**
     * @brief Set w
     *
     * @param w of the discrete cauchy formula
     */
    void set_w( const value_type& w) { m_w=w; }   
    /**
     * @brief Compute operator
     *
     * i.e. \f[ y= W (w^2 I +V A)*x \f]
     * @param x left-hand-side
     * @param y result
     */ 
    void symv( const Container& x, Container& y) 
    {

        dg::blas2::symv(m_A, x,y); //y =  A x
        dg::blas1::axpby(m_w*m_w, x, 1.,y); //y = w^2 x +  A x 
    } 
  private:
    Matrix m_A;
    Container m_one;
    value_type m_w;
    unsigned m_size;

};

template<class Matrix, class Container>
struct CauchySqrtIntT
{
  public:
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    CauchySqrtIntT() {}
    /**
     * @brief Construct Rhs operator
     *
     * @param T Helmholtz operator
     * @param copyable copyable container
     * @param eps Accuarcy for CG solve
     */
    CauchySqrtIntT( const Matrix& T, const Container& copyable, value_type eps)
    {
        construct(T, copyable, eps);
    }
    void construct(const Matrix& T,  const Container& copyable,  value_type eps) 
    {
         m_helper = m_helper2 = copyable;
         m_A = T;
         m_size = copyable.size();
         m_op.construct(T, copyable);
         m_invert.construct( m_helper, m_size*m_size, eps, 1, false, 1.); //weights not multiplied on rhs
    }
    /**
     * @brief Resize matrix T and vectors weights
     *
     * @param T Matrix
     */    
    void new_size( unsigned new_max) { 
        m_helper.resize(new_max);
        m_helper2.resize(new_max);
        m_invert.set_size(m_helper, new_max*new_max);
        m_op.new_size(new_max);
        m_size = new_max;
    }
    /**
     * @brief Set the Matrix T
     *
     * @param T Matrix
     */
     void set_T( const Matrix& T) { 
         m_A=T; 
         m_op.set_T(T);
    }  
    /**
     * @brief Compute rhs term (including inversion of lhs) 
     *
     * i.e. \f[ b=  \frac{- 2 K' \sqrt{m}}{\pi iter} T \sum_{j=1}{iter} (w^2 I -T)^{-1} cn dn  x \f]
     * @param y  is \f[ y\f]
     * @param b is \f[ b\approx \sqrt{A} x\f]
     * @note The Jacobi elliptic functions are related to the Mathematica functions via jacobi_cn(k,u ) = JacobiCN_(u,k^2), ... and the complete elliptic integral of the first kind via comp_ellint_1(k) = EllipticK(k^2) 
     */
    void operator()(const Container& x, Container& b, const value_type& minEV, const value_type& maxEV, const unsigned& iter)
    {
        dg::blas1::scal(b,0.0);
        value_type sn=0.;
        value_type cn=0.;
        value_type dn=0.;
        value_type w = 0.;
        value_type t=0.;
        const value_type k2 = minEV/maxEV;
        const value_type sqrt1mk2 = sqrt(1.-k2);
        const value_type Ks=std::comp_ellint_1(sqrt1mk2 );
        const value_type fac = -2.* Ks*sqrt(minEV)/(M_PI*iter);
        for (unsigned j=1; j<iter+1; j++)
        {
            t  = (j-0.5)*Ks/iter; //imaginary part .. 1i missing
            cn = 1./boost::math::jacobi_cn(sqrt1mk2, t); 
            sn = boost::math::jacobi_sn(sqrt1mk2, t)*cn;
            dn = boost::math::jacobi_dn(sqrt1mk2, t)*cn;
            w = sqrt(minEV)*sn;
            dg::blas1::axpby(cn*dn, x, 0.0 , m_helper); //m_helper = cn dn x
            m_op.set_w(w);
            m_invert( m_op, m_helper2, m_helper);      // m_helper2 = (w^2 +T)^(-1) cn dn x
            dg::blas1::axpby(-fac, m_helper2, 1.0, b); // b += -fac A (w^2 +T)^(-1) cn dn x
        }
        dg::blas2::symv(m_A, b, m_helper); // A (w^2 +T)^(-1) cn dn x
        dg::blas1::pointwiseDot(m_op.inv_weights(),  m_helper, b);  // fac A (-w^2 I -V A)^(-1) cn dn x

    }
  private:
    Container m_helper, m_helper2;
    Matrix m_A;
    SqrtCauchyIntOpT<Matrix, Container> m_op;
    unsigned m_size;
    dg::Invert<Container> m_invert;
};

template< class Matrix, class Container>
struct dg::TensorTraits< SqrtCauchyIntOpT<  Matrix, Container> >
{
    using value_type  = dg::get_value_type<Container>;
    using tensor_category = dg::SelfMadeMatrixTag;
};
