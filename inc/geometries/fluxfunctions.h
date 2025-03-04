#pragma once
#include <functional>
#include "dg/algorithm.h"

namespace dg
{
namespace geo
{

///@addtogroup fluxfunctions
///@{
/*! @brief Inject both 2d and 3d \c operator() to a 2d functor
 *
 * The purpose of this class is to extend any 2d Functor to a
 * 3d Functor by defining \f$ f(x,y,z) := f(x,y)\f$. This class is
 * especially useful in an interface since any 2d functor can be converted
 * to it (type erasure property of the \c std::function that we use
 * to implement this class).
 * @note If you want to write a functor that is both 2d and 3d directly,
 * it is easier to derive from \c aCylindricalFunctor
 * @sa this class is an alternative to \c aCylindricalFunctor and
 * \c aCylindricalFunctor can be converted to this class
 */
template<class real_type>
struct RealCylindricalFunctor
{
    RealCylindricalFunctor(){}
    /**
    * @brief Construct from any binary functor
    *
    * @tparam BinaryFunctor Interface must be <tt> real_type(real_type x, real_type y)</tt>
    * @param f a 2d functor
    */
    template<class BinaryFunctor>
    RealCylindricalFunctor( BinaryFunctor f):
        m_f(f) {}
    /// @return f(R,Z)
    real_type operator()( real_type R, real_type Z) const{
        return m_f(R,Z);
    }
    /// @return f(R,Z)
    real_type operator()( real_type R, real_type Z, real_type phi) const{
        return m_f(R,Z);
    }
    private:
    std::function<real_type(real_type,real_type)> m_f;
};

///Most of the times we use \c double
using CylindricalFunctor = RealCylindricalFunctor<double>;

/**
* @brief Represent functions written in cylindrical coordinates
        that are independent of the angle phi serving as both 2d and 3d functions

* The rational is that these functors can serve as both 2d and 3d functors
* where the 3d functor trivially redirects to the 2d version.
* This behaviour is injected into all classes that derive from this class
* via the Curiously Recurring Template Pattern (CRTP).
* @sa \c aCylindricalFunctor
* @sa An alternative is \c RealCylindricalFunctor
* @tparam Derived Interface: <tt> double do_compute(double,double) const</tt>
*/
template<class Derived>
struct aCylindricalFunctor
{
    /**
    * @brief <tt> do_compute(R,Z)</tt>
    *
    * @param R radius (cylindrical coordinate)
    * @param Z height (cylindrical coordinate)
    *
    * @return f(R,Z)
    */
    double operator()(double R, double Z) const
    {
        const Derived& underlying = static_cast<const Derived&>(*this);
        return underlying.do_compute(R,Z);
    }
    /**
    * @brief <tt> do_compute(R,Z)</tt>
    *
    * @param R radius (cylindrical coordinate)
    * @param Z height (cylindrical coordinate)
    * @param phi angle (cylindrical coordinate)
    *
    * @return f(R,Z)
    */
    double operator()(double R, double Z, double phi)const
    {
        const Derived& underlying = static_cast<const Derived&>(*this);
        return underlying.do_compute(R,Z);
    }
#ifndef __CUDACC__ //nvcc below 10 has problems with the following construct
    //This trick avoids that classes inherit from the wrong Base:
    private:
    friend Derived;
    aCylindricalFunctor(){}
    /**
    * @brief We do not allow object slicing so the copy is protected
    */
    aCylindricalFunctor(const aCylindricalFunctor&){}
    /**
    * @brief We do not allow object slicing so the assignment is protected
    */
    aCylindricalFunctor& operator=(const aCylindricalFunctor&){return *this;}
#endif //__CUDACC__
};

/**
 * @brief \f$ f(x,y) = c\f$
 */
struct Constant: public aCylindricalFunctor<Constant>
{
    Constant(double c):c_(c){}
    double do_compute(double R,double Z)const{return c_;}
    private:
    double c_;
};
/**
 * @brief
 * \f$ f(R,Z)= \begin{cases}
 0 \text{ if } Z < Z_X \\
 1 \text{ else }
 \end{cases}
 \f$
 @note the 1 is inclusive i.e. if Z == Z_X the functor always returns 1
 */
struct ZCutter : public aCylindricalFunctor<ZCutter>
{
    ZCutter(double ZX, int sign = +1): m_heavi( ZX, sign){}
    double do_compute(double R, double Z) const {
        return m_heavi(Z);
    }
    private:
    dg::Heaviside m_heavi;
};

/**
 * @brief This function extends another function beyond the grid boundaries
 * @sa dg::geo::periodify
 */
struct Periodify : public aCylindricalFunctor<Periodify>
{
    /**
     * @brief Construct from grid
     *
     * @param functor the functor to periodify
     * @param g The grid provides the shift member
     */
    Periodify( CylindricalFunctor functor, dg::Grid2d g): m_g( g), m_f(functor) {}
    /**
     * @brief provide 2d grid boundaries by hand
     *
     * @param functor the functor to periodify
     * @param R0 left boundary in R
     * @param R1 right boundary in R
     * @param Z0 lower boundary in Z
     * @param Z1 upper boundary in Z
     * @param bcx boundary condition in x (determines how function is periodified)
     * @param bcy boundary condition in y (determines how function is periodified)
     */
    Periodify( CylindricalFunctor functor, double R0, double R1, double Z0,
            double Z1, dg::bc bcx, dg::bc bcy):
        m_g( R0, R1, Z0, Z1, 3, 10, 10, bcx, bcy), m_f(functor)
    {}
    double do_compute( double R, double Z) const
    {
        bool negative = false;
        dg::create::detail::shift( negative, R, m_g.bcx(), m_g.x0(), m_g.x1());
        dg::create::detail::shift( negative, Z, m_g.bcy(), m_g.y0(), m_g.y1());
        if( negative) return -m_f(R,Z);
        return m_f( R, Z);
    }
    private:
    dg::Grid2d m_g;
    CylindricalFunctor m_f;
};

/**
* @brief This struct bundles a function and its first derivatives
*
* @snippet flux_t.cpp hector
*/
struct CylindricalFunctorsLvl1
{
    ///the access functions are undefined as long as the class remains empty
    CylindricalFunctorsLvl1(){}
    /**
    * @brief Construct with given functors
    *
    * @param f \f$ f(x,y)\f$ the function in some coordinates (x,y)
    * @param fx \f$ \partial f / \partial x \f$ its derivative in the first coordinate
    * @param fy \f$ \partial f / \partial y \f$ its derivative in the second coordinate
    */
    CylindricalFunctorsLvl1(  CylindricalFunctor f,  CylindricalFunctor fx,
        CylindricalFunctor fy) : p_{{ f, fx, fy}} {
    }
    ///copy given functors
    void reset( CylindricalFunctor f, CylindricalFunctor fx, CylindricalFunctor fy)
    {
        p_[0] = f;
        p_[1] = fx;
        p_[2] = fy;
    }
    /// \f$ f \f$
    const CylindricalFunctor& f()const{return p_[0];}
    /// \f$ \partial f / \partial x \f$
    const CylindricalFunctor& dfx()const{return p_[1];}
    /// \f$ \partial f / \partial y\f$
    const CylindricalFunctor& dfy()const{return p_[2];}
    private:
    std::array<CylindricalFunctor,3> p_;
};


/**
* @brief This struct bundles a function and its first and second derivatives
*
* @snippet flux_t.cpp hector
*/
struct CylindricalFunctorsLvl2
{
    ///the access functions are undefined as long as the class remains empty
    CylindricalFunctorsLvl2(){}
    /**
    * @copydoc CylindricalFunctorsLvl1::CylindricalFunctorsLvl1(CylindricalFunctor,CylindricalFunctor,CylindricalFunctor)
    * @param fxx \f$ \partial^2 f / \partial x^2\f$ second derivative in first coordinate
    * @param fxy \f$ \partial^2 f / \partial x \partial y\f$ second mixed derivative
    * @param fyy \f$ \partial^2 f / \partial y^2\f$ second derivative in second coordinate
    */
    CylindricalFunctorsLvl2(  CylindricalFunctor f,  CylindricalFunctor fx,
        CylindricalFunctor fy,   CylindricalFunctor fxx,
        CylindricalFunctor fxy,  CylindricalFunctor fyy):
        f0(f,fx,fy), f1(fxx,fxy,fyy)
    { }
    ///Replace with given Functors
    void reset( CylindricalFunctor f, CylindricalFunctor fx,
        CylindricalFunctor fy, CylindricalFunctor fxx,
        CylindricalFunctor fxy, CylindricalFunctor fyy)
    {
        f0.reset( f,fx,fy), f1.reset(fxx,fxy,fyy);
    }
    ///type conversion: Lvl2 can also be used as Lvl1
    operator CylindricalFunctorsLvl1 ()const {return f0;}
    /// \f$ f \f$
    const CylindricalFunctor& f()const{return f0.f();}
    /// \f$ \partial f / \partial x \f$
    const CylindricalFunctor& dfx()const{return f0.dfx();}
    /// \f$ \partial f / \partial y\f$
    const CylindricalFunctor& dfy()const{return f0.dfy();}
    /// \f$ \partial^2f/\partial x^2\f$
    const CylindricalFunctor& dfxx()const{return f1.f();}
    /// \f$ \partial^2 f / \partial x \partial y\f$
    const CylindricalFunctor& dfxy()const{return f1.dfx();}
    /// \f$ \partial^2f/\partial y^2\f$
    const CylindricalFunctor& dfyy()const{return f1.dfy();}
    private:
    CylindricalFunctorsLvl1 f0,f1;
};


/**
 * @brief This function finds critical points of psi (any point with vanishing gradient, including the X-point or O-point) via Newton iteration applied to the gradient of psi
 *
 * Newton iteration applied to \f$ \nabla \psi (\vec x) = 0 \f$ reads
 * \f[ \vec x_{i+1} = \vec x_i - H^{-1} \nabla \psi (\vec x_i)\f]
 * where H is the Hessian matrix.
 * The inverse of the Hessian matrix is computed analytically
 * @param psi \f$ \psi(R,Z)\f$
 * @param RC start value on input, critical point on output
 * @param ZC start value on input, critical point on output
 * @return 0 if no critical point or Hessian (determinant) is zero,
 * 1 if local minimum,
 * 2 if local maximum,
 * 3 if saddle point
 * @ingroup misc_geo
 */
inline int findCriticalPoint( const CylindricalFunctorsLvl2& psi, double& RC, double& ZC)
{
    std::array<double, 2> X{ {0,0} }, XN(X), X_OLD(X);
    X[0] = RC, X[1] = ZC;
    double eps = 1e10, eps_old= 2e10;
    unsigned counter = 0; //safety measure to avoid deadlock
    double psipRZ = psi.dfxy()(X[0], X[1]);
    double psipRR = psi.dfxx()(X[0], X[1]), psipZZ = psi.dfyy()(X[0],X[1]);
    double psipR  = psi.dfx()(X[0], X[1]), psipZ = psi.dfy()(X[0], X[1]);
    double D0 =  (psipZZ*psipRR - psipRZ*psipRZ);
    if(D0 == 0) // try to change initial guess slightly if we are very lucky
    {
        X[0] *= 1.0001, X[1]*=1.0001;
        psipRZ = psi.dfxy()(X[0], X[1]);
        psipRR = psi.dfxx()(X[0], X[1]), psipZZ = psi.dfyy()(X[0],X[1]);
        psipR  = psi.dfx()(X[0], X[1]), psipZ = psi.dfy()(X[0], X[1]);
        D0 =  (psipZZ*psipRR - psipRZ*psipRZ);
    }
    double Dinv = 1./D0;
    while( (eps < eps_old || eps > 1e-7) && eps > 1e-10 && counter < 100)
    {
        //newton iteration
        XN[0] = X[0] - Dinv*(psipZZ*psipR - psipRZ*psipZ);
        XN[1] = X[1] - Dinv*(-psipRZ*psipR + psipRR*psipZ);
        XN.swap(X);
        eps = sqrt( (X[0]-X_OLD[0])*(X[0]-X_OLD[0]) + (X[1]-X_OLD[1])*(X[1]-X_OLD[1]));
        X_OLD = X; eps_old= eps;
        psipRZ = psi.dfxy()(X[0], X[1]);
        psipRR = psi.dfxx()(X[0], X[1]), psipZZ = psi.dfyy()(X[0],X[1]);
        psipR  = psi.dfx()(X[0], X[1]), psipZ = psi.dfy()(X[0], X[1]);
        D0 = (psipZZ*psipRR - psipRZ*psipRZ);
        Dinv = 1./D0;
        if( D0 == 0) break;
        counter++;
    }
    if ( counter >= 100 || D0 == 0|| std::isnan( Dinv) )
        return 0;
    RC = X[0], ZC = X[1];
    if( Dinv > 0 &&  psipRR > 0)
        return 1; //local minimum
    if( Dinv > 0 &&  psipRR < 0)
        return 2; //local maximum
    //if( Dinv < 0)
    return 3; //saddle point
}

/**
 * @brief This function finds O-points of psi
 *
 * Same as \c findCriticalPoint except that this function throws if it does
 * not find a local minimum or a local maximum
 * @param psi \f$ \psi(R,Z)\f$
 * @param RC start value on input, O-point on output
 * @param ZC start value on input, O-point on output
 * @return 1 if local minimum, 2 if local maximum,
 * @ingroup misc_geo
 */
inline int findOpoint( const CylindricalFunctorsLvl2& psi, double& RC, double& ZC)
{
    int point = findCriticalPoint( psi, RC, ZC);
    if( point == 3 || point == 0 )
        throw dg::Error(dg::Message(_ping_)<<"There is no O-point near "<<RC<<" "<<ZC);
    return point;
}

/**
 * @brief This function finds X-points of psi
 *
 * Same as \c findCriticalPoint except that this function throws if it does
 * not find a saddle point
 * @param psi \f$ \psi(R,Z)\f$
 * @param RC start value on input, X-point on output
 * @param ZC start value on input, X-point on output
 * @ingroup misc_geo
 */
inline void findXpoint( const CylindricalFunctorsLvl2& psi, double& RC, double& ZC)
{
    int point = findCriticalPoint( psi, RC, ZC);
    if( point != 3)
        throw dg::Error(dg::Message(_ping_)<<"There is no X-point near "<<RC<<" "<<ZC);
}


/// A symmetric 2d tensor field and its divergence
///@snippet flux_t.cpp hector
struct CylindricalSymmTensorLvl1
{
    /**
     * @brief Initialize with the identity tensor
     */
    CylindricalSymmTensorLvl1( ){
        reset( Constant(1), Constant(0), Constant(1), Constant(0), Constant(0));
    }
    /**
     * @brief Copy given functors
     *
     * let's assume the tensor is called \f$ \chi \f$ (chi)
     * @param chi_xx contravariant xx component \f$ \chi^{xx}\f$
     * @param chi_xy contravariant xy component \f$ \chi^{xy}\f$
     * @param chi_yy contravariant yy component \f$ \chi^{yy}\f$
     * @param divChiX \f$ \partial_x \chi^{xx} + \partial_y\chi^{yx}\f$ is the x-component of the divergence of the tensor \f$ \chi\f$
     * @param divChiY \f$ \partial_x \chi^{xy} + \partial_y\chi^{yy}\f$ is the y-component of the divergence of the tensor \f$ \chi \f$
    */
    CylindricalSymmTensorLvl1(  CylindricalFunctor chi_xx,
        CylindricalFunctor chi_xy,   CylindricalFunctor chi_yy,
        CylindricalFunctor divChiX,  CylindricalFunctor divChiY) :
        p_{{ chi_xx,chi_xy,chi_yy,divChiX,divChiY}}
    {
    }
    ///replace with given functors
    void reset( CylindricalFunctor chi_xx, CylindricalFunctor chi_xy,
        CylindricalFunctor chi_yy, CylindricalFunctor divChiX,
        CylindricalFunctor divChiY)
    {
        p_[0] = chi_xx;
        p_[1] = chi_xy;
        p_[2] = chi_yy;
        p_[3] = divChiX;
        p_[4] = divChiY;
    }
    ///xy component \f$ \chi^{xx}\f$
    const CylindricalFunctor& xx()const{return p_[0];}
    ///xy component \f$ \chi^{xy}\f$
    const CylindricalFunctor& xy()const{return p_[1];}
    ///yy component \f$ \chi^{yy}\f$
    const CylindricalFunctor& yy()const{return p_[2];}
     /// \f$ \partial_x \chi^{xx} + \partial_y\chi^{yx}\f$ is the x-component of the divergence of the tensor \f$ \chi\f$
    const CylindricalFunctor& divX()const{return p_[3];}
     /// \f$ \partial_x \chi^{xy} + \partial_y\chi^{yy}\f$ is the y-component of the divergence of the tensor \f$ \chi \f$
    const CylindricalFunctor& divY()const{return p_[4];}
    private:
    std::array<CylindricalFunctor,5> p_;
};

/// A vector field with three components that depend only on the first two coordinates
///@snippet ds_t.cpp doxygen
struct CylindricalVectorLvl0
{
    CylindricalVectorLvl0(){}
    ///Copy given Functors
    CylindricalVectorLvl0(  CylindricalFunctor v_x,
        CylindricalFunctor v_y,
        CylindricalFunctor v_z): p_{{v_x, v_y, v_z}}{}
    ///replace with given functors
    void reset(  CylindricalFunctor v_x,  CylindricalFunctor v_y,
        CylindricalFunctor v_z)
    {
        p_[0] = v_x;
        p_[1] = v_y;
        p_[2] = v_z;
    }
    /// x-component of the vector
    const CylindricalFunctor& x()const{return p_[0];}
    /// y-component of the vector
    const CylindricalFunctor& y()const{return p_[1];}
    /// z-component of the vector
    const CylindricalFunctor& z()const{return p_[2];}
    private:
    std::array<CylindricalFunctor,3> p_;
};

/**
* @brief This struct bundles a vector field and its divergence
*/
struct CylindricalVectorLvl1
{
    ///the access functions are undefined as long as the class remains empty
    CylindricalVectorLvl1(){}
    ///Copy given Functors
    CylindricalVectorLvl1(  CylindricalFunctor v_x,
        CylindricalFunctor v_y,
        CylindricalFunctor v_z,
        CylindricalFunctor div,
        CylindricalFunctor divvvz
        ): f0{v_x, v_y, v_z},
        m_div(div), m_divvvz(divvvz) {}
    ///replace with given functors
    void reset(  CylindricalFunctor v_x,
        CylindricalFunctor v_y,
        CylindricalFunctor v_z,
        CylindricalFunctor div,
        CylindricalFunctor divvvz
        )
    {
        f0.reset( v_x,v_y,v_z);
        m_div = div;
        m_divvvz = divvvz;
    }
    ///type conversion: Lvl2 can also be used as Lvl1
    operator CylindricalVectorLvl0 ()const {return f0;}
    /// x-component of the vector
    const CylindricalFunctor& x()const{return f0.x();}
    /// y-component of the vector
    const CylindricalFunctor& y()const{return f0.y();}
    /// z-component of the vector
    const CylindricalFunctor& z()const{return f0.z();}
    /// \f$\nabla\cdot v\f$
    const CylindricalFunctor& div()const{return m_div;}
    /// \f$\nabla\cdot (v/z)\f$
    const CylindricalFunctor& divvvz()const{return m_divvvz;}
    private:
    CylindricalVectorLvl0 f0;
    CylindricalFunctor m_div, m_divvvz;
};

/**
 * @brief Return scalar product of two vector fields \f$ v_0w_0 + v_1w_1 + v_2w_2\f$
 */
struct ScalarProduct : public aCylindricalFunctor<ScalarProduct>
{
    ScalarProduct( CylindricalVectorLvl0 v, CylindricalVectorLvl0 w) : m_v(v), m_w(w){}
    double do_compute( double R, double Z) const
    {
        return m_v.x()(R,Z)*m_w.x()(R,Z)
             + m_v.y()(R,Z)*m_w.y()(R,Z)
             + m_v.z()(R,Z)*m_w.z()(R,Z);
    }
  private:
    CylindricalVectorLvl0 m_v, m_w;
};

/**
 * @brief Return norm of scalar product of two vector fields \f$ \sqrt{v_0w_0 + v_1w_1 + v_2w_2}\f$
 *
 * short for \c dg::compose( sqrt, ScalarProduct( v,w))
 */
struct SquareNorm : public aCylindricalFunctor<SquareNorm>
{
    SquareNorm( CylindricalVectorLvl0 v, CylindricalVectorLvl0 w) : m_s(v, w){}
    double do_compute( double R, double Z) const
    {
        return sqrt(m_s(R,Z));
    }
  private:
    ScalarProduct m_s;
};


/*!@brief \f$ \chi^{ij} = b^ib^j\f$
 *
 * Creates the two times contravariant tensor that,
 * when applied to a covariant vector, creates a vector
 * aligned to \c b.
 *
 * @param bhat The (unit) vector field \c b to align to
 * @param g The vector field is pushed unto this grid
 * @return The tensor \c chi living on the coordinate system given by \c g
 * @tparam Geometry3d A three-dimensional geometry
 */
template<class Geometry3d>
dg::SparseTensor<typename Geometry3d::host_vector> createAlignmentTensor(
    const dg::geo::CylindricalVectorLvl0& bhat, const Geometry3d& g)
{
    using host_vector = typename Geometry3d::host_vector;
    SparseTensor<host_vector> t;
    std::array<host_vector,3> bt;
    dg::pushForward( bhat.x(), bhat.y(), bhat.z(), bt[0], bt[1], bt[2], g);
    std::vector<host_vector> chi(6, dg::evaluate( dg::zero,g));
    dg::blas1::pointwiseDot( bt[0], bt[0], chi[0]);
    dg::blas1::pointwiseDot( bt[0], bt[1], chi[1]);
    dg::blas1::pointwiseDot( bt[0], bt[2], chi[2]);
    dg::blas1::pointwiseDot( bt[1], bt[1], chi[3]);
    dg::blas1::pointwiseDot( bt[1], bt[2], chi[4]);
    dg::blas1::pointwiseDot( bt[2], bt[2], chi[5]);
    t.idx(0,0) = 0, t.idx(0,1) = t.idx(1,0) = 1,
        t.idx(0,2) = t.idx(2,0) = 2;
    t.idx(1,1) = 3, t.idx(1,2) = t.idx(2,1) = 4;
    t.idx(2,2) = 5;
    t.values() = chi;
    return t;
}
/*!@brief \f$ h^{ij} = g^{ij} - b^ib^j\f$
 *
 * Creates the two times contravariant tensor that,
 * when applied to a covariant vector, creates a vector
 * perpendicular to \c b.
 *
 * @param bhat The (unit) vector field \c b
 * @param g The vector field is pushed unto this grid
 * @return The tensor \c h living on the coordinate system given by \c g
 * @tparam Geometry3d A three-dimensional geometry
 */
template<class Geometry3d>
dg::SparseTensor<typename Geometry3d::host_vector> createProjectionTensor(
    const dg::geo::CylindricalVectorLvl0& bhat, const Geometry3d& g)
{
    using host_vector = typename Geometry3d::host_vector;
    dg::SparseTensor<host_vector> t = dg::geo::createAlignmentTensor( bhat, g);
    dg::SparseTensor<host_vector> m = g.metric();
    dg::blas1::axpby( 1., m.value(0,0), -1., t.values()[0]);
    dg::blas1::axpby( 1., m.value(0,1), -1., t.values()[1]);
    dg::blas1::axpby( 1., m.value(0,2), -1., t.values()[2]);
    dg::blas1::axpby( 1., m.value(1,1), -1., t.values()[3]);
    dg::blas1::axpby( 1., m.value(1,2), -1., t.values()[4]);
    dg::blas1::axpby( 1., m.value(2,2), -1., t.values()[5]);
    return t;
}

///@}
}//namespace geo
}//namespace dg
