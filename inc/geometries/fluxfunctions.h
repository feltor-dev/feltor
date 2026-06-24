#pragma once
#include <functional>
#include "dg/algorithm.h"

namespace dg
{
namespace geo
{

///@addtogroup fluxfunctions
///@{

/**
* @brief Represent (potentially axisymmetric) functions \f$f(R,Z,\varphi)\f$ written in Cylindrical coordinates

* The rational is that axisymmetric functors \f$f(R,Z,\varphi)\f$ can serve as
* both 2d and 3d functors as it is independent of the angle:
* \f[
* f_{2d}(R,Z)\equiv f(R,Z,\varphi)
* \f]
*
* If a given 3d functor \f$ f(R,Z,\varphi)\f$ is not axisymmetric a
* corresponding 2d functor can be defined by fixing the angle to a given value
* \f[
* f_{2d,0}(R,Z) \equiv f(R,Z,\varphi_0)
* \f]
* \f$ \phi_0 = 0\f$ by default, use \c set_phi to change.
*
* This behaviour is injected into all classes that derive from this class
* via the Curiously Recurring Template Pattern (CRTP).
* All classes need to implement the \c do_compute function which computes \f$f(R,Z,\varphi)\f$
* @sa \c aCylindricalFunctor
* @sa The default implementation is \c RealCylindricalFunctor
* @tparam Derived Interface: <tt> double do_compute(double,double,double) const</tt>
*/
template<class Derived>
struct aCylindricalFunctor
{
    /**
    * @brief <tt>do_compute(R,Z, P0)</tt>
    *
    * @param R radius (cylindrical coordinate)
    * @param Z height (cylindrical coordinate)
    *
    * @return f(R,Z,P0)
    *
    * @sa set_phi to change the default P0 value
    * @attention The \c do_compute function should be considered private and
    * its interface can change
    */
    double operator()(double R, double Z) const
    {
        return operator()(R,Z,m_P);
    }
    /**
    * @brief <tt> do_compute(R,Z,P)</tt>
    *
    * @param R radius (cylindrical coordinate)
    * @param Z height (cylindrical coordinate)
    * @param P toroidal angle (clockwise when seen from above)
    *
    * @return f(R,Z,P)
    * @attention The \c do_compute function should be considered private and
    * its interface can change
    */
    double operator()(double R, double Z, double P)const
    {
        const Derived& underlying = static_cast<const Derived&>(*this);
        return underlying.do_compute(R,Z,P);
    }
    /**
     * @brief Set the default angle \c P0 that the 2d operator uses
     *
     * This is only relevant for non-axisymmetric functors and only for the 2d operator.
     * @param phi New P0 value for the 2d operator
     */
    void set_phi( double P0) { m_P = P0;}
    /// @brief Read access to the default phi value P0
    double get_phi() const { return m_P;}
    private:
    double m_P = 0.;
#ifndef __CUDACC__ //nvcc below 10 has problems with the following construct
    //This trick avoids that classes inherit from the wrong Base:
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

/*! @brief Inject both 2d and 3d \c operator() to a 2d functor
 *
 * The purpose of this class is to serve as a general purpose
 * functor parameter in interfaces
 * to catch any other functor of type \c aCylindricalFunctor or
 * in fact any 2d or 3d functor type (type erasure property of the \c
 * std::function that we use to implement this class).
 * @note If you want to avoid the indirection inherent in the \c std::function
 * it is easier to derive from \c aCylindricalFunctor
 * @sa \c aCylindricalFunctor can be converted to this class
 */
struct CylindricalFunctor : public aCylindricalFunctor<CylindricalFunctor>
{
    CylindricalFunctor(){}

    template<class TernaryFunctor>
    CylindricalFunctor( const TernaryFunctor& f):
        m_f(f) {}
    double do_compute( double R, double Z, double P) const { return m_f(R,Z,P);}
    private:
    // Optimization note: from https://www.boost.org/doc/libs/1_45_0/doc/html/function/faq.html#id1284915
    // the performance overhead of calling std::function is about (20 +- 10)ns
    std::function<double(double,double,double)> m_f;
};

//If ever we need float there is an issue with templated Derived classes:
//https://stackoverflow.com/questions/2940402/templated-derived-class-in-crtp-curiously-recurring-template-pattern
//Essentially then **all** derived classes need to be templates
//using CylindricalFunctor = RealCylindricalFunctor;
/**
 * @brief \f$ f(R,Z,P) = c\f$
 */
struct Constant: public aCylindricalFunctor<Constant>
{
    Constant(double c):c_(c){}
    double do_compute(double,double,double)const{return c_;}
    private:
    double c_;
};
/**
 * @brief
 * \f$ f(R,Z,P)= \begin{cases}
 0 \text{ if } Z < Z_X \\
 1 \text{ else }
 \end{cases}
 \f$
 @note the 1 is inclusive i.e. if Z == Z_X the functor always returns 1
 */
struct ZCutter : public aCylindricalFunctor<ZCutter>
{
    ZCutter(double ZX, int sign = +1): m_heavi( ZX, sign){}
    double do_compute(double, double Z,double) const {
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
    double do_compute( double R, double Z, double P) const
    {
        bool negative = false;
        dg::create::detail::shift( negative, R, m_g.bcx(), m_g.x0(), m_g.x1());
        dg::create::detail::shift( negative, Z, m_g.bcy(), m_g.y0(), m_g.y1());
        if( negative) return -m_f(R,Z,P);
        return m_f( R, Z, P);
    }
    private:
    dg::Grid2d m_g;
    CylindricalFunctor m_f;
};

/**
* @brief This struct bundles a function and its first derivatives
*
* @snippet flux_b.cpp hector
*/
struct CylindricalFunctorsLvl1
{
    ///the access functions are undefined as long as the class remains empty
    CylindricalFunctorsLvl1(){}
    /**
    * @brief Construct with given functors
    *
    * @param f \f$ f(x,y,z)\f$ the function in some coordinates (x,y,z)
    * @param fx \f$ \partial f / \partial x \f$ its derivative in the first coordinate
    * @param fy \f$ \partial f / \partial y \f$ its derivative in the second coordinate
    * @param fz \f$ \partial f / \partial z \f$ its derivative in the third coordinate
    */
    CylindricalFunctorsLvl1(  const CylindricalFunctor& f,  const CylindricalFunctor& fx,
        const CylindricalFunctor& fy, const CylindricalFunctor& fz = Constant(0)) : p_{{ f, fx, fy, fz}} {
    }
    ///copy given functors
    void reset( const CylindricalFunctor& f, const CylindricalFunctor& fx, const CylindricalFunctor& fy, const CylindricalFunctor& fz = Constant(0))
    {
        p_[0] = f;
        p_[1] = fx;
        p_[2] = fy;
        p_[3] = fz;
    }
    /// \f$ f \f$
    const CylindricalFunctor& f()const{return p_[0];}
    /// \f$ \partial f / \partial x \f$
    const CylindricalFunctor& dfx()const{return p_[1];}
    /// \f$ \partial f / \partial y\f$
    const CylindricalFunctor& dfy()const{return p_[2];}
    /// \f$ \partial f / \partial z\f$
    const CylindricalFunctor& dfz()const{return p_[3];}
    private:
    std::array<CylindricalFunctor,4> p_;
};


/**
* @brief This struct bundles a function and its first and second derivatives
*
* @snippet flux_b.cpp hector
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
        f0(f,fx,fy), f1(fxx,fxy,fyy), f2(Constant(0), Constant(0), Constant(0))
    { }
    CylindricalFunctorsLvl2(  CylindricalFunctor f,  CylindricalFunctor fx,
        CylindricalFunctor fy,   CylindricalFunctor fz,
        CylindricalFunctor fxx, CylindricalFunctor fxy, CylindricalFunctor fxz,
        CylindricalFunctor fyy, CylindricalFunctor fyz,
        CylindricalFunctor fzz
        ):
        f0(f,fx,fy, fz), f1(fxx,fxy,fyy), f2(fxz, fyz, fzz)
    { }
    ///Replace with given Functors
    void reset( CylindricalFunctor f, CylindricalFunctor fx,
        CylindricalFunctor fy, CylindricalFunctor fxx,
        CylindricalFunctor fxy, CylindricalFunctor fyy)
    {
        f0.reset( f,fx,fy), f1.reset(fxx,fxy,fyy), f2.reset( Constant(0), Constant(0), Constant(0));
    }
    ///Replace with given Functors
    void reset(  CylindricalFunctor f,  CylindricalFunctor fx,
        CylindricalFunctor fy,   CylindricalFunctor fz,
        CylindricalFunctor fxx, CylindricalFunctor fxy, CylindricalFunctor fxz,
        CylindricalFunctor fyy, CylindricalFunctor fyz,
        CylindricalFunctor fzz)
    {
        f0.reset(f,fx,fy, fz), f1.reset(fxx,fxy,fyy), f2.reset(fxz, fyz, fzz);
    }
    ///type conversion: Lvl2 can also be used as Lvl1
    operator CylindricalFunctorsLvl1 ()const {return f0;}
    /// \f$ f \f$
    const CylindricalFunctor& f()const{return f0.f();}
    /// \f$ \partial f / \partial x \f$
    const CylindricalFunctor& dfx()const{return f0.dfx();}
    /// \f$ \partial f / \partial y\f$
    const CylindricalFunctor& dfy()const{return f0.dfy();}
    /// \f$ \partial f / \partial z\f$
    const CylindricalFunctor& dfz()const{return f0.dfz();}
    /// \f$ \partial^2f/\partial x^2\f$
    const CylindricalFunctor& dfxx()const{return f1.f();}
    /// \f$ \partial^2 f / \partial x \partial y\f$
    const CylindricalFunctor& dfxy()const{return f1.dfx();}
    /// \f$ \partial^2 f / \partial x \partial z\f$
    const CylindricalFunctor& dfxz()const{return f2.f();}
    /// \f$ \partial^2f/\partial y^2\f$
    const CylindricalFunctor& dfyy()const{return f1.dfy();}
    /// \f$ \partial^2f/\partial y\partial z\f$
    const CylindricalFunctor& dfyz()const{return f2.dfx();}
    /// \f$ \partial^2f/\partial z^2\f$
    const CylindricalFunctor& dfzz()const{return f2.dfy();}
    private:
    CylindricalFunctorsLvl1 f0,f1,f2;
};


/**
 * @brief This function finds critical points of psi (any point with vanishing gradient in R and Z, including the X-point or O-point) via Newton iteration applied to the gradient of psi
 *
 * Newton iteration applied to \f$ \nabla \psi (\vec x) = 0 \f$ reads
 * \f[ \vec x_{i+1} = \vec x_i - H^{-1} \nabla \psi (\vec x_i)\f]
 * where H is the Hessian matrix.
 * The inverse of the Hessian matrix is computed analytically
 * @param psi \f$ \psi(R,Z)\f$
 * @param RC start value on input, critical point on output
 * @param ZC start value on input, critical point on output
 * @param P0 The plane in which to find vanishing gradient in
 * @return 0 if no critical point or Hessian (determinant) is zero,
 * 1 if local minimum,
 * 2 if local maximum,
 * 3 if saddle point
 * @ingroup misc_geo
 */
inline int findCriticalPoint( const CylindricalFunctorsLvl2& psi, double& RC, double& ZC, double P0 = 0.)
{
    std::array<double, 2> X{ {0,0} }, XN(X), X_OLD(X);
    X[0] = RC, X[1] = ZC;
    double eps = 1e10, eps_old= 2e10;
    unsigned counter = 0; //safety measure to avoid deadlock
    double psipRZ = psi.dfxy()(X[0], X[1], P0);
    double psipRR = psi.dfxx()(X[0], X[1], P0), psipZZ = psi.dfyy()(X[0],X[1], P0);
    double psipR  = psi.dfx()(X[0], X[1], P0), psipZ = psi.dfy()(X[0], X[1], P0);
    double D0 =  (psipZZ*psipRR - psipRZ*psipRZ);
    if(D0 == 0) // try to change initial guess slightly if we are very lucky
    {
        X[0] *= 1.0001, X[1]*=1.0001;
        psipRZ = psi.dfxy()(X[0], X[1], P0);
        psipRR = psi.dfxx()(X[0], X[1], P0), psipZZ = psi.dfyy()(X[0],X[1], P0);
        psipR  = psi.dfx()(X[0], X[1], P0), psipZ = psi.dfy()(X[0], X[1], P0);
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
        psipRZ = psi.dfxy()(X[0], X[1], P0);
        psipRR = psi.dfxx()(X[0], X[1], P0), psipZZ = psi.dfyy()(X[0],X[1], P0);
        psipR  = psi.dfx()(X[0], X[1], P0), psipZ = psi.dfy()(X[0], X[1], P0);
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
 * @param P0 Default plane to find O-point in
 * @return 1 if local minimum, 2 if local maximum,
 * @ingroup misc_geo
 */
inline int findOpoint( const CylindricalFunctorsLvl2& psi, double& RC, double& ZC, double P0 = 0.)
{
    int point = findCriticalPoint( psi, RC, ZC, P0);
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
 * @param P0 Default plane to find X-point in
 * @ingroup misc_geo
 */
inline void findXpoint( const CylindricalFunctorsLvl2& psi, double& RC, double& ZC, double P0 = 0.)
{
    int point = findCriticalPoint( psi, RC, ZC, P0);
    if( point != 3)
        throw dg::Error(dg::Message(_ping_)<<"There is no X-point near "<<RC<<" "<<ZC);
}


/// A symmetric 2d tensor field and its divergence
///@snippet flux_b.cpp hector
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

/// @brief A vector field with three components
///@snippet ds_b.cpp doxygen
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
    double do_compute( double R, double Z, double P) const
    {
        return m_v.x()(R,Z)*m_w.x()(R,Z,P)
             + m_v.y()(R,Z)*m_w.y()(R,Z,P)
             + m_v.z()(R,Z)*m_w.z()(R,Z,P);
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
    double do_compute( double R, double Z, double P) const
    {
        return sqrt(m_s(R,Z,P));
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
