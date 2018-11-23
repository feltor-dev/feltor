#pragma once
#include <functional>
#include "dg/backend/memory.h"
#include "dg/topology/geometry.h"

namespace dg
{
namespace geo
{

/*! @brief Inject both 2d and 3d \c operator() to a 2d functor
 *
 * The purpose of this class is to extend any 2d Functor to a
 * 3d Functor by defining \f$ f(x,y,z) := f(x,y)\f$. This class is
 * especially useful in an interface since any 2d functor can be converted
 * to it (type erasure property of the \c std::function that we use
 * to implement this class).
 * @note If you want to write a functor that is both 2d and 3d directly,
 * it is easier to derive from \c aCylindricalFunctor
 * @ingroup fluxfunctions
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
/// @ingroup fluxfunctions
using CylindricalFunctor = RealCylindricalFunctor<double>;

/**
* @brief Represent functions written in cylindrical coordinates
        that are independent of the angle phi serving as both 2d and 3d functions

* The rational is that these functors can serve as both 2d and 3d functors
* where the 3d functor trivially redirects to the 2d version.
* This behaviour is injected into all classes that derive from this class
* via the Curiously Recurring Template Pattern (CRTP).
* @ingroup fluxfunctions
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
};

/**
 * @brief The constant functor
 * \f[ f(x,y) = c\f]
* @ingroup fluxfunctions
 */
struct Constant: public aCylindricalFunctor<Constant>
{
    Constant(double c):c_(c){}
    double do_compute(double R,double Z)const{return c_;}
    private:
    double c_;
};

/**
* @brief This struct bundles a function and its first derivatives
*
* @snippet hector_t.cu doxygen
* @ingroup fluxfunctions
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
* @snippet hector_t.cu doxygen
* @ingroup fluxfunctions
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

/// A symmetric 2d tensor field and its divergence
///@snippet hector_t.cu doxygen
///@ingroup fluxfunctions
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
///@snippet ds_t.cu doxygen
///@ingroup fluxfunctions
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

/*!@brief \f[ \chi^{ij} = b^ib^j\f]
 *
 * Creates the two times contravariant tensor that,
 * when applied to a covariant vector, creates a vector
 * aligned to \c b.
 *
 * @param bhat The (unit) vector field \c b to align to
 * @param g The vector field is pushed unto this grid
 * @return The tensor \c chi living on the coordinate system given by \c g
 * @tparam Geometry3d A three-dimensional geometry
 * @ingroup fluxfunctions
 */
template<class Geometry3d>
dg::SparseTensor<dg::get_host_vector<Geometry3d>> createAlignmentTensor(
    const dg::geo::CylindricalVectorLvl0& bhat, const Geometry3d& g)
{
    using host_vector = dg::get_host_vector<Geometry3d>;
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
/*!@brief \f[ \chi^{ij} = g^{ij} - b^ib^j\f]
 *
 * Creates the two times contravariant tensor that,
 * when applied to a covariant vector, creates a vector
 * perpendicular to \c b.
 *
 * @param bhat The (unit) vector field \c b
 * @param g The vector field is pushed unto this grid
 * @return The tensor \c chi living on the coordinate system given by \c g
 * @tparam Geometry3d A three-dimensional geometry
 * @ingroup fluxfunctions
 */
template<class Geometry3d>
dg::SparseTensor<dg::get_host_vector<Geometry3d>> createProjectionTensor(
    const dg::geo::CylindricalVectorLvl0& bhat, const Geometry3d& g)
{
    using host_vector = dg::get_host_vector<Geometry3d>;
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
