#pragma once

namespace dg
{
namespace geo
{
///@addtogroup fluxfunctions
///@{

/**
* @brief This functor represents functions written in cylindrical coordinates
        that are independent of the angle phi
*/
struct aBinaryFunctor
{
    /**
    * @brief The function value
    *
    * @param R radius (cylindrical coordinate)
    * @param Z height (cylindrical coordinate)
    *
    * @return f(R,Z)
    */
    virtual double operator()(double R, double Z) const=0;
    /**
    * @brief Redirects to the 2D version
    *
    * @param R radius (cylindrical coordinate)
    * @param Z height (cylindrical coordinate)
    * @param phi angle (cylindrical coordinate)
    *
    * @return f(R,Z)
    */
    double operator()(double R, double Z, double phi)
    {
        return this->operator()(R,Z);
    }
    /**
    * @brief abstract copy of a binary functor
    *
    * @return a functor on the heap
    */
    virtual aBinaryFunctor* clone()const=0;
    virtual ~aBinaryFunctor(){}
    protected:
    /**
    * @brief We do not allow object slicing so the copy is protected
    */
    aBinaryFunctor(const aBinaryFunctor&){}
    /**
    * @brief We do not allow object slicing so the assignment is protected
    */
    aBinaryFunctor& operator=(const aBinaryFunctor&){return *this}
};

/**
* @brief Intermediate implementation helper class for the clone pattern with CRTP

    https://katyscode.wordpress.com/2013/08/22/c-polymorphic-cloning-and-the-crtp-curiously-recurring-template-pattern/
*/
template<class Derived>
struct dg::geo::aCloneableBinaryFunctor : public dg::geo::aBinaryFunctor
{
    /**
    * @brief Returns a copy of the functor dynamically allocated on the heap
    *
    * @return new copy of the functor
    */
    virtual Derived* clone() const
    {
        return new Derived(static_cast<Derived const &>(*this));
    }
};
/**
 * @brief With this adapater class you can make any Functor cloneable
 *
 * @tparam BinaryFunctor must overload the operator() like
 * double operator()(double,double)const;
 */
template<class BinaryFunctor>
struct BinaryFunctorAdapter : public dg::geo::aCloneableBinaryFunctor<Adapter>
{
    BinaryFunctorAdapter( const BinaryFunctor& f):f_(f){}
    double operator()(double x, double y)const{return f_(x,y);}
    private:
    BinaryFunctor f_;
};
/**
 * @brief Use this function when you want to convert any Functor to aBinaryFunctor
 *
 * @tparam BinaryFunctor must overload the operator() like
 * double operator()(double,double)const;
 * @param f const reference to a functor class
 * @return a newly allocated instance of aBinaryFunctor on the heap
 * @note the preferred way is to derive your Functor from aCloneableBinaryFunctor but if you can't or don't want to for whatever reason then use this to make one
 */
temmplate<class BinaryFunctor>
aBinaryFunctor* make_aBinaryFunctor(const BinaryFunctor& f){return new BinaryFunctorAdapter<BinaryFunctor>(f);}

/**
* @brief This struct bundles a function and its first derivatives
*/
struct BinaryFunctorsLvl1 
{
    ///the access functions are undefined as long as the class remains empty
    BinaryFunctorsLvl1(){}
    /**
    * @brief Take ownership of newly allocated functors
    *
    * @param f \f$ f(x,y)\f$ the function in some coordinates (x,y)
    * @param fx \f$ \partial f / \partial x \f$ its derivative in the first coordinate
    * @param fy \f$ \partial f / \partial y \f$ its derivative in the second coordinate
    */
    BinaryFunctorsLvl1( const aBinaryFunctor* f, const aBinaryFunctor* fx, const aBinaryFunctor* fy)
    {
        reset(f,fx,fy);
    }
    void reset( const aBinaryFunctor* f, const aBinaryFunctor* fx, const aBinaryFunctor* fy)
    {
        p_[0].reset(f);
        p_[1].reset(fx);
        p_[2].reset(fy);
    }
    /// \f$ f \f$
    const aBinaryFunctor& f()const{return p_[0].get();}
    /// \f$ \partial f / \partial x \f$ 
    const aBinaryFunctor& dfx()const{return p_[1].get();}
    /// \f$ \partial f / \partial y\f$
    const aBinaryFunctor& dfy()const{return p_[2].get();}
    private:
    Handle<aBinaryFunctor> p_[3];
};
/**
* @brief This struct bundles a function and its first and second derivatives
*/
struct BinaryFunctorsLvl2 
{
    ///the access functions are undefined as long as the class remains empty
    BinaryFunctorsLvl2(){}
    /**
    * @copydoc BinaryFunctorsLvl1
    * @param fxx \f$ \partial^2 f / \partial x^2\f$ second derivative in first coordinate
    * @param fxy \f$ \partial^2 f / \partial x \partial y\f$ second mixed derivative 
    * @param fyy \f$ \partial^2 f / \partial y^2\f$ second derivative in second coordinate
    */
    BinaryFunctorsLvl2( const aBinaryFunctor* f, const aBinaryFunctor* fx, const aBinaryFunctor* fy, const aBinaryFunctor* fxx, const aBinaryFunctor* fxy, const aBinaryFunctor* fyy): f(f,fx,fy), f1(fxx,fxy,fyy) 
    { }
    void reset( const aBinaryFunctor* f, const aBinaryFunctor* fx, const aBinaryFunctor* fy, const aBinaryFunctor* fxx, const aBinaryFunctor* fxy, const aBinaryFunctor* fyy){ 
        f.reset(f,fx,fy), f1.reset(fxx,fxy,fyy) 
    }
    operator BinaryFunctorsLvl1 ()const {return f;}
    /// \f$ f \f$
    const aBinaryFunctor& f()const{return f.f();}
    /// \f$ \partial f / \partial x \f$ 
    const aBinaryFunctor& dfx()const{return f.dfx();}
    /// \f$ \partial f / \partial y\f$
    const aBinaryFunctor& dfy()const{return f.dfy();}
    /// \f$ \partial^2f/\partial x^2\f$
    const aBinaryFunctor& dfxx()const{return f1.f();}
    /// \f$ \partial^2 f / \partial x \partial y\f$
    const aBinaryFunctor& dfxy()const{return f1.fx();}
    /// \f$ \partial^2f/\partial y^2\f$
    const aBinaryFunctor& dfyy()const{return f1.fy();}
    private:
    BinaryFunctorsLvl1 f,f1;
};

/// A symmetric 2d tensor field and its divergence
struct BinarySymmTensorLvl1
{
    BinarySymmTensorLvl1( ){}
    /**
     * @brief Take ownership of newly allocated functors
     *
     * let's assume the tensor is called \f$ \chi \f$ (chi)
     * @param chi_xx contravariant xx component \f$ \chi^{xx}\f$ 
     * @param chi_xy contravariant xy component \f$ \chi^{xy}\f$ 
     * @param chi_yy contravariant yy component \f$ \chi^{yy}\f$ 
     * @param divChiX \f$ \partial_x \chi^{xx} + \partial_y\chi^{yx}\f$ is the x-component of the divergence of the tensor \f$ \chi\f$
     * @param divChiY \f$ \partial_x \chi^{xy} + \partial_y\chi^{yy}\f$ is the y-component of the divergence of the tensor \f$ \chi \f$
    */
    BinarySymmTensorLvl1( const aBinaryFunctor* chi_xx, const aBinaryFunctor* chi_xy, const aBinaryFunctor* chi_yy, const aBinaryFunctor* divChiX, const aBinaryFunctor* divChiY)
    {
        reset(chi_xx,chi_xy,chi_yy,divChiX,divChiY);
    }
    void reset( const aBinaryFunctor* chi_xx, const aBinaryFunctor* chi_xy, const aBinaryFunctor* chi_yy, const aBinaryFunctor* divChiX, const aBinaryFunctor* divChiY)
    {
        p_[0].reset( chi_xx);
        p_[1].reset( chi_xy);
        p_[2].reset( chi_yy);
        p_[3].reset( divChiX);
        p_[4].reset( divChiY);
    }
    ///xy component \f$ \chi^{xx}\f$ 
    const aBinaryFunctor& xx()const{return p_[0].get();}
    ///xy component \f$ \chi^{xy}\f$ 
    const aBinaryFunctor& xy()const{return p_[1].get();}
    ///yy component \f$ \chi^{yy}\f$ 
    const aBinaryFunctor& yy()const{return p_[2].get();}
     /// \f$ \partial_x \chi^{xx} + \partial_y\chi^{yx}\f$ is the x-component of the divergence of the tensor \f$ \chi\f$
    const aBinaryFunctor& divX()const{return p_[3].get();}
     /// \f$ \partial_x \chi^{xy} + \partial_y\chi^{yy}\f$ is the y-component of the divergence of the tensor \f$ \chi \f$
    const aBinaryFunctor& divY()const{return p_[4].get();}
    private:
    Handle<aBinaryFunctor> p_[5];
};

/// A vector field with three components that depend only on the first two coordinates
struct BinaryVectorLvl0
{
    BinaryVectorLvl0(){}
    BinaryVectorLvl0( const aBinaryFunctor* v_x, const aBinaryFunctor* v_y, const aBinaryFunctor* v_z)
    {
        reset(v_x,v_y,v_z);
    }
    void reset( const aBinaryFunctor* v_x, const aBinaryFunctor* v_y, const aBinaryFunctor* v_z)
    {
        p_[0].reset(v_x);
        p_[1].reset(v_y);
        p_[2].reset(v_z);
    }
    /// x-component of the vector
    const aBinaryFunctor& x()const{return p_[0];}
    /// y-component of the vector
    const aBinaryFunctor& y()const{return p_[1];}
    /// z-component of the vector
    const aBinaryFunctor& z()const{return p_[2];}
    private:
    Handle<aBinaryFunctor> p_[3];
};

struct Constant:public aCloneableBinaryOperator<Constant> 
{ 
    Constant(double c):c_(c){}
    double operator()(double R,double Z)const{return c_;}
    private:
    double c_;
};

///@}
}//namespace geo
}//namespace dg
