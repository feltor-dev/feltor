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
    protected:
    virtual ~aBinaryFunctor(){}
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
* @brief Implementation helper for the clone pattern

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
* @brief Manage and deep copy binary Functors on the heap
*/
struct aBinaryFunctorBundle
{
    protected:
    aBinaryFunctorBundle( const aBinaryFunctorBundle& b)
    {
        //deep copy
        for( unsigned i=0; i<p_.size(); i++)
            p_[i] = b.p_[i]->clone();
    }
    aBinaryFunctorBundle& operator=( const aBinaryFunctorBundle& b)
    {
        aBinaryFunctorBundle temp(b);
        std::swap( temp.p_, p_);
        return *this;
    }
    ~aBinaryFunctorBundle(){
        for( unsigned i=0; i<p_.size(); i++)
            delete p_[i];
    }
    std::vector<aBinaryFunctor*> p_;
};

/**
* @brief This struct bundles a function and its first derivatives
*/
struct BinaryFunctorsLvl1 : public aBinaryFunctorBundle
{
    /**
    * @brief Take ownership of newly allocated functors
    *
    * @param f f 
    * @param fx partial f / partial x
    * @param fy partial f / partial y
    */
    BinaryFunctorsLvl1( aBinaryFunctor* f, aBinaryFunctor* fx, aBinaryFunctor* fy): p_(3)
    {
        p_[0] = f;
        p_[1] = fx;
        p_[2] = fy;
    }
    ///f 
    const aBinaryFunctor& f()const{return *p_[0];}
    /// partial f / partial x
    const aBinaryFunctor& fx()const{return *p_[1];}
    /// partial f / partial y
    const aBinaryFunctor& fy()const{return *p_[2];}
};
/**
* @brief This struct bundles a function and its first and second derivatives
*/
struct BinaryFunctorsLvl2 : public aBinaryFunctorBundle
{
    /**
    * @brief Take ownership of newly allocated functors
    *
    * @param f f 
    * @param fx partial f / partial x
    * @param fy partial f / partial y
    * @param fxx partial2 f / partial x2
    * @param fxy partial2 f / partial x /partial y
    * @param fyy partial2 f / partial y2
    */
    BinaryFunctorsLvl2( aBinaryFunctor* f, aBinaryFunctor* fx, aBinaryFunctor* fy,
    aBinaryFunctor* fxx, aBinaryFunctor* fxy, aBinaryFunctor* fyy): p_(6)
    {
        p_[0] = f;
        p_[1] = fx;
        p_[2] = fy;
        p_[3] = fxx;
        p_[4] = fxy;
        p_[5] = fyy;
    }
    /// f
    const aBinaryFunctor& f()const{return *p_[0];}
    /// partial f /partial x
    const aBinaryFunctor& fx()const{return *p_[1];}
    /// partial f /partial y
    const aBinaryFunctor& fy()const{return *p_[2];}
    /// partial^2f/partial x^2
    const aBinaryFunctor& fxx()const{return *p_[3];}
    /// partial^2 f / partial x partial y
    const aBinaryFunctor& fxy()const{return *p_[4];}
    /// partial^2f/partial y^2
    const aBinaryFunctor& fyy()const{return *p_[5];}
};
/**
* @brief This struct bundles a symmetric tensor and its divergence
*/
struct BinarySymmTensorLvl1 : public aBinaryFunctorBundle
{
    /**
    * @brief Take ownership of newly allocated functors
    *
    * let's assume the tensor is called chi
    * @param chi_xx contravariant xx component
    * @param chi_xy contravariant xy component
    * @param chi_yy contravariant yy component
     * @param divChiX \f$ \partial_x \chi^{xx} + \partial_y\chi^{yx}\f$ is the x-component of the divergence of the tensor \f$ \chi\f$
     * @param divChiY \f$ \partial_x \chi^{xy} + \partial_y\chi^{yy}\f$ is the y-component of the divergence of the tensor \f$ \chi \f$
    */
    BinarySymmTensorLvl1( aBinaryFunctor* chi_xx, aBinaryFunctor* chi_xy, aBinaryFunctor* chi_yy,
    aBinaryFunctor* divChiX, aBinaryFunctor* divChiY): p_(5)
    {
        p_[0] = chi_xx;
        p_[1] = chi_xy;
        p_[2] = chi_yy;
        p_[3] = divChiX;
        p_[4] = divChiY;
    }
    ///xx component
    const aBinaryFunctor& xx()const{return *p_[0];}
    ///xy component
    const aBinaryFunctor& xy()const{return *p_[1];}
    ///yy component
    const aBinaryFunctor& yy()const{return *p_[2];}
    ///x component of the divergence 
    const aBinaryFunctor& divX()const{return *p_[3];}
    ///y component of the divergence 
    const aBinaryFunctor& divY()const{return *p_[4];}
};


///@}
}//namespace geo
}//namespace dg
