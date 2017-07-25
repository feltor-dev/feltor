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
    /// constructor is deleted
    aBinaryFunctorBundle();
    aBinaryFunctorBundle( const aBinaryFunctorBundle& b)
    {
        p_.resize( b.p_.size());
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
    * @param f \f$ f(x,y)\f$ the function in some coordinates (x,y)
    * @param fx \f$ \partial f / \partial x \f$ its derivative in the first coordinate
    * @param fy \f$ \partial f / \partial y \f$ its derivative in the second coordinate
    */
    BinaryFunctorsLvl1( aBinaryFunctor* f, aBinaryFunctor* fx, aBinaryFunctor* fy): p_(3)
    {
        p_[0] = f;
        p_[1] = fx;
        p_[2] = fy;
    }
    /// \f$ f \f$
    const aBinaryFunctor& f()const{return *p_[0];}
    /// \f$ \partial f / \partial x \f$ 
    const aBinaryFunctor& dfx()const{return *p_[1];}
    /// \f$ \partial f / \partial y\f$
    const aBinaryFunctor& dfy()const{return *p_[2];}
};
/**
* @brief This struct bundles a function and its first and second derivatives
*/
struct BinaryFunctorsLvl2 : public BinaryFunctorsLvl1
{
    /**
    * @copydoc BinaryFunctorsLvl1
    * @param fxx \f$ \partial^2 f / \partial x^2\f$ second derivative in first coordinate
    * @param fxy \f$ \partial^2 f / \partial x \partial y\f$ second mixed derivative 
    * @param fyy \f$ \partial^2 f / \partial y^2\f$ second derivative in second coordinate
    */
    BinaryFunctorsLvl2( aBinaryFunctor* f, aBinaryFunctor* fx, aBinaryFunctor* fy,
    aBinaryFunctor* fxx, aBinaryFunctor* fxy, aBinaryFunctor* fyy): BinaryFunctorsLvl1(f,fx,fy) 
    {
        p_.push_back( fxx);
        p_.push_back( fxy);
        p_.push_back( fyy);
    }
    /// \f$ \partial^2f/\partial x^2\f$
    const aBinaryFunctor& dfxx()const{return *p_[3];}
    /// \f$ \partial^2 f / \partial x \partial y\f$
    const aBinaryFunctor& dfxy()const{return *p_[4];}
    /// \f$ \partial^2f/\partial y^2\f$
    const aBinaryFunctor& dfyy()const{return *p_[5];}
};
/**
* @brief This struct bundles a symmetric tensor and its divergence
*/
struct BinarySymmTensorLvl1 : public aBinaryFunctorBundle
{
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
    BinarySymmTensorLvl1( aBinaryFunctor* chi_xx, aBinaryFunctor* chi_xy, aBinaryFunctor* chi_yy,
    aBinaryFunctor* divChiX, aBinaryFunctor* divChiY): p_(5)
    {
        p_[0] = chi_xx;
        p_[1] = chi_xy;
        p_[2] = chi_yy;
        p_[3] = divChiX;
        p_[4] = divChiY;
    }
    ///xy component \f$ \chi^{xx}\f$ 
    const aBinaryFunctor& xx()const{return *p_[0];}
    ///xy component \f$ \chi^{xy}\f$ 
    const aBinaryFunctor& xy()const{return *p_[1];}
    ///yy component \f$ \chi^{yy}\f$ 
    const aBinaryFunctor& yy()const{return *p_[2];}
     /// \f$ \partial_x \chi^{xx} + \partial_y\chi^{yx}\f$ is the x-component of the divergence of the tensor \f$ \chi\f$
    const aBinaryFunctor& divX()const{return *p_[3];}
     /// \f$ \partial_x \chi^{xy} + \partial_y\chi^{yy}\f$ is the y-component of the divergence of the tensor \f$ \chi \f$
    const aBinaryFunctor& divY()const{return *p_[4];}
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
