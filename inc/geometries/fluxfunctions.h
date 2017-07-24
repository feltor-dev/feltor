#pragma once

namespace dg
{
namespace geo
{
/**
* @brief This functor represents functions written in cylindrical coordinates
        that are independent of the angle phi
  @ingroup fluxfunctions
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
  @ingroup fluxfunctions
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



struct aBinaryFunctorBundle
{
    aBinaryFunctorBundle( 
        aBinaryFunctor* psip,
        aBinaryFunctor* psipR,
        aBinaryFunctor* psipZ,
        aBinaryFunctor* psipRR,
        aBinaryFunctor* psipRZ,
        aBinaryFunctor* psipZZ,
        ):p_(6){ 
            R0_(R0),
            p_[0] = psip,
            p_[1] = psipR,
            p_[2] = psipZ,
            p_[3] = psipRR,
            p_[4] = psipRZ,
            p_[5] = psipZZ,
        }
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
    private:
    std::vector<aBinaryFunctor*> p_;
};
}//namespace geo
}//namespace dg
