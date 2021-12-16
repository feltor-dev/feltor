#pragma once
#include <functional>
#include "blas1.h"
#include "backend/tensor_traits.h"

/*! @file
  @brief An abstract ODE integrator
  */
namespace dg
{

/*!@brief A generic interface class to Feltor's numerical integrators
 *
 * @copydoc hide_ContainerType
 */
template<class ContainerType>
struct aOdeIntegrator
{
    using value_type = dg::get_value_type<ContainerType>;
    using container_type = ContainerType;

    /**
     * @brief Integrate a differential equation between given bounds
     *
     * @note This function is re-entrant
     * @param stepper can be called like \c stepper( t0, u0, dt)
     * @param t0 initial time
     * @param u0 initial value at \c t0
     * @param t1 end time
     * @param u1 (write only) contains the result corresponding to t1 on output
     * @param dt The initial timestep guess (if 0 the function chooses something
     * for you). The exact value is not really
     * important, the stepper does not even have to succeed. Usually the
     * control function will very(!) quickly adapt the stepsize in just one or
     * two steps (even if it's several orders of magnitude off in the beginning).
     * @attention The integrator may throw if it detects too small timesteps, too
     * many failures, NaN, Inf, or other non-sanitary behaviour
     * @copydoc hide_ContainerType
     */
    void integrate( value_type t0, const ContainerType& u0,
                   value_type t1, ContainerType& u1)
    {
        if( t0 == t1)
        {
            dg::blas1::copy( u0, u1);
            return;
        }
        try{
            do_integrate( t0, u0, t1, u1, true);
        }
        catch ( dg::Error& err)
        {
            err.append_line( dg::Message(_ping_) << "Error in ODE integrate");
            throw;
        }
    }
    void integrate_min( value_type t0, const ContainerType& u0,
                   value_type& t1, ContainerType& u1)
    {
        if( t0 == t1)
        {
            dg::blas1::copy( u0, u1);
            return;
        }
        try{
            do_integrate_min( t0, u0, t1, u1, false);
        }
        catch ( dg::Error& err)
        {
            err.append_line( dg::Message(_ping_) << "Error in ODE integrate_min");
            throw;
        }
    }
    /**
    * @brief Abstract clone method that returns a copy on the heap
    *
    * @return a copy of *this on the heap
    */
    virtual aOdeIntegrator* clone() const=0;
    virtual ~aOdeIntegrator(){}
    protected:
    ///empty
    aOdeIntegrator(){}
    ///empty
    aOdeIntegrator(const aOdeIntegrator& ){}
    ///return *this
    aOdeIntegrator& operator=(const aOdeIntegrator& ){ return *this; }
    private:
    // the bool indicates whether or not to check the end time condition
    virtual void do_integrate(value_type t0, const container_type& u0, value_type& t1, container_type& u1, bool check) const = 0;
};

}//namespace dg
