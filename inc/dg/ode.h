#pragma once
#include <functional>
#include "blas1.h"
#include "backend/tensor_traits.h"

/*! @file
  @brief An abstract ODE integrator
  */
namespace dg
{
///@addtogroup time
///@{

/*!@brief Abstract timeloop independent of stepper and ODE
 *
 * This class enables to write abstract time-loops that are independent of
 * the used stepper (e.g. dg::RungeKutta, dg::ExplicitMultistep, ...) and
 * the differential equation in use. The recommended way to implement it
 * is using a std::function that erases the latter tpyes and emulates the
 * step function of the stepper type.
 * @copydoc hide_ContainerType
 * @ingroup time_utils
 */
template<class ContainerType>
struct aTimeloop
{
    using value_type = dg::get_value_type<ContainerType>;
    using container_type = ContainerType;

    /**
     * @brief Integrate a differential equation between given bounds
     *
     * Integrate an ode from <tt> t = t0 </tt> until <tt> t == t1 </tt>
     * using a set of discrete steps and
     * forcing the last timestep to land on t1 exactly.
     * @note This function is re-entrant, i.e. if you integrate from t0 to t1
     * and then from t1 to t2 the timestepper does not need to re-initialize on
     * the second call
     * @param t0 initial time
     * @param u0 initial value at \c t0
     * @param t1 end time
     * @param u1 (write only) contains the result corresponding to t1 on output.
     * May alias \c u0.
     * @note May not work for a multistep integrator if the interval does not
     * evenly multiply the (fixed) timestep
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
            err.append_line( dg::Message(_ping_) << "Error in aTimeloop::integrate with t0 "<<t0<<" and t1 "<<t1);
            throw;
        }
    }

    /**
     * @brief Integrate a differential equation at least between given bounds
     *
     * Integrate an ode from <tt> t = t0 </tt> until <tt> t >= t1 </tt>
     * using a set of discrete steps
     * **without** forcing a certain timestep to match t1 exactly.
     * For example, if \c t0=0 and \c t1=1 and the timestep is \c dt=0.6 then
     * the timestepper without forcing stops at \c t1=1.2
     * This behaviour is useful especially for adaptive timesteppers because
     * it allows to implement timeloops with minimal interference with the
     * controller
     * @note This function is re-entrant, i.e. if you integrate from t0 to t1
     * and then from t1 to t2 the timestepper does not need to re-initialize on
     * the second call
     * @param t0 initial time
     * @param u0 initial value at \c t0
     * @param t1 (read-write) on entry this is the value to which to integrate
     * the ode on exit it is the value to where the ode is actually integrated
     * @param u1 (write only) contains the result corresponding to t1 on output
     */
    void integrate_at_least( value_type t0, const ContainerType& u0,
                   value_type& t1, ContainerType& u1)
    {
        if( t0 == t1)
        {
            dg::blas1::copy( u0, u1);
            return;
        }
        try{
            do_integrate( t0, u0, t1, u1, false);
        }
        catch ( dg::Error& err)
        {
            err.append_line( dg::Message(_ping_) << "Error in aTimeloop::integrate_at_least with t0 "<<t0<<" and t1 "<<t1);
            throw;
        }
    }

    /**
    * @brief Abstract copy method that returns a copy of *this on the heap
    *
    * @return a copy of *this on the heap
    * @sa dg::ClonePtr
    */
    virtual aTimeloop* clone() const=0;

    virtual ~aTimeloop(){}
    protected:
    ///empty
    aTimeloop(){}
    ///empty
    aTimeloop(const aTimeloop& ){}
    ///return *this
    aTimeloop& operator=(const aTimeloop& ){ return *this; }
    private:
    // the bool indicates whether or not to check the end time condition
    virtual void do_integrate(value_type t0, const container_type& u0, value_type& t1, container_type& u1, bool check) const = 0;
};

///@}

}//namespace dg
