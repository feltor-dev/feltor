#pragma once
#include <functional>
#include "blas1.h"
#include "backend/tensor_traits.h"

/*! @file
  @brief An abstract ODE integrator
  */
namespace dg
{

/**
 * @brief Switch for the Timeloop integrate function
 * @ingroup time_utils
 */
enum class to
{
    exact, //!< match the ending exactly
    at_least //!< integrate to the end or further
};

/**
 * @brief Convert integration mode to string
 *
 * Converts
 * - dg::to::exact to "exact"
 * - dg::to::at_least to "at_least"
 * - default "Not specified"
 * @param mode the mode
 *
 * @return string as defined above
 * @ingroup time_utils
 */
inline std::string to2str( enum to mode)
{
    std::string s;
    switch(mode)
    {
        case(dg::to::exact): s = "exact"; break;
        case(dg::to::at_least): s = "at_least"; break;
        default: s = "Not specified!!";
    }
    return s;
}
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
     * @attention The function may throw \c dg::Error (or anything derived
     * from \c std::exception).
     */
    void integrate( value_type t0, const ContainerType& u0,
                   value_type t1, ContainerType& u1)
    {
        if( t0 == t1)
        {
            dg::blas1::copy( u0, u1);
            return;
        }
        value_type time = t0;
        try{
            do_integrate( time, u0, t1, u1, dg::to::exact);
        }
        catch ( dg::Error& err)
        {
            err.append_line( dg::Message(_ping_) << "Error in aTimeloop::integrate at time "<<time<<" with t0 "<<t0<<" and t1 "<<t1);
            throw;
        }
    }

    /**
     * @brief Build your own timeloop
     *
     * Integrate an ode from <tt> t = t0 </tt> until <tt> t >= t1 </tt>
     * using a set of discrete steps **with or
     * without** forcing a certain timestep to match t1 exactly.
     * For example, if \c t0=0 and \c t1=1 and the timestep is \c dt=0.6 then
     * the timestepper without forcing stops at \c t1=1.2
     * This behaviour is useful especially for adaptive timesteppers because
     * it allows to implement timeloops with minimal interference with the
     * controller
     * @code
     * double time = t_begin;
     * double deltaT = (t_end - t_begin) / (double)maxout;
     * for( unsigned u=1; u<=maxout; u++)
     * {
     *     timeloop.integrate( time, u0, t_begin + u*deltaT, u0,
     *          u<maxout ? dg::to::at_least :  dg::to::exact);
     *     // Takes as many steps as an uninterupted integrate
     *     // from t_begin to t_end
     *
     *     // Warning: the following version does not(!) finish at t_end:
     *     //timeloop.integrate( time, u0, time+deltaT, u0,
     *     //     u<maxout ? dg::to::at_least :  dg::to::exact);
     *  }
     * @endcode
     * @note This function is re-entrant, i.e. if you integrate from t0 to t1
     * and then from t1 to t2 the timestepper does not need to re-initialize on
     * the second call
     * @param t0 (read-write) initial time on entry; on exit it is the value to
     * where the ode is actually integrated, corresponding to \c u1
     * @param u0 initial value at \c t0
     * @param t1 (read-only) end time
     * @param u1 (write only) contains the result corresponding to exactly or
     * at least t1 on output (may alias u0)
     * @param mode either integrate exactly to \c t1 or at least to \c t1. In
     * \c dg::at_least mode the timestep is bound only by \c t1-t0
     * @attention The function may throw \c dg::Error (or anything derived
     * from \c std::exception).
     */
    void integrate( value_type& t0, const ContainerType& u0,
                   value_type t1, ContainerType& u1, enum to mode )
    {
        if( t0 == t1)
        {
            dg::blas1::copy( u0, u1);
            return;
        }

        value_type t_begin = t0;
        try{
            do_integrate( t0, u0, t1, u1, mode);
        }
        catch ( dg::Error& err)
        {
            err.append_line( dg::Message(_ping_) << "Error in aTimeloop::integrate at time "<<t0<<" with t0 "<<t_begin<<" and t1 "<<t1 << " and mode "<<to2str(mode));
            throw;
        }
    }

    /**
     * @brief The current timestep
     *
     * @return The \c dt value at the end of the latest call to integrate.
     * If integrate
     * fails for some reason then return the timestep at which the failure
     * happens.  Undefined if \c integrate has not
     * been called at least once.
     */
    value_type get_dt() const { return do_dt(); }


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
    virtual void do_integrate(value_type& t0, const container_type& u0,
            value_type t1, container_type& u1, enum to mode) const = 0;
    virtual value_type do_dt() const = 0;


};

///@}

}//namespace dg
