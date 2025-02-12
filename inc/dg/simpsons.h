#pragma once

#include <list>
#include "backend/exceptions.h"
#include "blas1.h"

/*! @file
 * @brief Equidistant time-integrator
 */
namespace dg{

/**
 * @brief Time integration based on Simpson's rule
 *
 * The intention of this class is to provide a means to continuously
 * integrate a sample of \f$( t_i, u_i)\f$ pairs that become available
 * one after the
 * other (e.g. from the integration of an ODE) and approximate
 *
 * \f[ \int_{t_0}^T u(t) dt \f]
 *
 * @note The algorithm simply integrates the Lagrange-polynomial through up to three data points. For equidistant Data points this equals either the Trapezoidal (linear) or the Simpson's rule (quadratic)
 * @sa For an explanation of Simpson's rule: https://en.wikipedia.org/wiki/Simpson%27s_rule
 *
 * The class works by first calling the \c init function to set the left-side
 * boundary and then adding values as they become available.
 * Calling the \c flush member resets the integral to 0 and the right boundary as
 * the new left boundary
 * @snippet{trimleft} simpsons_t.cpp docu
 * @copydoc hide_ContainerType
 * @ingroup integration
 */
template<class ContainerType>
struct Simpsons
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    /*! @brief Set integration order without initializing values
     * @param order number of vectors to use for integration.
         Choose 2 (linear) or 3 (parabola) integration.
     */
    Simpsons( unsigned order = 3): m_counter(0), m_order(order), m_t0(0)
    {
        set_order(order);
    }
    ///@copydoc Simpsons(unsigned)
    void set_order( unsigned order){
        m_order=order;
        m_t.resize( order-1);
        m_u.resize(order-1);
        if( !(order == 2 || order == 3))
            throw dg::Error(dg::Message()<<"Integration order must be either 2 or 3!");
    }
    ///Access current integration order
    unsigned get_order() const{return m_order;}
    /*! @brief Initialize the left-side boundary of the integration
     * @param t0 left-side time
     * @param u0 left-side value
     * @note The integral itself is initialized to zero
     */
    void init(  value_type t0, const ContainerType& u0) {
        m_integral = u0;
        for( auto& u: m_u)
            u = u0;
        for( auto& t: m_t)
            t = 0;
        m_t.front() = t0;
        flush();
        m_t0 = t0;
    }
    /*! @brief Reset the integral to zero and the last (t,u) pair in the add function as the new left-side
     */
    void flush() {
        m_counter = 0; //since the counter becomes zero we do not need to touch m_u and m_t since the next add triggers the Trapezoidal rule
        dg::blas1::scal( m_integral, 0.);
        m_t0 = m_t.front();
    }
    /*! @brief Add a new (t,u) pair to the time integral
     *
     * The times in subsequent calls must strictly increase
     * @param t_new time (must be strictly larger than in the previous call)
     * @param u_new value (must have the same size as \c u0 in the init function)
     * @attention the \c init function must be called before you can add values to the integral
     */
    void add( value_type t_new, const ContainerType& u_new){
        if( t_new < m_t.front())
            throw dg::Error(dg::Message()<<"New time must be strictly larger than old time (or you forgot to call the init function)!");
        auto pt0 = m_t.begin();
        auto pt1 = std::next( pt0);
        auto pu0 = m_u.begin();
        auto pu1 = std::next( pu0);
        value_type t0 = *pt1, t1 = *pt0, t2 = t_new;
        if( m_counter % 2 == 0 || m_order == 2)
        {
            //Trapezoidal rule
            dg::blas1::axpbypgz( 0.5*(t2 - t1), u_new,
                0.5*(t2 - t1), *pu0 , 1., m_integral);
        }
        else
        {
            //Simpson's rule
            value_type pre0 = (2.*t0-3.*t1+t2)*(t2-t0)/(6.*(t0-t1));
            value_type pre1 = (t2-t0)*(t2-t0)*(t2-t0)/(6.*(t0-t1)*(t1-t2));
            value_type pre2 = (t0-3.*t1+2.*t2)*(t0-t2)/(6.*(t1-t2));

            dg::blas1::axpby( pre2, u_new, 1., m_integral);
            dg::blas1::axpbypgz(
                pre1-0.5*(t1-t0), *pu0, //subtract last Trapezoidal step
                pre0-0.5*(t1-t0), *pu1,
                1., m_integral);
        }
        //splice does not copy or move anything, only the internal pointers of the list nodes are re-pointed
        m_t.splice( pt0, m_t, pt1, m_t.end());//permute elements
        m_u.splice( pu0, m_u, pu1, m_u.end());
        m_t.front() = t_new; //and now remove zeroth element
        dg::blas1::copy( u_new, m_u.front());
        m_counter++;
    }

    /*! @brief Access the current value of the time integral
     * @return the current integral
     */
    const ContainerType& get_integral() const{
        return m_integral;
    }
    /*! @brief Access the left and right boundary in time
     *
     * associated with the current value of the integral
     * @return the current integral boundaries
     */
    std::array<value_type,2> get_boundaries() const{
        std::array<value_type,2> times{ m_t0, m_t.front()};
        return times;
    }
    private:
    unsigned m_counter, m_order;
    ContainerType m_integral;
    //we use a list here to avoid explicitly calling the swap function
    std::list<value_type> m_t;
    std::list<ContainerType> m_u;
    value_type m_t0;
};

}//namespace dg
