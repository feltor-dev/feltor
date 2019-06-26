#pragma once

#include <list>
#include "backend/exceptions.h"
#include "blas1.h"

/*! @file
 * @brief Equidistant time-integrators
 */

namespace dg{

template<class ContainerType>
struct SimpsonsRule
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    SimpsonsRule() = default;
    SimpsonsRule( const ContainerType& copyable):
        m_counter(0), m_integral( copyable),
        m_t(2,0.), m_u( 2, copyable)
    {
    }
    const ContainerType& copyable() const{ return m_integral;}
    void init(  value_type t0, const ContainerType& u0) {
        dg::blas1::copy( u0, m_u.front());
        m_t.front() = t0;
        flush();
    }
    void flush() {
        m_counter = 0;
        dg::blas1::scal( m_integral, 0.);
    }
    void add( value_type t_new, const ContainerType& u_new){
        if( t_new < m_t.front())
            throw dg::Error(dg::Message()<<"New time must be strictly larger than old time!");
        auto pt0 = m_t.begin();
        auto pt1 = std::next( pt0);
        auto pu0 = m_u.begin();
        auto pu1 = std::next( pu0);
        value_type t0 = *pt1, t1 = *pt0, t2 = t_new;
        if( m_counter % 2 == 0)
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
            //Also subtract old Trapezoidal rule
            dg::blas1::axpbypgz(
                pre1-0.5*(t1-t0), *pu0,
                pre0-0.5*(t1-t0), *pu1,
                1., m_integral);
        }
        m_t.splice( pt0, m_t, pt1, m_t.end());
        m_u.splice( pu0, m_u, pu1, m_u.end());
        m_t.front() = t_new; //and now remove zeroth element (it is moved now!)
        dg::blas1::copy( u_new, m_u.front());
        m_counter++;
    }
    const ContainerType& get_sum() const{
        return m_integral;
    }
    private:
    unsigned m_counter;
    ContainerType m_integral;
    std::list<value_type> m_t;
    std::list<ContainerType> m_u;
};

}//namespace dg
