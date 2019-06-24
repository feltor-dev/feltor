#pragma once

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
    SimpsonsRule( const ContainerType& copyable): m_integral( m_integral), m_head( m_head)
    {
    }
    const ContainerType& copyable() const{ return m_integral;}
    void init(  value_type t0, const ContainerType& u0) {
        dg::blas1::copy( u0, m_u0);
        m_t[0] = t0;
        flush();
    }
    const ContainerType& flush() {
        m_counter = 0;
        dg::blas1::scal( m_integral, 0.);
    }
    void add( value_type t_new, const ContainerType& u_new){
        if( t_new < m_t0)
            throw dg::Error(dg::Message()<<"New time must be strictly larger than old time!");
        if( m_counter % 2 == 0)
        {
            //Trapezoidal rule
            dg::blas1::axpbypgz( 0.5/(t_new - m_t[0]), u_new,
                0.5/(t_new - m_t[0]), m_u[0] , 1., m_integral);
            m_t[1] = m_t[0];
            m_u[1].swap( m_u[0]);
            m_t[0] = t_new;
            dg::blas1::copy( u_new, m_u[0]);
        }
        else
        {
            //Simpson's rule
            value_type t0 = m_t[1], t1 = m_t[0], t2 = t_new;
            value_type pre = (t2-t0)/6./(t0-t1)/(t1-t2);
            value_type pre2 = t0*t0+3.*t1*t1+2.*t0*t2-4.*t0*t1-2.*t1*t2;
            value_type pre1 = -(t0-t2)*(t0-t2);
            value_type pre0 = t2*t2+3.*t1*t1+2.*t0*t2-4.*t1*t2-2.*t0*t1;

            dg::blas1::axpby( pre2/pre, u_new, 1., m_integral);
            //Also subtract old Trapezoidal rule
            dg::blas1::axpbypgz(
                pre1/pre-0.5/(m_t[0] - m_t[1]), m_u[0],
                pre0/pre-0.5/(m_t[0] - m_t[1]), m_u[1],
                1., m_integral);
            m_t[1] = m_t[0];
            m_u[1].swap( m_u[0]);
            m_t[0] = t_new;
            dg::blas1::copy( u_new, m_u[0]);
        }
        m_counter++;
    }
    const ContainerType& get_sum() const{
        return m_integral;
    }
    private:
    unsigned m_counter;
    ContainerType m_integral;
    std::vector<value_type> m_t;
    std::vector<ContainerType> m_u;
};
}//namespace dg
