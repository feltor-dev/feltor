#pragma once

#include <map>
#include <functional>
#include "dg/algorithm.h"

struct ShearLayer{

    ShearLayer( double rho, double delta, double lx, double ly): m_rho(rho), m_delta(delta), m_lx(lx), m_ly(ly) {}
    DG_DEVICE
    double operator()(double x, double y) const{
    if( x<= m_lx/2.)
        return m_delta*cos(2.*M_PI*y/m_ly) - 1./m_rho/cosh( (2.*M_PI*x/m_lx-M_PI/2.)/m_rho)/cosh( (2.*M_PI*x/m_lx-M_PI/2.)/m_rho);
    return m_delta*cos(2.*M_PI*y/m_ly) + 1./m_rho/cosh( (3.*M_PI/2.-2.*M_PI*x/m_lx)/m_rho)/cosh( (3.*M_PI/2.-2.*M_PI*x/m_lx)/m_rho);
    }
    private:
    double m_rho, m_delta, m_lx, m_ly;    
};
