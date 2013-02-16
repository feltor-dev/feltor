#ifndef _DG_LAPLACE_
#define _DG_LAPLACE_

#include "projection_functions.h"
#include "operators.h"

namespace dg
{

template<size_t n>
class Laplace
{
  public:
    Laplace( double c = 1);
    const Operator<double,n>& get_a() const {return a;}
    const Operator<double,n>& get_b() const {return b;}
  private:
    Operator<double, n> a,b;

};

template<size_t n>
Laplace<n>::Laplace( double c) 
{
    Operator<double, n> l( lilj);
    Operator<double, n> r( rirj);
    Operator<double, n> lr( lirj);
    Operator<double, n> rl( rilj);
    Operator<double, n> d( pidxpj);
    Operator<double, n> s( pipj);
    Operator<double, n> t( pipj_inv);
    t *= c;

    a = lr*t*rl+(d+l)*t*(d+l).transpose() + (l+r);
    b = -((d+l)*t*rl+rl);
};

template<size_t n>
class Laplace_Dir
{
  public:
    Laplace_Dir( double c = 1);
    const Operator<double,n>& get_a() const {return a;}
    const Operator<double,n>& get_b() const {return b;}
    const Operator<double,n>& get_ap() const {return a;}
    const Operator<double,n>& get_bp() const {return b;}
  private:
    Operator<double, n> a,b;
    Operator<double, n> ap,bp;

};

template<size_t n>
Laplace_Dir<n>::Laplace_Dir( double c) 
{
    Operator<double, n> l( lilj);
    Operator<double, n> r( rirj);
    Operator<double, n> lr( lirj);
    Operator<double, n> rl( rilj);
    Operator<double, n> d( pidxpj);
    Operator<double, n> s( pipj);
    Operator<double, n> t( pipj_inv);
    t *= c;

    a = lr*t*rl+(d+l)*t*(d+l).transpose() + (l+r);
    b = -((d+l)*t*rl+rl);
    ap = d*t*d.transpose() + l + r;
    bp = -((d+l)*t*rl + rl);
};

}

#endif // _DG_LAPLACE_
