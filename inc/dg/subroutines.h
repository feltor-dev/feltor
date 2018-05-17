#pragma once
#include "dg/geometry/functions.h"

namespace dg{

///@addtogroup functions
///@{

///\f$ x=y\f$
struct equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1& out, T2 in) const
    {
        out = in;
    }
};
///\f$ x+=y\f$
struct plus_equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1& out, T2 in) const
    {
        out += in;
    }
};
///\f$ x-=y\f$
struct minus_equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1& out, T2 in) const
    {
        out -= in;
    }
};
///\f$ x*=y\f$
struct times_equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1& out, T2 in) const
    {
        out *= in;
    }
};
///\f$ x/=y\f$
struct divides_equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1& out, T2 in) const
    {
        out /= in;
    }
};
///@}

}//namespace dg
