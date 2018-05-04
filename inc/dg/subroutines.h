#pragma once

namespace dg{


struct equals
{
    template< class T1, class T2>
    void operator()( T1& out, T2 in) const
    {
        out = in;
    }
};
struct plus_equals
{
    template< class T1, class T2>
    void operator()( T1& out, T2 in) const
    {
        out += in;
    }
};
struct minus_equals
{
    template< class T1, class T2>
    void operator()( T1& out, T2 in) const
    {
        out -= in;
    }
};
struct times_equals
{
    template< class T1, class T2>
    void operator()( T1& out, T2 in) const
    {
        out *= in;
    }
};
struct divides_equals
{
    template< class T1, class T2>
    void operator()( T1& out, T2 in) const
    {
        out /= in;
    }
};

}//namespace dg
