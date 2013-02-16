#ifndef _DG_PROJECTION_
#define _DG_PROJECTION_

#include <iostream>

namespace dg{

double pipj( unsigned i, unsigned j)
{
    if( i==j) 
        return 2./(2.*(double)i+1.);
    return 0;
}
double pipj_inv( unsigned i, unsigned j)
{
    if( i==j)
        return (2.*(double)i+1.)/2.;
    return 0;
}
double pidxpj( unsigned i, unsigned j)
{
    if( i < j)
    {
        if( (i+j)%2 == 0)
            return 0;
        else
            return 2;
    }
    return 0;
}
double rirj( unsigned i, unsigned j)
{
    return 1.;
}
double rilj( unsigned i, unsigned j)
{
    if( j%2==0)
        return 1.;
    return -1.;
}
double lirj( unsigned i, unsigned j)
{
    return rilj(j,i);
}
double lilj( unsigned i, unsigned j)
{
    if( (i+j)%2 ==0)
        return 1.;
    return -1.;
}
}


#endif //_DG_PROJECTION_
