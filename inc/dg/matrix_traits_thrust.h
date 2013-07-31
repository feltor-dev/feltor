#ifndef _DG_MATRIX_TRAITS_THRUST
#define _DG_MATRIX_TRAITS_THRUST

#include "matrix_traits.h"
#include "matrix_categories.h"
#include "vector_categories.h"

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

#include "grid.cuh"

namespace dg{

///@cond
template< class T>
struct MatrixTraits<thrust::host_vector<T> > {
    typedef T value_type;
    typedef ThrustVectorTag matrix_category; 
};
template< class T>
struct MatrixTraits<thrust::device_vector<T> > {
    typedef T value_type;
    typedef ThrustVectorTag matrix_category; 
};
///@endcond

namespace create{

///@addtogroup highlevel
///@{

///@cond
/**
* @brief create host_vector containing 1d L-space weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template< class T>
thrust::host_vector<T> s1d( const Grid1d<T>& g)
{
    thrust::host_vector<T> v( g.size());
    for( unsigned i=0; i<g.N(); i++)
        for( unsigned j=0; j<g.n(); j++)
            v[i*g.n()+j] = g.hx() / (double)(2*j+1);
    return v;
}
/**
* @brief create host_vector containing 1d L-space inverse weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template< class T>
thrust::host_vector<T> t1d( const Grid1d<T>& g)
{
    thrust::host_vector<T> v = s1d( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}
///@endcond
/**
* @brief create host_vector containing 2d L-space weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template< class T>
thrust::host_vector<T> s2d( const Grid<T>& g)
{
    thrust::host_vector<T> v( g.size());
    unsigned n= g.n();
    for( unsigned i=0; i<g.Ny(); i++)
        for( unsigned j=0; j<g.Nx(); j++)
            for( unsigned k=0; k<n; k++)
                for( unsigned l=0; l<n; l++)
                    v[i*n*n*g.Nx()+j*n*n+k*n+l] = g.hx()*g.hy() / (double)((2*k+1)*(2*l+1));
    return v;
}
/**
* @brief create host_vector containing 2d L-space inverse weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template< class T>
thrust::host_vector<T> t2d( const Grid<T>& g)
{
    thrust::host_vector<T> v = s2d( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;

}


///@cond
/**
* @brief create host_vector containing 1d X-space weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template <class T>
thrust::host_vector<T> w1d( const Grid1d<T>& g)
{
    thrust::host_vector<T> v( g.size());
    for( unsigned i=0; i<g.N(); i++)
        for( unsigned j=0; j<g.n(); j++)
            v[i*g.n() + j] = g.h()/2.*g.dlt().weights()[j];
    return v;
}
/**
* @brief create host_vector containing 1d X-space inverse weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template <class T>
thrust::host_vector<T> v1d( const Grid1d<T>& g)
{
    thrust::host_vector<T> v = w1d( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}

namespace detail{

int get_i( unsigned n, int idx) { return idx%(n*n)/n;}
int get_j( unsigned n, int idx) { return idx%(n*n)%n;}
}//namespace detail
///@endcond

/**
* @brief create host_vector containing 2d X-space weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template <class T>
thrust::host_vector<T> w2d( const Grid<T>& g)
{
    thrust::host_vector<T> v( g.size());
    for( unsigned i=0; i<g.size(); i++)
        v[i] = g.hx()*g.hy()/4.*g.dlt().weights()[detail::get_i(g.n(), i)]*g.dlt().weights()[detail::get_j(g.n(), i)];
    return v;
}
/**
* @brief create host_vector containing 2d X-space inverse weight coefficients
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template <class T>
thrust::host_vector<T> v2d( const Grid<T>& g)
{
    thrust::host_vector<T> v = w2d( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}
///@cond
/**
* @brief create host_vector containing 1d X-space abscissas 
*
* @tparam T value type
* @param g The grid 
*
* @return Host Vector
*/
template <class T>
thrust::host_vector<T> abscissas( const Grid1d<T>& g)
{
    thrust::host_vector<T> v(g.size()); 
    T xp=1.;
    for( unsigned i=0; i<g.N(); i++)
    {
        for( unsigned j=0; j<g.n(); j++)
            v[i*g.n()+j] =  g.h()/2.*(xp + g.dlt().abscissas()[j])+g.x0();
        xp+=2.;
    }
    return v;
}
///@endcond

///@}
}//namespace create

}//namespace dg

#endif//_DG_MATRIX_TRAITS_THRUST
