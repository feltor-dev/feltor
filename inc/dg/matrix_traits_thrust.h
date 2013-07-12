#ifndef _DG_MATRIX_TRAITS_THRUST
#define _DG_MATRIX_TRAITS_THRUST

#include <stdexcept>
#include "matrix_traits.h"
#include "matrix_categories.h"
#include "vector_categories.h"

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

#include "grid.cuh"
#include "dlt.h"
namespace dg{

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

namespace create{

template< class T>
thrust::host_vector<T> s1d( const Grid1d<T>& g)
{
    thrust::host_vector<T> v( g.size());
    for( unsigned i=0; i<g.N(); i++)
        for( unsigned j=0; j<g.n(); j++)
            v[i*g.n()+j] = g.hx() / (double)(2*j+1);
    return v;
}
template< class T>
thrust::host_vector<T> t1d( const Grid1d<T>& g)
{
    thrust::host_vector<T> v = s1d( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}
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
template< class T>
thrust::host_vector<T> t2d( const Grid<T>& g)
{
    thrust::host_vector<T> v = s2d( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;

}


template <class T>
thrust::host_vector<T> w1d( const Grid1d<T>& g)
{
    thrust::host_vector<T> weights( g.n());
    switch(g.n())
    {
        case( 1): 
            for( unsigned i=0; i<g.n(); i++)
                weights[i] = DLT<1>::weight[i];
            break;
        case( 2): 
            for( unsigned i=0; i<g.n(); i++)
                weights[i] = DLT<2>::weight[i];
            break;
        case( 3): 
            for( unsigned i=0; i<g.n(); i++)
                weights[i] = DLT<3>::weight[i];
            break;
        case( 4): 
            for( unsigned i=0; i<g.n(); i++)
                weights[i] = DLT<4>::weight[i];
            break;
        case( 5): 
            for( unsigned i=0; i<g.n(); i++)
                weights[i] = DLT<5>::weight[i];
            break;
        default:
            throw std::out_of_range("not implemented yet");
    }
    thrust::host_vector<T> v( g.size());
    for( unsigned i=0; i<g.N(); i++)
        for( unsigned j=0; j<g.n(); j++)
            v[i*g.n() + j] = g.h()/2.*weights[j];
    return v;
}
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
template <class T>
thrust::host_vector<T> w2d( const Grid<T>& g)
{
    thrust::host_vector<T> weights( g.n());
    switch( g.n())
    {
        case( 1): 
            for( unsigned i=0; i<g.n(); i++)
                weights[i] = DLT<1>::weight[i];
            break;
        case( 2): 
            for( unsigned i=0; i<g.n(); i++)
                weights[i] = DLT<2>::weight[i];
            break;
        case( 3): 
            for( unsigned i=0; i<g.n(); i++)
                weights[i] = DLT<3>::weight[i];
            break;
        case( 4): 
            for( unsigned i=0; i<g.n(); i++)
                weights[i] = DLT<4>::weight[i];
            break;
        case( 5): 
            for( unsigned i=0; i<g.n(); i++)
                weights[i] = DLT<5>::weight[i];
            break;
        default:
            throw std::out_of_range("not implemented yet");
    }
    thrust::host_vector<T> v( g.size());
    for( unsigned i=0; i<g.size(); i++)
        v[i] = g.hx()*g.hy()/4.*weights[detail::get_i(g.n(), i)]*weights[detail::get_j(g.n(), i)];
    return v;
}
template <class T>
thrust::host_vector<T> v2d( const Grid<T>& g)
{
    thrust::host_vector<T> v = w2d( g);
    for( unsigned i=0; i<g.size(); i++)
        v[i] = 1./v[i];
    return v;
}
template <class T>
thrust::host_vector<T> abscissas( const Grid1d<T>& g)
{
    thrust::host_vector<T> abscissas( g.n());
    switch( g.n())
    {
        case( 1): 
            for( unsigned i=0; i<g.n(); i++)
                abscissas[i] = DLT<1>::abscissa[i];
            break;
        case( 2): 
            for( unsigned i=0; i<g.n(); i++)
                abscissas[i] = DLT<2>::abscissa[i];
            break;
        case( 3): 
            for( unsigned i=0; i<g.n(); i++)
                abscissas[i] = DLT<3>::abscissa[i];
            break;
        case( 4): 
            for( unsigned i=0; i<g.n(); i++)
                abscissas[i] = DLT<4>::abscissa[i];
            break;
        case( 5): 
            for( unsigned i=0; i<g.n(); i++)
                abscissas[i] = DLT<5>::abscissa[i];
            break;
        default:
            throw std::out_of_range("not implemented yet");
    }
    thrust::host_vector<T> v(g.size()); 
    T xp=1.;
    for( unsigned i=0; i<g.N(); i++)
    {
        for( unsigned j=0; j<g.n(); j++)
            v[i*g.n()+j] =  g.h()/2.*(xp + abscissas[j])+g.x0();
        xp+=2.;
    }
    return v;
}

}//namespace create

}//namespace dg

#endif//_DG_MATRIX_TRAITS_THRUST
