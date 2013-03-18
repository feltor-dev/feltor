#ifndef _DG_LAPLACE_CUH
#define _DG_LAPLACE_CUH

#include <cusp/coo_matrix.h>

#include "functions.h"
#include "operators.cuh"

namespace dg
{

template<class container = cups::coo_matrix< int, double, cusp::host_memory> >
class Laplace
{
  public:
    typedef container Matrix;  //datatype of the underlying container
    const container& data() const {return M;}
    container& data() { return M;}
    Laplace( container& m): M(m) {}
  private:
    Matrix& M;
  protected:
    typedef cusp::coo_matrix<int, double, cusp::host_memory> HMatrix; 
    void add_index( HMatrix&, int&, unsigned i, unsigned j, unsigned k, unsigned l, double value );
};

template< size_t n, class container>
class Laplace_Per : public Laplace< container>
{
  public:
    Laplace( unsigned N, double h, double alpha = 1.);
    typedef Laplace<container> View;
  private:
    
    Operator<double, n> a,b;
};


template<class container>
void Laplace<container>::add_index( HMatrix& hm, int& number, unsigned i, unsigned j, unsigned k, unsigned l, double value )
{
    hm.row_indices[number] = n*i+k;
    hm.column_indices[number] = n*j+l;
    hm.values[number] = value;

    number++;

}

//change next line if you change sparse matrix type
template<size_t n>
Laplace<n>::Laplace( unsigned N, double h, double alpha): M( /*n*N, n*N, 3*n*n*N, 3*n*/)
{
    HMatrix A( n*N, n*N, 3*n*n*N);
    Operator<double, n> l( detail::lilj);
    Operator<double, n> r( detail::rirj);
    Operator<double, n> lr( detail::lirj);
    Operator<double, n> rl( detail::rilj);
    Operator<double, n> d( detail::pidxpj);
    Operator<double, n> t( detail::pipj_inv);
    t *= 2./h;
    a = lr*t*rl+(d+l)*t*(d+l).transpose() + alpha*(l+r);
    b = -((d+l)*t*rl+alpha*rl);
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            add_index( A, number, 0,0,k,l, a(k,l));
        for( unsigned l=0; l<n; l++)
            add_index( A, number, 0,1,k,l, b(k,l));
        for( unsigned l=0; l<n; l++)
            add_index( A, number, 0,N-1,k,l, b(l,k));
    }
    for( unsigned i=1; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                add_index(A, number, i, i-1, k, l, b(l,k));
            for( unsigned l=0; l<n; l++)
                add_index(A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                add_index(A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            add_index( A, number, N-1,0,  k,l, b(k,l));
        for( unsigned l=0; l<n; l++)
            add_index( A, number, N-1,N-2,k,l, b(l,k));
        for( unsigned l=0; l<n; l++)
            add_index( A, number, N-1,N-1,k,l, a(k,l));
    }
    M=A; //copy matrix to device

};


template<size_t n>
class Laplace_Dir
{
  public:
    Laplace_Dir( double h = 2.);
    const Operator<double,n>& get_a() const {return a;}
    const Operator<double,n>& get_b() const {return b;}
    const Operator<double,n>& get_ap() const {return ap;}
    const Operator<double,n>& get_bp() const {return bp;}
  private:
    Operator<double, n> a,b;
    Operator<double, n> ap,bp;

};

template<size_t n>
Laplace_Dir<n>::Laplace_Dir( double h) 
{
    Operator<double, n> l( detail::lilj);
    Operator<double, n> r( detail::rirj);
    Operator<double, n> lr( detail::lirj);
    Operator<double, n> rl( detail::rilj);
    Operator<double, n> d( detail::pidxpj);
    Operator<double, n> s( detail::pipj);
    Operator<double, n> t( detail::pipj_inv);
    t *= 2./h;

    a = lr*t*rl+(d+l)*t*(d+l).transpose() + (l+r);
    b = -((d+l)*t*rl+rl);
    ap = d*t*d.transpose() + l + r;
    bp = -(d*t*rl + rl);
};

} //namespace dg

#include "blas/thrust_vector.cuh"
#include "blas/laplace.cuh"

#endif // _DG_LAPLACE_CUH
