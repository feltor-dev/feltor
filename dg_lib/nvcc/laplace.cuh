#ifndef _DG_LAPLACE_CUH
#define _DG_LAPLACE_CUH

#include <cassert>
#include <cusp/coo_matrix.h>

#include "functions.h"
#include "operators.cuh"

namespace dg
{

template<size_t n>
class Laplace
{
  public:
    typedef cusp::coo_matrix<int, double, cusp::host_memory> HMatrix; 
    typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix; 
    Laplace( unsigned N, double h = 2., double alpha = 1.);
    const DMatrix& get_m() const {return M;}
  private:
    Operator<double, n> a,b;
    DMatrix M;
    void add_index( HMatrix&, int&, unsigned i, unsigned j, unsigned k, unsigned l, double value );

};

template<size_t n>
void Laplace<n>::add_index( HMatrix& hm, int& number, unsigned i, unsigned j, unsigned k, unsigned l, double value )
{
    hm.row_indices[number] = n*i+k;
    hm.column_indices[number] = n*j+l;
    hm.values[number] = value;

    number++;

}

//change here if you change sparse matrix type
template<size_t n>
Laplace<n>::Laplace( unsigned N, double h, double alpha): M( n*N, n*N, 3*n*n*N, 3*n)
{
    HMatrix A( n*N, n*N, 3*n*n*N);
    Operator<double, n> l( lilj);
    Operator<double, n> r( rirj);
    Operator<double, n> lr( lirj);
    Operator<double, n> rl( rilj);
    Operator<double, n> d( pidxpj);
    Operator<double, n> t( pipj_inv);
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
    //std::cout << "ORDERED?\n";
    //std::cout << A.is_sorted_by_row_and_column() << std::endl;
    //cusp::ell_matrix<int, double, cusp::host_memory> C(A);
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
    Operator<double, n> l( lilj);
    Operator<double, n> r( rirj);
    Operator<double, n> lr( lirj);
    Operator<double, n> rl( rilj);
    Operator<double, n> d( pidxpj);
    Operator<double, n> s( pipj);
    Operator<double, n> t( pipj_inv);
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
