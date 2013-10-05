#include <vector>
#include <cusp/coo_matrix.h>
#include "grid.cuh"
#include "matrix_traits_thrust.h"

namespace dg{

namespace create{
namespace detail{

template<class T>
struct HelperMatrix
{
    HelperMatrix( unsigned m, unsigned n):rows_(m), cols_(n), data_(m*n){}
    /*! @brief access operator
     *
     * A range check is performed if DG_DEBUG is defined
     * @param i row index
     * @param j column index
     * @return reference to value at that location
     */
    T& operator()(const size_t i, const size_t j){
#ifdef DG_DEBUG
        assert( i<rows_ && j < cols_);
#endif
        return data_[ i*cols_+j];
    }
    /*! @brief const access operator
     *
     * @param i row index
     * @param j column index
     * @return const value at that location
     */
    const T& operator()(const size_t i, const size_t j) const {
#ifdef DG_DEBUG
        assert( i<rows_ && j < cols_);
#endif
        return data_[ i*cols_+j];
    }
    unsigned rows() const {return rows_;}
    unsigned cols() const {return cols_;}
    const std::vector<T>& data() const {return data_;}
    /*! @brief puts a matrix linewise in output stream
     *
     * @tparam Ostream The stream e.g. std::cout
     * @param os the outstream
     * @param mat the matrix to output
     * @return the outstream
     */
    template< class Ostream>
    friend Ostream& operator<<(Ostream& os, const HelperMatrix& mat)
    {
        for( size_t i=0; i < mat.rows_ ; i++)
        {
            for( size_t j = 0;j < mat.cols_; j++)
                os << mat(i,j) << " ";
            os << "\n";
        }
        return os;
    }

  private:
    unsigned rows_, cols_;
    std::vector<T> data_;
};

/**
 * @brief Compute the Kronecker product between two matrices
 *
 * See wikipedia for definition of the Kronecker Product
 * @param m1 left hand side
 * @param m2 right hand side
 *
 * @return  The Kronecker Product
 */
HelperMatrix<double> kronecker( const HelperMatrix<double>& m1, const HelperMatrix<double>& m2)
{
    HelperMatrix<double> prod( m1.rows()*m2.rows(), m1.cols()*m2.cols());
    for( unsigned i=0; i<m1.rows(); i++)
        for( unsigned j=0; j<m1.cols(); j++)
            for( unsigned k=0; k<m2.rows(); k++)
                for( unsigned l=0; l<m2.cols(); l++)
                    prod(i*m2.rows()+k, j*m2.cols()+l) = m1(i,j)*m2(k,l);
    return prod;
};


double LegendreP( unsigned n, double x)
{
    if( n==0 ) return 1;
    if( n==1 ) return x;
    return ((double)(2*n-1)*x*LegendreP( n-1, x) - (double)(n-1)*LegendreP( n-2, x))/(double)(n);
}
/**
 * @brief Create a projection matrix 
 *
 * Size is n_new*N_factor x n_old. It's intention is to project a given 
 * polynomial (x-space) in a cell to N subcells. 
 * @param n_old degree of given polynomial
 * @param n_new degree of polynomial in subcells (must be smaller than n_old)
 * @param N number of subcells 
 *
 * @return projection matrix in vector
 */
detail::HelperMatrix<double> projection( unsigned n_old, unsigned n_new, unsigned N)
{
    assert( n_old > 0);
    assert( n_new <= n_old && n_new > 0);
    assert( N > 0);
    dg::Grid1d<double> g_( -1, 1, n_new, N);
    thrust::host_vector<double> x = dg::create::abscissas( g_);
    unsigned rows = n_new*N, cols_i = n_new, cols = n_old;
    detail::HelperMatrix<double> project( rows,cols_i); 
    for( unsigned k=0; k<rows; k++)
        for( unsigned j=0; j<cols_i; j++)
        {
            project( k, j) = detail::LegendreP( j, x[k]) ;
        }
            
    detail::HelperMatrix<double> total( rows,cols);
    DLT<double> dlt = g_.dlt();
    //multiply p_ki*f_ij
    for( unsigned k=0; k<rows; k++)
        for( unsigned j=0; j<cols; j++)
            for( unsigned i=0; i<cols_i; i++)
                total( k, j) += project( k, i)*dlt.forward()[i*cols+j];
    return total;
}

cusp::coo_matrix< int, double, cusp::host_memory> diagonal_matrix( unsigned N, const detail::HelperMatrix<double>& hm)
{
    unsigned rows = hm.rows(), cols = hm.cols();
    cusp::coo_matrix<int, double, cusp::host_memory> A(N*rows, N*cols, N*rows*cols);
    unsigned number = 0;
    for( unsigned k=0; k<N; k++)
        for( unsigned i=0; i<rows; i++)
            for( unsigned j=0; j<cols; j++)
            {
                A.row_indices[number]      = k*rows+i;
                A.column_indices[number]   = k*cols+j;
                A.values[number]           = hm(i,j);
                number++;
            }
    return A;
}
}//namespace detail

cusp::coo_matrix< int, double, cusp::host_memory> projection1d( const Grid1d<double>& g1, const Grid1d<double>& g2)
{
    assert( g1.x0() == g2.x0()); assert( g1.x1() == g2.x1());
    assert( g2.N() % g1.N() == 0);
    unsigned Nf = g2.N()/g1.N();
    detail::HelperMatrix<double> p = dg::create::detail::projection( g1.n(), g2.n(), Nf);
    return dg::create::detail::diagonal_matrix( g1.N(), p);
}
cusp::coo_matrix< int, double, cusp::host_memory> projection2d( const Grid<double>& g1, const Grid<double>& g2)
{
    assert( g1.x0() == g2.x0()); assert( g1.x1() == g2.x1());
    assert( g1.y0() == g2.y0()); assert( g1.y1() == g2.y1());
    assert( g2.Nx() % g1.Nx() == 0);
    assert( g2.Ny() % g1.Ny() == 0);
    unsigned Nfx = g2.Nx()/g1.Nx();
    unsigned Nfy = g2.Ny()/g1.Ny();
    Grid1d<double> g1x( g1.x0(), g1.x1(), g1.n(), g1.Nx()); 
    Grid1d<double> g1y( g1.y0(), g1.y1(), g1.n(), g1.Ny());
    Grid1d<double> g2x( g2.x0(), g2.x1(), g2.n(), g2.Nx()); 
    Grid1d<double> g2y( g2.y0(), g2.y1(), g2.n(), g2.Ny());
    
    detail::HelperMatrix<double> px( dg::create::detail::projection( g1.n(), g2.n(), Nfx));
    detail::HelperMatrix<double> py( dg::create::detail::projection( g1.n(), g2.n(), Nfy));
    detail::HelperMatrix<double> p = kronecker( py, px); 
    return dg::create::detail::diagonal_matrix( g1.Nx()*g1.Ny(), p);
}


}//namespace create
}//namespace dg
