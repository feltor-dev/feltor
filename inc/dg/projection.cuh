#include <vector>
#include <cusp/coo_matrix.h>
#include "grid.cuh"
#include "xspacelib.cuh"
#include "matrix_traits_thrust.h"

namespace dg{

namespace create{
namespace detail{

template<class T>
struct HelperMatrix
{
    HelperMatrix( unsigned m, unsigned n):rows_(m), cols_(n), data_(m*n){}
    HelperMatrix( unsigned m, unsigned n, T value):rows_(m), cols_(n), data_(m*n, value){}
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
    unsigned rows = n_new*N, cols_i = n_new;
    detail::HelperMatrix<double> project( rows, cols_i); 
    for( unsigned k=0; k<rows; k++)
        for( unsigned j=0; j<cols_i; j++)
        {
            project( k, j) = detail::LegendreP( j, x[k]) ;
        }
    detail::HelperMatrix<double> total( rows, n_old, 0.);
    dg::Grid1d<double> g2( -1, 1, n_old, 1);
    DLT<double> dlt = g2.dlt();
    //multiply p_ki*f_ij
    for( unsigned k=0; k<rows; k++)
        for( unsigned j=0; j<n_old; j++)
            for( unsigned i=0; i<cols_i; i++)
                total( k, j) += project( k, i)*dlt.forward()[i*n_old+j];

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
/**
 * @brief Create a 2D projection matrix onto a finer grid
 *
 * Grid space must be equal. Nx and Ny of the second grid must be multiples of 
 * Nx and Ny of the first grid.
 * @param g1 Grid of the original vector
 * @param g2 Grid of the target vector
 *
 * @return Projection matrix
 */
cusp::coo_matrix< int, double, cusp::host_memory> projection2d( const Grid<double>& g1, const Grid<double>& g2)
{
    //TODO: projection in y direction needs permutation
    assert( g1.x0() == g2.x0()); assert( g1.x1() == g2.x1());
    assert( g1.y0() == g2.y0()); assert( g1.y1() == g2.y1());
    assert( g2.Nx() % g1.Nx() == 0);
    assert( g2.Ny() % g1.Ny() == 0);
    unsigned Nfx = g2.Nx()/g1.Nx();
    unsigned Nfy = g2.Ny()/g1.Ny();
    
    typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix;
    thrust::host_vector<int> map1 = dg::create::scatterMap( g2.n()*Nfx, g2.n()*Nfy, g1.Nx(), g1.Ny()); //map to permute continuous in one cell to continuous in all cells
    thrust::host_vector<int> map2 = dg::create::scatterMap( g2.n(), g2.Nx(), g2.Ny());//map to permute contiguous to new grid
    Matrix perm1 = dg::create::scatter( map1); 
    Matrix perm2 = dg::create::gather( map2); 
    Matrix perm;
    //permutation
    cusp::multiply( perm2, perm1, perm);

    detail::HelperMatrix<double> px( dg::create::detail::projection( g1.n(), g2.n(), Nfx));
    detail::HelperMatrix<double> py( dg::create::detail::projection( g1.n(), g2.n(), Nfy));
    detail::HelperMatrix<double> p = kronecker( py, px); 
    //copy p to cusp matrix
    Matrix project = dg::create::detail::diagonal_matrix( g1.Nx()*g1.Ny(), p);
    cusp::multiply( perm, project, perm1);

    /*
    //copy C to a HelperMatrix and create matrix for all cells
    detail::HelperMatrix<double> pp( project.num_rows, project.num_cols, 0.); 
    for( unsigned i=0; i<project.num_entries; i++)
        pp(project.row_indices[i], project.column_indices[i]) = project.values[i];
    project = dg::create::detail::diagonal_matrix( g1.Nx()*g1.Ny(), pp);
    */

    return perm1;
}


}//namespace create

unsigned gcd( unsigned a, unsigned b)
{
    unsigned r2 = std::max(a,b);
    unsigned r1 = std::min(a,b);
    while( r1!=0)
    {
        r2 = r2%r1;
        std::swap( r1, r2);
    }
    return r2;
}
unsigned lcm( unsigned a, unsigned b)
{
    unsigned g = gcd( a,b);
    return a/g*b;
}

//eventuell könnte man zwei Projektionsmatrizen malnehmen um eine kleinere zu erhalten
template <typename container>
struct DifferenceNorm
{
    typedef typename container::value_type value_type;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    typedef cusp::csr_matrix<int, double, MemorySpace> Matrix;
    DifferenceNorm( const Grid<double>& g1, const Grid<double>& g2)
    {
        //find common grid
        Grid<double> gC(    g1.x0(), g1.x1(), g1.y0(), g1.y1(), 
                            std::min( g1.n(), g2.n()), 
                            lcm( g1.Nx(), g2.Nx()), 
                            lcm( g1.Ny(), g2.Ny()) );
        p1 = dg::create::projection2d( g1, gC);
        p2 = dg::create::projection2d( g2, gC);
        w2d = dg::create::w2d( gC); v11 = w2d, v22 = w2d;
        wg1 = dg::create::w2d( g1); 
        wg2 = dg::create::w2d( g2); 
    }
    double operator()( const container& v1, const container& v2)
    {
        double f2, g2, fg;
        f2 = blas2::dot( wg1, v1);
        g2 = blas2::dot( wg2, v2);

        blas2::gemv( p1, v1, v11);
        blas2::gemv( p2, v2, v22);
        fg = blas2::dot( v11, w2d, v22);
        return sqrt( f2 - 2.*fg + g2);
    }
    double sum( const container& v1, const container& v2)
    {
        double f2, g2, fg;
        f2 = blas2::dot( wg1, v1);
        g2 = blas2::dot( wg2, v2);

        blas2::gemv( p1, v1, v11);
        blas2::gemv( p2, v2, v22);
        fg = blas2::dot( v11, w2d, v22);
        return sqrt( f2 + 2.*fg + g2);
    }
  private:
    container wg1, wg2;
    container w2d, v11, v22;
    Matrix p1, p2;
};



}//namespace dg
