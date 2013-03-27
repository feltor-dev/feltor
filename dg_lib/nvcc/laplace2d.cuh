#ifndef _DG_LAPLACE2D_CUH
#define _DG_LAPLACE2D_CUH

#include <cassert>
#include <cusp/coo_matrix.h>

#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>

#include "preconditioner.cuh"

namespace dg
{

namespace create{

namespace detail{

struct AddIndex2d{
    AddIndex2d( size_t M, size_t n, size_t m):M(M), n(n), m(m), number(0) {}
    void operator() ( cusp::array1d< int, cusp::host_memory>& Idx, 
                      unsigned i, unsigned j, unsigned k, unsigned l)
    {
        Idx[ number] = M*n*m*i + n*m*j + m*k + l;
        number ++;
    }
    void operator() ( cusp::array1d< double, cusp::host_memory>& Val, double value)
    {
        Val[ number] = value;
        number ++;
    }
  private:
    size_t M, n, m;
    unsigned number;
};

} //namespace detail

/**
 * @brief Create 2D Tensor by summing 2 diagonal Tensor products
 *
 * Compute the sum LHS x D1 + D2 x RHS, where x denotes a tensor
 * product, LHS denotes the matrix for the first (y-) index and RHS     
 * the one for the second (x-) index. D1 and D2 are diagonal matrices. 
 * @tparam n The number of Legendre coefficients per dimension
 * @param lhs Matrix for first index ( e.g. a Laplacian)
 * @param D1  Diagonal matrix for second index
 * @param D2  Diagonal matrix for first index
 * @param rhs Matrix for second index (e.g. another Laplacian)
 *
 * @return The assembled and sorted tensor matrix
 */
template< size_t n >
cusp::coo_matrix< int, double, cusp::host_memory> tensorSum( 
        const cusp::coo_matrix< int, double, cusp::host_memory>& lhs,
        const dg::S1D<double, n>& D1, 
        const dg::S1D<double, n>& D2, 
        const cusp::coo_matrix< int, double, cusp::host_memory>& rhs)
{
    //assert quadratic matrices
    assert( lhs.num_rows == lhs.num_cols);
    assert( rhs.num_rows == rhs.num_cols);
    //assert dg matrices
    assert( lhs.num_rows%n == 0);
    assert( rhs.num_rows%n == 0);
    unsigned Nx = rhs.num_rows/n; 
    unsigned Ny = lhs.num_rows/n; 
    //taken from the cusp examples:
    //dimensions of the matrix
    int num_cols = lhs.num_rows*rhs.num_rows, num_rows( num_cols);
    int num_triplets    = lhs.values.size()*rhs.num_rows + lhs.num_rows*rhs.values.size();
    // allocate storage for unordered triplets
    cusp::array1d< int,     cusp::host_memory> I( num_triplets); // row indices
    cusp::array1d< int,     cusp::host_memory> J( num_triplets); // column indices
    cusp::array1d< double,  cusp::host_memory> V( num_triplets); // values
    //fill triplet arrays
    detail::AddIndex2d addIndexRow( Nx, n,n );
    detail::AddIndex2d addIndexCol( Nx, n,n );
    detail::AddIndex2d addIndexVal( Nx, n,n );
    //First D2 x RHS
    for( unsigned i=0; i<Ny; i++)
        for( unsigned k=0; k<n; k++)
            for( unsigned j=0; j<rhs.num_entries; j++)
            {
                addIndexRow( I, i, rhs.row_indices[j]/n, k, rhs.row_indices[j]%n);
                addIndexCol( J, i, rhs.column_indices[j]/n, k, rhs.column_indices[j]%n);
                addIndexVal( V, D2(i*n+k)*rhs.values[j]);
            }
    //Second LHS x D1
    for( unsigned i=0; i<Nx; i++)
        for( unsigned k=0; k<n; k++)
            for( unsigned j=0; j<lhs.num_entries; j++)
            {
                addIndexRow( I, lhs.row_indices[j]/n,i, lhs.row_indices[j]%n, k);
                addIndexCol( J, lhs.column_indices[j]/n,i, lhs.column_indices[j]%n, k);
                addIndexVal( V, lhs.values[j]*D1(i*n+k) );
            }
    cusp::array1d< int,     cusp::device_memory> dI( I); // row indices
    cusp::array1d< int,     cusp::device_memory> dJ( J); // column indices
    cusp::array1d< double,  cusp::device_memory> dV( V); // values
#ifdef DG_DEBUG
    std::cout << "Values ready! Now sort...\n";
#endif //DG_DEBUG
    // sort triplets by (i,j) index using two stable sorts (first by J, then by I)
    thrust::stable_sort_by_key(dJ.begin(), dJ.end(), thrust::make_zip_iterator(thrust::make_tuple(dI.begin(), dV.begin())));
    thrust::stable_sort_by_key(dI.begin(), dI.end(), thrust::make_zip_iterator(thrust::make_tuple(dJ.begin(), dV.begin())));
#ifdef DG_DEBUG
    std::cout << "Sort ready! Now compute unique number of nonzeros ...\n";
#endif //DG_DEBUG
    // compute unique number of nonzeros in the output
    int num_entries = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(dI.begin(), dJ.begin())),
                                            thrust::make_zip_iterator(thrust::make_tuple(dI.end (),  dJ.end()))   - 1,
                                            thrust::make_zip_iterator(thrust::make_tuple(dI.begin(), dJ.begin())) + 1,
                                            int(0),
                                            thrust::plus<int>(),
                                            thrust::not_equal_to< thrust::tuple<int,int> >()) + 1;

    // allocate output matrix
    cusp::coo_matrix<int, double, cusp::device_memory> A(num_rows, num_cols, num_entries);
#ifdef DG_DEBUG
    std::cout << "Computation ready! Now reduce to unique number of nonzeros ...\n";
#endif //DG_DEBUG
    // sum values with the same (i,j) index
    thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(dI.begin(), dJ.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(dI.end(),   dJ.end())),
                          dV.begin(),
                          thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
                          A.values.begin(),
                          thrust::equal_to< thrust::tuple<int,int> >(),
                          thrust::plus<double>());
    return A;
}


} //namespace create

} //namespace dg

#include "blas/thrust_vector.cuh"
#include "blas/laplace.cuh"

#endif // _DG_LAPLACE2D_CUH

