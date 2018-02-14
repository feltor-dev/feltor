#pragma once

#include <cassert>

#include <cusp/coo_matrix.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>



/**@file
* @brief a tensor product function between two cusp matrices
*/
namespace dg
{


///@cond
namespace detail{

struct AddIndex2d{
    AddIndex2d( size_t M, size_t n, size_t m):M(M), n(n), m(m), number(0) {}

    void operator() ( cusp::array1d< int, cusp::host_memory>& Idx, 
                      unsigned i, unsigned j, unsigned k, unsigned l)
    {
        //choose vector layout, old or new
        //Idx[ number] = M*n*m*i + n*m*j + m*k + l;
        Idx[ number] = M*m*(i*n+k) + m*j + l;
        number ++;
    }
    template<class T>
    void operator() ( cusp::array1d< T, cusp::host_memory>& Val, T value)
    {
        Val[ number] = value;
        number ++;
    }
  private:
    size_t M, n, m;
    unsigned number;
};

} //namespace detail
///@endcond

/**
* @brief Form the DG tensor product between two DG matrices
*
* @ingroup lowlevel
* Takes care of correct permutation of indices.
* @tparam T value type
* @param n # of Legendre coefficients per dimension
* @param lhs The left hand side (1D )
* @param rhs The right hand side (1D ) 
*
* @return A newly allocated cusp matrix containing the tensor product
* @note use cusp::add and cups::multiply to add and multiply matrices
*/
template< class T>
cusp::coo_matrix< int, T, cusp::host_memory> dgtensor( unsigned n,
        const cusp::coo_matrix< int, T, cusp::host_memory>& lhs,
        const cusp::coo_matrix< int, T, cusp::host_memory>& rhs)
{    
    //assert quadratic matrices
    assert( lhs.num_rows == lhs.num_cols);
    assert( rhs.num_rows == rhs.num_cols);
    //assert dg matrices
    assert( lhs.num_rows%n == 0);
    assert( rhs.num_rows%n == 0);
    unsigned Nx = rhs.num_rows/n; 
    //unsigned Ny = lhs.num_rows/n; 
    //taken from the cusp examples:
    //dimensions of the matrix
    int num_rows     = lhs.num_rows*rhs.num_rows;
    int num_cols     = num_rows;
    int num_triplets = lhs.num_entries*rhs.num_entries;
    // allocate storage for unordered triplets
    cusp::array1d< int,     cusp::host_memory> I( num_triplets); // row indices
    cusp::array1d< int,     cusp::host_memory> J( num_triplets); // column indices
    cusp::array1d< T,  cusp::host_memory> V( num_triplets); // values
    //fill triplet arrays
    detail::AddIndex2d addIndexRow( Nx, n,n );
    detail::AddIndex2d addIndexCol( Nx, n,n );
    detail::AddIndex2d addIndexVal( Nx, n,n );
    //LHS x RHS
    for( unsigned i=0; i<lhs.num_entries; i++)
        for( unsigned j=0; j<rhs.num_entries; j++)
        {
            addIndexRow( I, lhs.row_indices[i]/n, rhs.row_indices[j]/n, lhs.row_indices[i]%n, rhs.row_indices[j]%n);
            addIndexCol( J, lhs.column_indices[i]/n, rhs.column_indices[j]/n, lhs.column_indices[i]%n, rhs.column_indices[j]%n);
            addIndexVal( V, lhs.values[i]*rhs.values[j]);
        }
    cusp::array1d< int, cusp::host_memory> dI( I); // row indices
    cusp::array1d< int, cusp::host_memory> dJ( J); // column indices
    cusp::array1d< T,   cusp::host_memory> dV( V); // values
#ifdef DG_DEBUG
    //std::cout << "Values ready! Now sort...\n";
#endif //DG_DEBUG
    // sort triplets by (i,j) index using two stable sorts (first by J, then by I)
    thrust::stable_sort_by_key(dJ.begin(), dJ.end(), thrust::make_zip_iterator(thrust::make_tuple(dI.begin(), dV.begin())));
    thrust::stable_sort_by_key(dI.begin(), dI.end(), thrust::make_zip_iterator(thrust::make_tuple(dJ.begin(), dV.begin())));
#ifdef DG_DEBUG
    //std::cout << "Sort ready! Now compute unique number of values with different (i,j) index ...\n";
#endif //DG_DEBUG
    // compute unique number of ( values with different (i,j) index)  in the output
    int num_entries = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(dI.begin(), dJ.begin())),
                                            thrust::make_zip_iterator(thrust::make_tuple(dI.end (),  dJ.end()))   - 1,
                                            thrust::make_zip_iterator(thrust::make_tuple(dI.begin(), dJ.begin())) + 1,
                                            int(0),
                                            thrust::plus<int>(),
                                            thrust::not_equal_to< thrust::tuple<int,int> >()) + 1;

    // allocate output matrix
    cusp::coo_matrix<int, T, cusp::host_memory> A(num_rows, num_cols, num_entries);
#ifdef DG_DEBUG
    //std::cout << "Computation ready! Now sum values with same (i,j) index ...\n";
#endif //DG_DEBUG
    // sum values with the same (i,j) index
    thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(dI.begin(), dJ.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(dI.end(),   dJ.end())),
                          dV.begin(),
                          thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
                          A.values.begin(),
                          thrust::equal_to< thrust::tuple<int,int> >(),
                          thrust::plus<T>());
    return A; 
}


} //namespace dg


