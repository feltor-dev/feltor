#pragma once

#include <vector>

namespace dg
{

namespace create
{

///@cond
namespace detail
{
template<class T>
void sparsify( cusp::array1d<int, cusp::host_memory>& row_offsets,
         cusp::array1d<int, cusp::host_memory>& column_indices,
         cusp::array1d<T, cusp::host_memory>& values,
         const int i,
        const thrust::host_vector<T>& zw,
        const std::vector<int>& iz_zw,
        unsigned nnzmax, T threshold)
{
    //std::cout << "Entries in zw\n";
    //for( auto idx : iz_zw)
    //    std::cout << "[ "<<idx << " "<<zw[idx]<<"]\n";

    // Take zw and write it into z modulo the drop rules
    std::vector<std::pair<double, int>> pairs;
    std::vector<std::pair<int, double>> accept;
    // 1. Always take the diagonal element
    accept.push_back( {i, zw[i]});
    for( auto idx : iz_zw)
        if( idx != i) //we already have diagonal
            pairs.push_back( { zw[idx], idx});
    //std::cout << "Pairs \n";
    //for( auto pair : pairs)
    //    std::cout << pair.first << " "<<pair.second<<std::endl;
    std::sort( pairs.begin(), pairs.end(), std::greater<>());
    //std::cout << "Pairs after sort\n";
    // 2. Take nnzmax-1 largest values of zw: sort zw by size
    // but 3. only if entry is greater than threshold
    for( int k=0; k<(int)nnzmax-1; k++)
    {
        if( k < (int)pairs.size() && fabs(pairs[k].first) > threshold)
        {
            //std::cout << pairs[k].first << " "<<pairs[k].second<<std::endl;
            accept.push_back({pairs[k].second, pairs[k].first});
        }
    }
    // sort by index
    std::sort( accept.begin(), accept.end());
    //std::cout << "Accepted entries in zw\n";
    //for( auto pair : accept)
    //    std::cout << "[ "<<pair.first << " "<<pair.second<<"]\n";


    //write into matrix
    row_offsets.push_back(row_offsets[i]);
    for( auto pair : accept)
    {
        //std::cout<< "Entry "<<pair.first<<" "<<pair.second<<"\n";
        column_indices.push_back( pair.first);
        values.push_back( pair.second);
        row_offsets[i+1]++;
    }
}
}
///@endcond

/*!
 * @brief Left looking sparse inverse preconditioner for self-adjoint positive definit matrices
 * \f[ A^{-1} = S^T D^{-1} S W \f]
 *
 * where S is a lower triangular matrix, D is a diagonal matrix and W are the weights
 * @note Implements <a href="https://doi.org/10.1016/j.camwa.2015.12.008" target="_blank">Bertaccini and Filippone "Sparse approximate inverse preconditioners on high performance GPU platforms" (2016)</a>. S is the transpose of Z with respect to the original algorithm.
 * @param a input matrix in csr format (self-adjoint in weights), must be sorted by columns
 * @param s output matrix S
 * @param d diagonal output
 * @param weights weights in which the input matrix \c a is self-adjoint
 * @param nnzmax maximum number of non-zeroes in a row in \c s
 * @param threshold absolute limit under which entries are dropped from \c s and entries in \c d are strictly greater than
 * @note Sparse inverse preconditioners can be applied directly like any other
 * sparse matrix and do not need a linear system solve like in sparse LU
 * factorization type methods
 * @attention Use \c dg::nested_iterations if you have a geometry available,
 * which is the faster method. Tests on the \c dg::Elliptic matrix show no
 * significant speedup that could justify the additional costs to build and
 * apply the preconditioner. (So we did not bother to implement an MPI version)
 * @ingroup invert
 * @tparam T real type
 */
template<class T>
void sainv_precond(
        const cusp::csr_matrix<int, T, cusp::host_memory>& a,
        cusp::csr_matrix<int, T, cusp::host_memory>& s,
        thrust::host_vector<T>& d,
        const thrust::host_vector<T>& weights,
        unsigned nnzmax,
        T threshold)
{
    unsigned n = a.num_rows;
    //assert( a.num_rows == a.num_cols);

    d.resize( n, 0.);

    // Init z_0 = e_0 and p_0 = a_{11}
    for( int j = a.row_offsets[0]; j<a.row_offsets[1]; j++)
    {
        if( a.column_indices[j] == 0)
            d[0] = a.values[j]*weights[0];
    }
    if( fabs( d[0] ) < threshold)
        d[0] = threshold;
    //std::cout << "first diagonal "<<d[0]<<"\n";
    cusp::array1d<int, cusp::host_memory> row_offsets, column_indices;
    cusp::array1d<T, cusp::host_memory> values;

    row_offsets.push_back(0);
    row_offsets.push_back(1);
    column_indices.push_back( 0);
    values.push_back( 1.0);

    // Main loop
    for( int i = 1; i<(int)n; i++)
    {
        thrust::host_vector<T> zw( n, 0.);
        std::vector<int> iz_zw; // flags nonzeros in zw
        //zw = e_i
        zw[i] = 1.0;
        iz_zw.push_back(i);
        std::vector<int> s;
        // get column indices of row i that are smaller than i
        for( int k = a.row_offsets[i]; k<a.row_offsets[i+1]; k++)
        {
            if( a.column_indices[k] < i )
                s.push_back( a.column_indices[k]);
        }
        //std::cout << "Loop i = "<<i<<"\n";
        //std::cout<<  "s\n";
        //for( auto idx : s)
        //    std::cout << idx << " \n";
        while( !s.empty())
        {
            auto it = std::min_element( s.begin(), s.end());
            int j = *it; // j < i in all cases
            s.erase( it);
            // A_j * zw
            d[i] = 0.0;
            for( int k = a.row_offsets[j]; k<a.row_offsets[j+1]; k++)
            {
                //std::cout << "Multiply k " << k<<" "<<a.column_indices[k]<<" "<< a.values[k]<<" j = "<< j << " weights "<<weights[j]<<" zw "<<zw[a.column_indices[k]]<<"\n";
                d[i] += weights[j]*a.values[k]*zw[ a.column_indices[k]];
            }
            //std::cout << "d[i] "<<d[i]<<"\n";
            T alpha = d[i]/d[j];
            //std::cout << "alpha ij "<< i <<" "<<j<<" "<<alpha<<"\n";
            if( fabs( alpha) > threshold)
            {
                for( int k = row_offsets[j]; k<row_offsets[j+1]; k++)
                {
                    int zkk = column_indices[k];
                    //std::cout << "col  = "<<zkk<<std::endl;
                    zw[ zkk] -= alpha * values[k];
                    // flag a new nonzero in zw if necessary
                    if (std::find(iz_zw.begin(), iz_zw.end(), zkk) == iz_zw.end()) {
                        iz_zw.push_back( zkk);
                    }
                    //std::cout << "col  = "<<zkk<<std::endl;
                    // add indices to set of dot products to compute
                    for( int l = a.row_offsets[zkk]; l < a.row_offsets[zkk+1]; l++)
                    {
                        //std::cout << "l  = "<<l<<" "<<a.row_offsets[zkk]<< " "<<a.row_offsets[zkk+1]<<std::endl;
                        if ( (std::find(s.begin(), s.end(), l) == s.end()) && (j<l) && (l < i) ) {
                            s.push_back( l);
                        }
                    }
                    //std::cout << "col  = "<<zkk<<std::endl;
                }
            }
        }
        d[i] = 0.0;
        for( int k = a.row_offsets[i]; k<a.row_offsets[i+1]; k++)
            d[i] += a.values[k]*zw[ a.column_indices[k]]*weights[i];
        if( fabs(d[i]) < threshold)
            d[i] = threshold;
        //std::cout << "d[i] "<<d[i]<<"\n";
        //std::cout << "zw \n";
        //for(unsigned i=0; i<zw.size(); i++)
        //    std::cout << zw[i]<<"\n";
        // Apply drop rule to zw:
        detail::sparsify( row_offsets, column_indices, values, i, zw, iz_zw, nnzmax, threshold);
    }
    s.resize( n, n, values.size());
    s.column_indices = column_indices;
    s.row_offsets = row_offsets;
    s.values = values;


}
} //namespace create

}//namespace dg
