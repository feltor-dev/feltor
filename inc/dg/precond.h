#pragma once

#include <vector>

namespace dg
{

namespace create
{

/*!
 * @brief Left looking sparse inverse preconditioner for self-adjoint positive definit matrices
 * \f[ A^{-1} = Z D^{-1} Z^T W \f]
 * @param a input matrix in csr format (self-adjoint in weights), must be sorted by columns
 * @param z output matrix Z
 * @param d diagonal output
 * @param weights in which the input matrix \c a is self-adjoint
 * @param nnzmax number of non-zeroes per row in \c z
 * @param threshold absolute limit under which entries are dropped from \c z and entires in \c d are strictly greater than
 * @note Sparse inverse preconditioners can be applied directly like any other
 * sparse matrix and do not need a linear system solve like in sparse LU
 * factorization type methods
 */
void sainv_precond( const Matrix& a, Matrix& z, Vector& d, const Vector& weights, unsigned nnzmax, T threshold)
{
    unsigned n = a.num_rows;
    //assert( a.num_rows == a.num_cols);

    d.resize( n, 0.);

    // Init z_0 = e_0 and p_0 = a_{11}
    for( int j = a.row_offsets[0]; j<a.row_offsets[1]; j++)
    {
        if( a.column_indices[j] == 0)
            d[0] = a.values[j];
    }
    if( fabs( d[0] ) < threshold)
        d[0] = threshold;

    z.resize( n, n, n*nnzmax);
    z.row_offsets[0] = 0;
    z.row_offsets[1] = nnzmax;
    //artificially fill in explicit zeroes
    for( unsigned k=0; k<nnzmax; k++)
    {
        z.column_indices[k] = k;
        z.values[k] = k==0 ? 1.0 : 0.0;
    }

    // Main loop
    for( unsigned i = 1; i<n; i++)
    {
        Vector zval( n, 0.);
        std::vector<int> iz_zval; // flags nonzeros in zval
        //zw = e_i
        zval[i] = 1.0;
        iz_zval.push_back(i);
        std::vector<int> s;
        // get column indices of row i that are smaller than i
        for( unsigned k = a.row_offsets[i]; k<a.row_offsets[i+1]; k++)
            if( k < i )
                s.push_back( a.column_indices[k]);
        while( !s.empty())
        {
            auto it = std::min_element( s.begin(), s.end());
            int j = *it; // j < i in all cases
            s.erase( it);
            // A_j * zval
            for( unsigned k = a.row_offsets[j]; k<a.row_offsets[j+1]; k++)
                d[i] += a.values[k]*zval[ a.column_indices[k]];
            T alpha = d[i]/d[j];
            if( fabs( alpha) > threshold)
            {
                for( k = z.row_offsets[j]; k<z.row_offsets[j+1]; k++)
                {
                    unsigned zkk = z.column_indices[k];
                    zval[ zkk] -= alpha * z.values[k];
                    // flag a new nonzero in zval if necessary
                    if (std::find(iz_zval.begin(), iz_zval.end(), zkk) == iz_zval.end()) {
                        iz_zval.push_back( zkk);
                    }
                    // add indices to set of dot products to compute
                    for( l = a.row_offsets[k]; l < a.row_offsets[k+1]; l++)
                    {
                        if ( (std::find(s.begin(), s.end(), l) == s.end()) && (j<l) && (l < i) ) {
                            s.push_back( l);
                        }
                    }
                }
            }
        }
        for( unsigned k = a.row_offsets[i]; k<a.row_offsets[i+1]; k++)
            d[i] += a.values[k]*zval[ a.column_indices[k]];
        if( fabs(d[i]) < threshold)
            d[i] = threshold;
        // Apply drop rule to zval:
        // 1. sort zvalues by size
        std::vector<std::pair<double, unsigned>> pairs;
        for( unsigned k=0; k< iz_zval.size(); k++)
            pairs.push_back( { zval[iz_zval[k]], k});
        std::sort( pairs.begin(), pairs.end(), std::greater<>());
        // insert the first nnzmax values into z
        z.row_offsets[i+1] = z.row_offsets[i] + nnzmax;
        for( unsigned k=0; k<nnzmax; k++)
        {
        }



    }


}
} //namespace create

}//namespace dg
