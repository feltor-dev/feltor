#include "dx_matrix.cuh"

/***********************************************************************************
 *
 * Implementation of matrix vector multiplication of thrust_vectors and dx_matrix
 *
 ***********************************************************************************/


namespace dg
{
namespace blas2
{
namespace detail
{

template < class Matrix, class Vector1, class Vector2>
void doSymv(Matrix& mat, Vector1& x, Vector2& y, dx_matrixTag, ThrustVectorTag, ThrustVectorTag)
{
    typedef thrust::host_vector<double>::iterator ElementIterator;
    typedef thrust::host_vector<int>::iterator IndexIterator;
        // Here comes the implementation of the matrix-vector dot product
    dx_matrix_row_val rowval(mat, 0);
    dx_matrix_row_idx rowidx(mat, 0);
    double ip = 0.0;

    Vector1 tmp(y);
    for(int i = 0; i < mat.get_N() * mat.get_P(); i++)
    {
        rowval.update_row(mat, i);
        rowidx.update_row(mat, i);

        //thrust::permutation_iterator<ElementIterator, IndexIterator> x_it(x.begin(), rowidx.get_data().begin());
        //thrust::permutation_iterator<ElementIterator, IndexIterator> x_it_end(x.end(), rowidx.get_data().end());
        // 
        //std::cout << "********************doSymv************************" << std::endl;
        //thrust::host_vector<double>::iterator a_it = rowval.get_data().begin();
        //for(x_it = thrust::make_permutation_iterator(x.begin(), rowidx.get_data().begin()); x_it != x_it_end; x_it++)
        //{
        //    std::cout << *x_it << " *  " << *a_it++ << "  +  ";
        //}

        // Try out a cool scalar product
        ip = thrust::inner_product(rowval.get_data().begin(), rowval.get_data().end(),
                                   thrust::make_permutation_iterator(x.begin(), rowidx.get_data().begin()),
                                   0.0);
        //std::cout << " = "  << ip << std::endl;
        tmp[i] = ip;
    }
    y = tmp;
}

} //namespace detail
} //namespace blas2
} //namespace dg

