#pragma once

#include "blas1_dispatch_map.h"
#include "matrix_categories.h"

///@cond
namespace dg
{
namespace blas2
{
//forward declare blas2 symv functions
template< class MatrixType, class ContainerType1, class ContainerType2>
void symv( MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y);
template< class MatrixType, class ContainerType1, class ContainerType2>
void symv( get_value_type<ContainerType1> alpha,
                  MatrixType&& M,
                  const ContainerType1& x,
                  get_value_type<ContainerType1> beta,
                  ContainerType2& y);
namespace detail
{

template< class Vector1, class Matrix, class Vector2>
inline std::vector<int64_t> doDot_superacc( int* status, const Vector1& x,
    const Matrix& m, const Vector2& y, RecursiveVectorTag, RecursiveVectorTag)
{
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
    std::vector<int64_t> acc( exblas::BIN_COUNT, (int64_t)0);
    int i=0;
    for( auto el : get_idx<vector_idx>(x1,x2))
    {
        try{
        std::vector<int64_t> temp = doDot_superacc( status,
            do_get_vector_element(x,i,get_tensor_category<Vector1>()), m[i],
            do_get_vector_element(y,i,get_tensor_category<Vector2>()));
        int imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(temp[0]), imin, imax);
        for( int k=exblas::IMIN; k<=exblas::IMAX; k++)
            acc[k] += temp[k];
        if( (i+1)%128 == 0)
        {
            imin = exblas::IMIN, imax = exblas::IMAX;
            exblas::cpu::Normalize( &(acc[0]), imin, imax);
        }
        i++;
        }catch( std::out_of_range& err)
        {
            throw dg::Error( dg::Message(_ping_)<<"Wrong key '"<<el.first<<"' in blas1::dot "<<err.what());
        }
    }
    return acc;
}


} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond
