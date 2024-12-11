
#include <iostream>
#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>
#include <cusp/convert.h>
#include <cusp/array1d.h>
#include "../blas.h" // to include all tensor traits
#include "gather.h"


int main()
{
    thrust::host_vector<std::array<int,2>> gIdx = std::vector<std::array<int,2>>{
    {1,0}, {2,6}, {5,6}, {0,2}, {0,1}, {2,1}, {2,3}, {1,1}, {5,6}, {1,0}, {5,0}, {0,2}};
    std::cout<< "Values \n";
    for( unsigned u=0; u<gIdx.size(); u++)
        std::cout <<"{"<<gIdx[u][0]<<","<<gIdx[u][1]<<"} ";
    std::cout<<"\n"<<std::endl;
    for( unsigned test=0; test<2; test++)
    {
        dg::detail::Unique<std::array<int,2>> uni;
        if( test == 0)
        {
            std::cout << "Order preserving TEST\n";
            uni = dg::detail::find_unique_order_preserving(gIdx);
        }
        else
        {
            std::cout << "Stable sort TEST\n";
            uni = dg::detail::find_unique_stable_sort( gIdx);
        }
        auto sortedgIdx = gIdx;
        thrust::scatter( gIdx.begin(), gIdx.end(),
                     uni.gather1.begin(), sortedgIdx.begin());
        std::cout<< "Sorted indices \n";
        for( unsigned u=0; u<sortedgIdx.size(); u++)
            std::cout <<"{"<<sortedgIdx[u][0]<<","<<sortedgIdx[u][1]<<"} ";
        std::cout<<std::endl;
        std::cout<< "Unique values \n";
        for( unsigned u=0; u<uni.unique.size(); u++)
            std::cout <<"{"<<uni.unique[u][0]<<","<<uni.unique[u][1]<<"} ";
        std::cout<<std::endl;
        std::cout<< "Howmany Unique values \n";
        for( unsigned u=0; u<uni.unique.size(); u++)
            std::cout <<uni.howmany[u]<<" ";
        std::cout<<std::endl;
        auto num = gIdx; // consistency test
        thrust::gather( uni.gather2.begin(), uni.gather2.end(),
                    uni.unique.begin(), num.begin());
        std::cout<< "Sorted Unique values \n";
        for( unsigned u=0; u<sortedgIdx.size(); u++)
            std::cout <<"{"<<num[u][0]<<","<<num[u][1]<<"} ";
        std::cout<<std::endl;

        num = gIdx; // consistency test
        auto sort_map = uni.gather1;
        thrust::gather( uni.gather1.begin(), uni.gather1.end(),
                    uni.gather2.begin(), sort_map.begin());
        thrust::gather( sort_map.begin(), sort_map.end(),
                    uni.unique.begin(), num.begin());
        for( unsigned u=0; u<gIdx.size(); u++)
            assert( num[u] == gIdx[u]);
        std::cout << "Gather PASSED\n\n";
    }

    std::cout << "Test gIdx2unique_idx\n";
    thrust::host_vector<int> bufferIdx;
    auto recv_map = dg::detail::gIdx2unique_idx( gIdx, bufferIdx);

    for ( auto& idx : recv_map)
    {
        std::cout << "Receive ";
        for( unsigned u=0; u<idx.second.size(); u++)
            std::cout << "{"<<idx.first<<","<<idx.second[u]<<"} ";
        std::cout << "\n";
    }

    std::cout << "Test Local Gather Matrix\n";
    std::vector<int> idx = {7,4,5,2,4,9};

    dg::LocalGatherMatrix<thrust::host_vector> gather(idx);
    std::vector<int> values = {0,1,2,3,4,5,6,7,8,9};
    std::vector<int> buffer(idx.size());
    gather.gather( 1., values, 0., buffer);
    for( unsigned i=0; i<idx.size(); i++)
        assert( buffer[i] == idx[i]);
    std::cout << "Gather PASSED\n";
    // We test scatter by comparing to cusp::coo_matrix vector multiplication

    cusp::coo_matrix<int, int, cusp::host_memory> A(  values.size(), idx.size(), idx.size());
    for( unsigned i=0; i<idx.size(); i++)
    {
        A.row_indices[i] = idx[i];
        A.column_indices[i] = i;
        A.values[i] = 1;
    }
    A.sort_by_row_and_column();
    std::vector<int> cusp_result( values.size());
    cusp::array1d_view<typename std::vector<int>::const_iterator> cx(
        buffer.cbegin(), buffer.cend());
    cusp::array1d_view<typename std::vector<int>::iterator> cy( cusp_result.begin(),
        cusp_result.end());
    cusp::multiply( A, cx, cy);

    std::vector<int> feltor_result( values);
    gather.scatter_plus( buffer, feltor_result);
    for( unsigned i=0; i<values.size(); i++)
    {
        std::cout << i<<" "<<values[i] + cusp_result[i] <<" "<< feltor_result[i]<<"\n";

        assert( values[i] + cusp_result[i] == feltor_result[i]);
    }
    std::cout << "Scatter reduce PASSED\n";

    return 0;
}
