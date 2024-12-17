
#include <iostream>
#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>
#include <cusp/convert.h>
#include <cusp/array1d.h>
#include "index.h"


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
        std::cout<< "Sorted indices \n";
        for( unsigned u=0; u<sortedgIdx.size(); u++)
        {
            std::cout <<"{"<<num[u][0]<<","<<num[u][1]<<"} ";
            assert( num[u] == sortedgIdx[u]);
        }
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
    {
    std::cout << "Test gIdx2unique_idx\n";
    thrust::host_vector<int> bufferIdx;
    auto recv_map = dg::gIdx2unique_idx( gIdx, bufferIdx);

    for ( auto& idx : recv_map)
    {
        std::cout << "Receive ";
        for( unsigned u=0; u<idx.second.size(); u++)
            std::cout << "{"<<idx.first<<","<<idx.second[u]<<"} ";
        std::cout << "\n";
    }
    thrust::host_vector<int> ana = std::vector{ 2, 4, 7, 0, 1, 5, 6, 3, 7, 2, 8, 0};
    std::cout << "PASSED\n";
    assert( ana == bufferIdx);
    }
    {
        std::cout << "Test contiguous range function\n";
        thrust::host_vector<int> range = std::vector{4,5,6,1,3,4,5};
        auto chunks = dg::detail::find_contiguous_chunks( range);
        for( auto& chunk : chunks)
            std::cout << chunk.idx<<" "<<chunk.size<<"\n";
        assert( chunks[0].idx == 4); assert( chunks[0].size == 3);
        assert( chunks[1].idx == 1); assert( chunks[1].size == 1);
        assert( chunks[2].idx == 3); assert( chunks[2].size == 3);
        std::cout << "PASSED\n";

    }
    {
        std::cout << "Test some map functionality\n";
        std::map< int, thrust::host_vector<int> > send = {{0, std::vector{1,2,3,4}},{3, std::vector{5,6}}};
        thrust::host_vector<int> ana  = std::vector{1,2,3,4,5,6};
        auto flat = dg::detail::flatten_values( send);
        assert( flat == send);
        auto flatten = dg::detail::flatten_map( flat);
        assert( flatten == ana);
        std::cout << "PASSED\n";

    }

    return 0;
}
