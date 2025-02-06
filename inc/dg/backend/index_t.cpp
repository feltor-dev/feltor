
#include <iostream>
#include "index.h"
#include "catch2/catch_all.hpp"

TEST_CASE( "Test index lib")
{
    thrust::host_vector<std::array<int,2>> gIdx = std::vector<std::array<int,2>>{
    {1,0}, {2,6}, {5,6}, {0,2}, {0,1}, {2,1}, {2,3}, {1,1}, {5,6}, {1,0}, {5,0}, {0,2}};
    INFO("Values \n");
    for( unsigned u=0; u<gIdx.size(); u++)
        INFO("{"<<gIdx[u][0]<<","<<gIdx[u][1]<<"} ");
    INFO("\n\n");
    SECTION( "Test gIdx2unique_idx\n")
    {
        thrust::host_vector<int> bufferIdx;
        auto recv_map = dg::gIdx2unique_idx( gIdx, bufferIdx);

        for ( auto& idx : recv_map)
        {
            INFO( "Receive ");
            for( unsigned u=0; u<idx.second.size(); u++)
                INFO("{"<<idx.first<<","<<idx.second[u]<<"} ");
            INFO("\n");
        }
        thrust::host_vector<int> ana = std::vector{ 2, 4, 7, 0, 1, 5, 6, 3, 7, 2, 8, 0};
        CHECK( ana == bufferIdx);
    }

    auto test = GENERATE( 0,1);
    dg::detail::Unique<std::array<int,2>> uni;
    if( test == 0)
    {
        INFO("Order preserving TEST\n");
        uni = dg::detail::find_unique_order_preserving(gIdx);
    }
    else
    {
        INFO("Stable sort TEST\n");
        uni = dg::detail::find_unique_stable_sort( gIdx);
    }
    auto sortedgIdx = gIdx;
    thrust::scatter( gIdx.begin(), gIdx.end(),
                 uni.gather1.begin(), sortedgIdx.begin());
    INFO("Sorted indices \n");
    for( unsigned u=0; u<sortedgIdx.size(); u++)
        INFO("{"<<sortedgIdx[u][0]<<","<<sortedgIdx[u][1]<<"} ");
    INFO("Unique values \n");
    for( unsigned u=0; u<uni.unique.size(); u++)
        INFO("{"<<uni.unique[u][0]<<","<<uni.unique[u][1]<<"} ");
    INFO("Howmany Unique values \n");
    for( unsigned u=0; u<uni.unique.size(); u++)
        INFO(uni.howmany[u]<<" ");
    auto num = gIdx; // consistency test
    thrust::gather( uni.gather2.begin(), uni.gather2.end(),
                uni.unique.begin(), num.begin());
    INFO("Sorted indices \n");
    for( unsigned u=0; u<sortedgIdx.size(); u++)
    {
        INFO("{"<<num[u][0]<<","<<num[u][1]<<"} ");
        CHECK( num[u] == sortedgIdx[u]);
    }

    num = gIdx; // consistency test
    auto sort_map = uni.gather1;
    thrust::gather( uni.gather1.begin(), uni.gather1.end(),
                uni.gather2.begin(), sort_map.begin());
    thrust::gather( sort_map.begin(), sort_map.end(),
                uni.unique.begin(), num.begin());
    for( unsigned u=0; u<gIdx.size(); u++)
        CHECK( num[u] == gIdx[u]);
}
TEST_CASE("Test contiguous range function\n")
{
    thrust::host_vector<int> range = std::vector{4,5,6,1,3,4,5};
    auto chunks = dg::detail::find_contiguous_chunks( range);
    for( auto& chunk : chunks)
        INFO(chunk.idx<<" "<<chunk.size<<"\n");
    CHECK( chunks[0].idx == 4); CHECK( chunks[0].size == 3);
    CHECK( chunks[1].idx == 1); CHECK( chunks[1].size == 1);
    CHECK( chunks[2].idx == 3); CHECK( chunks[2].size == 3);

}
TEST_CASE("Test some map functionality\n")
{
    std::map< int, thrust::host_vector<int> > send = {{0, std::vector{1,2,3,4}},{3, std::vector{5,6}}};
    thrust::host_vector<int> ana  = std::vector{1,2,3,4,5,6};
    auto flat = dg::detail::flatten_values( send);
    CHECK( flat == send);
    auto flatten = dg::detail::flatten_map( flat);
    CHECK( flatten == ana);

}
