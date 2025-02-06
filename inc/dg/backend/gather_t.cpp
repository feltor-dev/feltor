
#include <iostream>
#include "gather.h"
#include "catch2/catch_all.hpp"

TEST_CASE( "Test Gather matrix")
{
    INFO("Test Local Gather Matrix\n");
    std::vector<int> idx = {7,4,5,2,4,9};

    dg::LocalGatherMatrix<thrust::host_vector> gather(idx);
    std::vector<int> values = {0,1,2,3,4,5,6,7,8,9};
    std::vector<int> buffer(idx.size());
    gather.gather( 1., values, 0., buffer);
    for( unsigned i=0; i<idx.size(); i++)
        CHECK( buffer[i] == idx[i]);
}
