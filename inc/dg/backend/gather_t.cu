
#include <iostream>
#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>
#include <cusp/convert.h>
#include <cusp/array1d.h>
#include "gather.h"


int main()
{

    std::cout << "Test Local Gather Matrix\n";
    std::vector<int> idx = {7,4,5,2,4,9};

    dg::LocalGatherMatrix<thrust::host_vector> gather(idx);
    std::vector<int> values = {0,1,2,3,4,5,6,7,8,9};
    std::vector<int> buffer(idx.size());
    gather.gather( 1., values, 0., buffer);
    for( unsigned i=0; i<idx.size(); i++)
        assert( buffer[i] == idx[i]);
    std::cout << "Gather PASSED\n";

    return 0;
}
