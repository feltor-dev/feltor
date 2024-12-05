
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
    {1,0}, {2,6}, {3,6}, {0,2}, {0,1}, {2,1}, {2,3}, {1,1}, {3,6}, {1,0}, {3,0}, {0,2}};
    thrust::host_vector<std::array<int,2>> unique;
    thrust::host_vector<int> bufferIdx, sort_map, reduction_keys;
    dg::detail::find_unique<thrust::host_vector>(
        gIdx, sort_map, reduction_keys, bufferIdx, unique);

    std::cout<< "Found unique values \n";
    for( unsigned u=0; u<unique.size(); u++)
        std::cout << unique[u][0]<<" "<<unique[u][1]<<"\n";
    std::cout << std::endl;
    thrust::host_vector<int> pids(unique.size());
    for( int i=0; i<(int)pids.size(); i++)
        pids[i] = unique[i][0];
    thrust::host_vector<int> locally_unique_pids, howmany;
    dg::detail::find_same<thrust::host_vector>( pids, locally_unique_pids, howmany);
    thrust::host_vector<int> sendTo( 7, 0 );
    for( unsigned i=0; i<locally_unique_pids.size(); i++)
        sendTo[locally_unique_pids[i]] = howmany[i];
    std::cout<< "Found unique pids \n";
    for( unsigned u=0; u<sendTo.size(); u++)
        std::cout << "pid "<<u<<" "<<sendTo[u]<<"\n";



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
        assert( values[i] + cusp_result[i] == feltor_result[i]);
    std::cout << "Scatter reduce PASSED\n";

    return 0;
}
