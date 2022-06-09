#include <iostream>
#include "filter.h"


int main()
{
    std::vector<double> x = {7,2,3, -20,4,7, 2.3,2.1,4, 4,15,-2};
    int row_offsets[2] = {0, 3};
    std::vector<int> column_indices = {0,1,2,3,4,5,6,7,8,9,10,11,12};
    double result;
    dg::CSRMedianFilter filter;
    for( unsigned i=1; i<12; i++)
    {
        row_offsets[1] = i;
        filter( 0, row_offsets, &column_indices[0], &x[0], &x[0], &result);
        std::cout << "Median of [";
        int k=0;
        for( k=0; k<row_offsets[1]-1; k++)
            std::cout << x[k]<<", ";
        std::cout<<x[k]<<"] is "<<result<<std::endl;
    }
}
