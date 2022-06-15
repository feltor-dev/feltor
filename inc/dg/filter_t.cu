#include <iostream>
#include <cmath>

#include "filter.h"


int main()
{
    std::vector<double> x = {7,2,3, -20,4,7, 2.3,2.1,4, 4,15,-2};
    std::vector<double> medians = {7,4.5,3,2.5,3,3.5,3,2.65,3,3.5,4,3.5};
    std::vector<double> mad = {0,2.5,1,2.5,1,2.5,1,1,1,1.3,1.9,1.45};
    int row_offsets[2] = {0, 3};
    std::vector<int> column_indices = {0,1,2,3,4,5,6,7,8,9,10,11,12};
    double result;
    dg::CSRMedianFilter median_filter;
    for( unsigned i=1; i<x.size()+1; i+=2)
    {
        row_offsets[1] = i;
        median_filter( 0, row_offsets, &column_indices[0], &x[0], &x[0], &result);
        std::cout << "Median of [";
        int k=0;
        for( k=0; k<row_offsets[1]-1; k++)
            std::cout << x[k]<<", ";
        std::cout<<x[k]<<"] is "<<result<<" ("<<medians[i-1]<<")"<<std::endl;
    }
}
