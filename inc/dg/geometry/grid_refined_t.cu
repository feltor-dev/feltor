#include <iostream>

#include "grid_refined.h"


int main ()
{
    std::cout<< "BOTH SIDES:\n";
    thrust::host_vector<double> both = dg::refined::detail::exponential_ref( 3, 2, 5, 2);
    thrust::host_vector<double> both_abs = dg::refined::detail::ref_abscissas( 10, 20, 2, 11, both);
    double sum = 0;
    for( unsigned i=0; i<both.size(); i++)
    {
        std::cout << both[i] << "\t" <<both_abs[i] << std::endl;
        sum += 1./both[i]/2.;
    }
    std::cout << "SUM IS: "<<sum<<" (5)\n";
    std::cout<< "LEFT SIDE:\n";
    thrust::host_vector<double> left = dg::refined::detail::exponential_ref( 2, 3, 5, 0);
    thrust::host_vector<double> both_left = dg::refined::detail::ref_abscissas( 10, 20, 3, 7, left);
    sum = 0;
    for( unsigned i=0; i<left.size(); i++)
    {
        std::cout << left[i] << "\t"<<both_left[i]<<std::endl;
        sum += 1./left[i]/3.;
    }
    std::cout << "SUM IS: "<<sum<<" (5)\n";
    std::cout<< "RIGHT SIDE:\n";
    thrust::host_vector<double> right = dg::refined::detail::exponential_ref( 5, 1, 5, 1);
    thrust::host_vector<double> both_right = dg::refined::detail::ref_abscissas( 10, 20, 1, 10, right);
    sum =0;
    for( unsigned i=0; i<right.size(); i++)
    {
        std::cout << right[i] <<"\t"<<both_right[i]<< std::endl;
        sum += 1./right[i]/1.;
    }

    std::cout << "SUM IS: "<<sum<<" (5)\n";

    return 0;
}
