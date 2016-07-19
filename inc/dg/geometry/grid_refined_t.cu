#include <iostream>

#include "grid_refined.h"


int main ()
{
    std::cout<< "BOTH SIDES:\n";
    thrust::host_vector<double> left, right, both, left_abs, right_abs, both_abs;
    thrust::host_vector<int> i_both, i_left, i_right;
    both = dg::refined::detail::exponential_ref( 3, 2, 5, 2, i_both);
    both_abs = dg::refined::detail::ref_abscissas( 10, 20, 2, 11, both);
    double sum = 0;
    for( unsigned i=0; i<both.size(); i++)
    {
        std::cout << both[i] <<"\t"<<i_both[i]<< "\t" <<both_abs[i] << std::endl;
        sum += 1./both[i]/2.;
    }
    std::cout << "SUM IS: "<<sum<<" (5)\n";
    std::cout<< "LEFT SIDE:\n";
    left = dg::refined::detail::exponential_ref( 2, 3, 5, 0, i_left);
    left_abs = dg::refined::detail::ref_abscissas( 10, 20, 3, 7, left);
    sum = 0;
    for( unsigned i=0; i<left.size(); i++)
    {
        std::cout << left[i] <<"\t"<<i_left[i]<<"\t"<<left_abs[i]<<std::endl;
        sum += 1./left[i]/3.;
    }
    std::cout << "SUM IS: "<<sum<<" (5)\n";
    std::cout<< "RIGHT SIDE:\n";
    right = dg::refined::detail::exponential_ref( 5, 1, 5, 1, i_right);
    right_abs = dg::refined::detail::ref_abscissas( 10, 20, 1, 10, right);
    sum =0;
    for( unsigned i=0; i<right.size(); i++)
    {
        std::cout << right[i] <<"\t"<<i_right[i]<<"\t"<<both_abs[i]<< std::endl;
        sum += 1./right[i]/1.;
    }

    std::cout << "SUM IS: "<<sum<<" (5)\n";

    return 0;
}
