#include <iostream>

#include "grid_refined.h"


int main ()
{
    std::cout<< "BOTH SIDES:\n";
    thrust::host_vector<double> left, right, both, left_abs, right_abs, both_abs;
    //don't forget to test the case add_x = 0 once in a while!
    dg::Grid1d<double> g( 0,1, 2,4, dg::PER);
    int node;
    std::cout<< "Type node to refine 0,..,4!\n";
    std::cin >> node;
    int new_N = dg::refined::detail::exponential_ref( 3, node, g, both, both_abs);
    double sum = 0;
    for( unsigned i=0; i<new_N*g.n(); i++)
    {
        std::cout << both[i] << "\t" <<both_abs[i] << std::endl;
        sum += 1./both[i]/g.n();
    }
    std::cout << "SUM IS: "<<sum<<" ("<<new_N<<")\n";
    std::cout<< "LEFT SIDE:\n";
    dg::Grid1d<double> gl( 0,1, 2,5, dg::DIR);
    new_N = dg::refined::detail::exponential_ref( 2, 0, gl, left, left_abs);
    sum = 0;
    for( unsigned i=0; i<new_N*gl.n(); i++)
    {
        std::cout << left[i] <<"\t"<<left_abs[i]<<std::endl;
        sum += 1./left[i]/gl.n();
    }
    std::cout << "SUM IS: "<<sum<<" ("<<new_N<<")\n";
    std::cout<< "RIGHT SIDE:\n";
    dg::Grid1d<double> gr( 0,1, 1, 5, dg::DIR);
    new_N = dg::refined::detail::exponential_ref( 5, gr.N(), gr, right, right_abs);
    sum =0;
    for( unsigned i=0; i<new_N*gr.n(); i++)
    {
        std::cout << right[i] <<"\t"<<both_abs[i]<< std::endl;
        sum += 1./right[i]/gr.n();
    }
    std::cout << "SUM IS: "<<sum<<" ("<<new_N<<")\n";

    return 0;
}
