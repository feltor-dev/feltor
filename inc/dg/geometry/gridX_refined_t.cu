#include <iostream>

#include "gridX_refined.h"


int main ()
{
    std::cout<< "BOTH SIDES:\n";
    thrust::host_vector<double> both, both_abs;
    //don't forget to test the case add_x = 0 once in a while!
    dg::GridX1d g( 0,1, 0.125, 2,8, dg::DIR);
    int new_N = dg::refined::detail::exponential_Xref( 3, g, both, both_abs);
    double sum = 0;
    for( unsigned i=0; i<new_N*g.n(); i++)
    {
        std::cout << both[i] << "\t" <<both_abs[i] << std::endl;
        sum += 1./both[i]/g.n();
    }
    std::cout << "SUM IS: "<<sum<<" ("<<new_N<<")\n";

    return 0;
}
