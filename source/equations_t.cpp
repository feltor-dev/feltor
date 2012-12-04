#include <iostream>

#include "equations.h"


using namespace std;
using namespace toefl;


int main()
{
    Physical phys;
    phys.d = 2.;
    phys.nu = 0.01;
    phys.g_e = 1;
    phys.kappa = 0;
    phys.g[0] = phys.g[1] = 1;
    phys.a[0] = phys.a[1] = 1;
    phys.mu[0] = phys.mu[1] = 1;
    phys.tau[0] = phys.tau[1] = 0;

    Equations equations( phys);
    Poisson poisson(phys);

    double laplace = -2;
    complex<double> dx = {0,1}, dy = dx;
    array<double,2> arr;
    QuadMat< complex<double>,2> m;

    equations(m, dx, dy );
    poisson(arr, laplace);
    double g1_i = poisson.gamma1_i( -1);
    double g1_z = poisson.gamma1_z( -1);
    //double g2_i = poisson.gamma2_i( -1);
    //double g2_z = poisson.gamma2_z( -1);
    phys.display( cout);
    cout << "equations2 with dx = " <<dx<<" dy = "<<dy<<"\n" << m <<endl;
    cout <<"Poisson2 with laplace = "<<laplace
         <<"\n p0 = "<<arr[0] << " p1 "<<arr[1]<<endl;
    cout << "gamma1_i = " <<g1_i << " gamma1_z = " << g1_z <<"\n";
    
    return 0;
}

