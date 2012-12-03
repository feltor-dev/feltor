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
    phys.kappa = 0.1;
    phys.g[0] = phys.g[1] = 1;
    phys.a[0] = phys.a[1] = 1;
    phys.mu[0] = phys.mu[1] = 1;
    phys.tau[0] = phys.tau[1] = 1;

    Equations equations( phys);
    Poisson poisson(phys);

    QuadMat< complex<double>,2> m = equations.withIons( 1, {0,2});

    array<double,2> arr = poisson.withIons( -1);
    double g1_i = poisson.gamma1_i( -1);
    double g1_z = poisson.gamma1_z( -1);
    double g2_i = poisson.gamma2_i( -1);
    double g2_z = poisson.gamma2_z( -1);
    cout << "equations\n" << m <<endl;
    cout << "p0 " <<arr[0] << " p1 "<<arr[1]<<endl;
    cout << g1_i << " " << g1_z <<" "<<g2_i<<" "<<g2_z<<"\n";
    
    return 0;
}

