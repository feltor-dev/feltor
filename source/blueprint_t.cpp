#include <iostream>
#include "read_input.h"
#include "blueprint.h"


using namespace std;
using namespace toefl;

int main()
{
    Physical phys; 
    Algorithmic alg;
    Boundary bound;
    vector<double> para;
    try{ para = read_input( "input.dat"); }
    catch (Message& m) {  m.display(); return -1;}
    phys.d = para[1];
    phys.g_e = phys.g[0] = para[2];
    phys.g[2] = para[3];
    phys.tau[0] = para[4];
    phys.tau[1] = para[22];
    phys.mu[1] = para[23];
    phys.a[1] = para[24];
    phys.kappa = para[6];

    phys.a[0] = 1. -phys.a[1];
    phys.g[1] = (phys.g_e - phys.a[1] * phys.g[1])/(1.-phys.a[1]);
    phys.mu[0] = 1.0;

    bound.ly = para[12];
    alg.nx = para[13];
    alg.ny = para[14];
    alg.dt = para[15];

    alg.h = bound.ly / (double)alg.ny;
    bound.lx = (double)alg.nx * alg.h;
    bound.bc_x = TL_PERIODIC;


    Blueprint bp( phys, bound, alg);
    try{ bp.consistencyCheck();}
    catch( Message& m) {m.display(); bp.display(); return -1;}
    bp.display(cout);

    return 0;
}
