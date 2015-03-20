#include <iostream>
#include "file/read_input.h"
#include "blueprint.h"


using namespace std;
using namespace toefl;

int main()
{
    Physical phys; 
    Algorithmic alg;
    Boundary bound;
    vector<double> para;
    try{ para = file::read_input( "input.txt"); }
    catch (Message& m) {  m.display(); return -1;}
    phys.d = para[7];
    phys.nu = para[8];
    phys.g_e = phys.g[0] = para[11];
    phys.g[1] = para[15];
    phys.tau[0] = para[12];
    phys.tau[1] = para[18];
    phys.mu[1] = para[17];
    phys.a[1] = para[16];
    phys.kappa = para[9];

    phys.a[0] = 1. -phys.a[1];
    phys.g[0] = (phys.g_e - phys.a[1] * phys.g[1])/(1.-phys.a[1]);
    phys.mu[0] = 1.0;//single charged ions

    bound.ly = para[4];
    alg.nx = para[1];
    alg.ny = para[2];
    alg.dt = para[3];

    alg.h = bound.ly / (double)alg.ny;
    bound.lx = (double)alg.nx * alg.h;
    bound.bc_x = TL_PERIODIC;


    Blueprint bp( phys, bound, alg);
    if( para[13])
        bp.enable( TL_IMPURITY);
    try{ bp.consistencyCheck();}
    catch( Message& m) {m.display(); bp.display(); return -1;}
    bp.display(cout);

    return 0;
}
