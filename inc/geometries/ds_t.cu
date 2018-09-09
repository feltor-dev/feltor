#include <iostream>

#include <cusp/print.h>
#define DG_BENCHMARK
#include "dg/algorithm.h"
#include "magnetic_field.h"
#include "ds.h"
#include "toroidal.h"


const double R_0 = 10;
const double I_0 = 20; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius
double funcNEU(double R, double Z, double phi)
{
    return sin(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*sin(phi);
}
double deriNEU(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z; //(grad psi)^2
    return ( Z     *M_PI/2.*cos(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*sin(phi)
           -(R-R_0)*M_PI/2.*sin(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*sin(phi)
           + I_0/R*sin(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*cos(phi)
           )/sqrt(I_0*I_0+r2);
}
double funcDIR(double R, double Z, double phi)
{
    return cos(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*sin(phi);
}
double deriDIR(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z; //(grad psi)^2
    return (-Z      *M_PI/2.*sin(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*sin(phi)
            +(R-R_0)*M_PI/2.*cos(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*sin(phi)
            +I_0/R*cos(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*cos(phi)
           )/sqrt(I_0*I_0+r2);
}

int main(int argc, char * argv[])
{
    std::cout << "First test the cylindrical version\n";
    std::cout << "Note that it's faster to compute with OMP_NUM_THREADS=1\n";
    std::cout << "Type n (3), Nx(20), Ny(20), Nz(20)\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "You typed "<<n<<" "<<Nx<<" "<<Ny<<" "<<Nz<<std::endl;
    std::cout << "Type mx (10) and my (10)\n";
    unsigned mx, my;
    std::cin >> mx>> my;
    std::cout << "You typed "<<mx<<" "<<my<<std::endl;
    std::cout << "Create parallel Derivative!\n";

    //![doxygen]
    const dg::CylindricalGrid3d g3d( R_0 - a, R_0+a, -a, a, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::DIR, dg::DIR);
    //create magnetic field
    const dg::geo::TokamakMagneticField mag = dg::geo::createCircularField( R_0, I_0);
    const dg::geo::BinaryVectorLvl0 bhat( (dg::geo::BHatR)(mag), (dg::geo::BHatZ)(mag), (dg::geo::BHatP)(mag));
    //create Fieldaligned object and construct DS from it
    dg::geo::Fieldaligned<dg::aProductGeometry3d,dg::IDMatrix,dg::DVec>  dsFA( bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, mx, my, true,true,true);
    dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec> ds( dsFA, dg::not_normed, dg::centered);
    ///##########################################################///
    //apply to function
    dg::DVec function = dg::evaluate( funcNEU, g3d), derivative(function);
    ds( function, derivative);
    //![doxygen]
    std::cout << "TEST NEU Boundary conditions!\n";
    dg::DVec solution = dg::evaluate( deriNEU, g3d);
    dg::blas1::axpby( 1., solution, -1., derivative);
    const dg::DVec vol3d = dg::create::volume( g3d);
    double norm = dg::blas2::dot( derivative, vol3d, derivative);
    double sol = dg::blas2::dot( vol3d, solution);
    std::cout << "Error centered derivative "<< sqrt( norm/sol )<<"\n";
    ds.forward( 1., function, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Error Forward  Derivative "<<sqrt( norm/sol)<<"\n";
    ds.backward( 1., function, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Error Backward Derivative "<<sqrt( norm/sol)<<"\n";
    ///##########################################################///
    {
    std::cout << "TEST DIR Boundary conditions!\n";
    //dsFA.construct( bhat, g3d, dg::DIR, dg::DIR, dg::geo::NoLimiter(), 1e-8, mx, my, true,true,true);
    //ds.construct( dsFA, dg::not_normed, dg::centered);
    dg::geo::Fieldaligned<dg::aProductGeometry3d,dg::IDMatrix,dg::DVec>  dsFA( bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, mx, my, true,true,true);
    dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec> ds( dsFA, dg::not_normed, dg::centered);
    //apply to function
    function = dg::evaluate( funcDIR, g3d);
    ds( function, derivative);
    //![doxygen]
    solution = dg::evaluate( deriDIR, g3d);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    sol = dg::blas2::dot( vol3d, solution);
    std::cout << "Error centered derivative "<< sqrt( norm/sol )<<"\n";
    ds.forward( 1., function, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Error Forward  Derivative "<<sqrt( norm/sol)<<"\n";
    ds.backward( 1., function, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Error Backward Derivative "<<sqrt( norm/sol)<<"\n";
    }
    ///##########################################################///
    std::cout << "TEST FIELDALIGNED EVALUATION of a Gaussian\n";
    dg::Gaussian init0(R_0+0.5, 0, 0.2, 0.2, 1);
    dg::GaussianZ modulate(0., M_PI/3., 1);
    function = dsFA.evaluate( init0, modulate, Nz/2, 2);
    ds( function, derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_mpit)\n";

    return 0;
}
