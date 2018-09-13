#include <iostream>
#include <iomanip>

#include <mpi.h>
#define DG_BENCHMARK
#include "dg/algorithm.h"
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
double deriAdjNEU(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z; //(grad psi)^2
    return Z/R/(I_0*I_0+r2)*funcNEU(R,Z,phi) + deriNEU(R,Z,phi);
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
double deriAdjDIR(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z; //(grad psi)^2
    return Z/R/(I_0*I_0+r2)*funcDIR(R,Z,phi) + deriDIR(R,Z,phi);
}


int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz;
    MPI_Comm comm;
    dg::mpi_init3d( dg::NEU, dg::NEU, dg::PER, n, Nx, Ny, Nz, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    dg::CylindricalMPIGrid3d g3d( R_0 - a, R_0+a, -a, a, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER, comm);
    if(rank==0)std::cout << "Create parallel Derivative with default mx = my = 10!\n";
    dg::geo::TokamakMagneticField mag = dg::geo::createCircularField( R_0, I_0);
    dg::geo::BinaryVectorLvl0 bhat( (dg::geo::BHatR)(mag), (dg::geo::BHatZ)(mag), (dg::geo::BHatP)(mag));
    dg::geo::Fieldaligned<dg::aProductMPIGeometry3d,dg::MIDMatrix,dg::MDVec>  dsFA( bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, 1, 1, true, true);
    dg::geo::DS<dg::aProductMPIGeometry3d, dg::MIDMatrix, dg::MDMatrix, dg::MDVec> ds( dsFA, dg::not_normed, dg::centered);
    ///##########################################################///
    //apply to function
    dg::MDVec function = dg::evaluate( funcNEU, g3d), derivative(function);
    ds.centered( function, derivative);
    if(rank==0)std::cout << "TEST NEU Boundary conditions!\n";
    dg::MDVec solution = dg::evaluate( deriNEU, g3d);
    const dg::MDVec vol3d = dg::create::volume( g3d);
    double sol = dg::blas2::dot( vol3d, solution);
    dg::blas1::axpby( 1., solution, -1., derivative);
    double norm = dg::blas2::dot( derivative, vol3d, derivative);
    if(rank==0)std::cout << "Error centered derivative \t"<< sqrt( norm/sol )<<"\n";
    ds.forward( 1., function, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    if(rank==0)std::cout << "Error Forward  Derivative \t"<<sqrt( norm/sol)<<"\n";
    ds.backward( 1., function, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    if(rank==0)std::cout << "Error Backward Derivative \t"<<sqrt( norm/sol)<<"\n";
    ///##########################################################///
    ///We unfortunately cannot test convergence of adjoint because
    ///b and therefore bf does not fulfill Neumann boundary conditions
    if(rank==0)std::cout << "TEST ADJOINT derivatives!\n";
    solution = dg::evaluate( deriAdjNEU, g3d);
    sol = dg::blas2::dot( vol3d, solution);

    ds.centeredDiv( function, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    if(rank==0)std::cout << "Error centered divergence \t"<< sqrt( norm/sol )<<"\n";
    ds.forwardDiv( 1., function, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    if(rank==0)std::cout << "Error Forward  divergence \t"<<sqrt( norm/sol)<<"\n";
    ds.backwardDiv( 1., function, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    if(rank==0)std::cout << "Error Backward divergence \t"<<sqrt( norm/sol)<<"\n";

    ///##########################################################///
    if(rank==0)std::cout << "TEST DIR Boundary conditions!\n";
    dsFA.construct( bhat, g3d, dg::DIR, dg::DIR, dg::geo::NoLimiter(), 1e-8, 10, 10, true, true);
    ds.construct( dsFA, dg::not_normed, dg::centered);
    //apply to function
    dg::MDVec functionDIR = dg::evaluate( funcDIR, g3d);
    solution = dg::evaluate( deriDIR, g3d);
    sol = dg::blas2::dot( vol3d, solution);

    ds.centered( functionDIR, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    if(rank==0)std::cout << "Error centered derivative \t"<< sqrt( norm/sol )<<"\n";
    ds.forward( 1., functionDIR, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    if(rank==0)std::cout << "Error Forward  Derivative \t"<<sqrt( norm/sol)<<"\n";
    ds.backward( 1., functionDIR, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    if(rank==0)std::cout << "Error Backward Derivative \t"<<sqrt( norm/sol)<<"\n";

    ///##########################################################///
    if(rank==0)std::cout << "TEST ADJOINT derivatives!\n";
    solution = dg::evaluate( deriAdjDIR, g3d);
    sol = dg::blas2::dot( vol3d, solution);

    ds.centeredDiv( functionDIR, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    if(rank==0)std::cout << "Error centered divergence \t"<< sqrt( norm/sol )<<"\n";
    ds.forwardDiv( 1., functionDIR, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    if(rank==0)std::cout << "Error Forward  divergence \t"<<sqrt( norm/sol)<<"\n";
    ds.backwardDiv( 1., functionDIR, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    if(rank==0)std::cout << "Error Backward divergence \t"<<sqrt( norm/sol)<<"\n";

    ///##########################################################///

    if(rank==0)std::cout << "TEST FIELDALIGNED EVALUATION of a Gaussian\n";
    dg::Gaussian init0(R_0+0.5, 0, 0.2, 0.2, 1);
    dg::GaussianZ modulate(0., M_PI/3., 1);
    function = ds.fieldaligned().evaluate( init0, modulate, Nz/2, 2);
    ds( function, derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    if(rank==0)std::cout << "Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_t)\n";
    MPI_Finalize();
    return 0;
}
