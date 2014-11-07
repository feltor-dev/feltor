#include <iostream>

#include <cusp/print.h>
#include <mpi.h>

#include "mpi_evaluation.h"
#include "mpi_precon.h"
#include "dz.h"
#include "functions.h"
#include "../blas2.h"
#include "../functors.h"
#include "interpolation.cuh"

#include "mpi_init.h"

struct Field
{
    Field( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    void operator()( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp)
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {
            double gradpsi = ((y[0][i]-R_0)*(y[0][i]-R_0) + y[1][i]*y[1][i])/I_0/I_0;
            yp[2][i] = y[0][i]*sqrt(1 + gradpsi);
            yp[0][i] = y[0][i]*y[1][i]/I_0;
            yp[1][i] = -y[0][i]*y[0][i]/I_0 + R_0/I_0*y[0][i] ;
        }
    }
    void operator()( const dg::HVec& y, dg::HVec& yp)
    {
        double gradpsi = ((y[0]-R_0)*(y[0]-R_0) + y[1]*y[1])/I_0/I_0;
        yp[2] = y[0]*sqrt(1 + gradpsi);
        yp[0] = y[0]*y[1]/I_0;
        yp[1] = y[0]/I_0*(R_0-y[0]) ;
    }
    private:
    double R_0, I_0;
};

double R_0 = 10;
double I_0 = 10;
//psi = 0.5*r^2
//b_phi = I_0/R/sqrt(I_0*I_0+r2) = I_0/R/B
//b_R =   Z/R/B
double func2d(double R, double Z)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    double B = sqrt(I_0*I_0+r2)/R;
    double bphi = I_0/R/R/B;
    return 1/bphi/R;
}
double func(double R, double Z, double phi)
{
    return -func2d(R,Z)*cos(phi);
}
double modulate( double R, double Z, double phi) {return -cos(phi);}
double deri(double R, double Z, double phi)
{
    //double r2 = (R-R_0)*(R-R_0)+Z*Z;
    return sin(phi)/R;
}
double deri2(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    double B = sqrt(I_0*I_0+r2)/R;
    double bphi = I_0/R/R/B;
    double bR = Z/R/B;
    return bphi/R*cos(phi) - bR*sin(phi)/R/R ;
}

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz; 
    MPI_Comm comm;
    mpi_init3d( dg::PER, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    Field field( R_0, I_0);
    double z0 = 0, z1 = 2.*M_PI;
    //double z0 = M_PI/2., z1 = 3./2.*M_PI;
    dg::MPI_Grid3d g3d( R_0 - 1, R_0+1, -1, 1, z0, z1, n, Nx, Ny, Nz, comm);
    dg::MPI_Grid3d g2d( R_0 - 1, R_0+1, -1, 1, z0, z1,  n, Nx, Ny, 1, comm);
    const dg::MPI_Precon w3d = dg::create::weights( g3d);
    const dg::MPI_Precon w2d = dg::create::weights( g2d);
    dg::DZ<dg::MMatrix, dg::MVec> dz( field, g3d, g3d.hz(), 1e-9, dg::DefaultLimiter(), dg::NEU);
    dg::DZ<dg::MMatrix, dg::MVec> dz2d( field, g2d, g3d.hz(), 1e-9, dg::DefaultLimiter(), dg::NEU);
    dz.set_boundaries( dg::PER, 0, 0);
    //dz.set_boundaries( dg::DIR, 0., -0.);

    dg::MVec function = dg::evaluate( func, g3d), derivative(function), 
             dzz(function);
    dg::MVec function2d = dg::evaluate( func, g2d), derivative2d( function2d) ;
    dg::MVec follow = dz.evaluate( func2d, 0), sinz(dg::evaluate( modulate, g3d));
    dg::blas1::pointwiseDot( follow, sinz, follow);
    dg::blas1::axpby( 1., function, -1., follow);
    double diff = dg::blas2::dot( w3d, follow);
    if(rank==0)std::cout << "Difference between function and followed evaluation: "<<diff<<"\n";
    const dg::MVec solution = dg::evaluate( deri, g3d);
    const dg::MVec solution2 = dg::evaluate( deri2, g3d);
    const dg::MVec solution2d = dg::evaluate( deri, g2d);
    dz( function, derivative);
    dz2d( function2d, derivative2d);
    dz.dzz( function, dzz);
    //dz( derivative, dzz);
    dg::blas1::axpby( 1., solution, -1., derivative);
    double norm = dg::blas2::dot( w3d, solution);
    if(rank==0)std::cout << "Norm Solution "<<sqrt( norm)<<"\n";
    double err = sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm);
    if(rank==0)std::cout << "Relative Difference in DZ is "<< err<<"\n";    
    dg::blas1::axpby( 1., solution2, -1., dzz);
    norm = dg::blas2::dot( w3d, solution2);
    err = sqrt( dg::blas2::dot( dzz, w3d, dzz)/norm);
    if(rank==0)std::cout << "Relative Difference in DZZ is "<< err<<"\n";    
    dg::blas1::axpby( 1., solution2d, -1., derivative2d);
    err = sqrt( dg::blas2::dot( w2d, derivative2d));
    if(rank==0)std::cout << "Difference in DZ2d is "<< err<<"\n";    
    MPI_Finalize();
    return 0;
}
