#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "dg/algorithm.h"
#include "guenter.h"
#include "fluxfunctions.h"
#include "magnetic_field.h"
#include "testfunctors.h"

const double R_0 = 10;
const double I_0 = 20; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius

int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny, Nz;
    std::cout << "# Test the 3d elliptic inversion with projection and alignment tensors!\n";
    std::cout << "# Type n (nx=ny=nz=n), Nx, Ny and Nz\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    double eps;
    std::cout << "# Type epsilon! \n";
    std::cin >> eps;
    //const dg::CylindricalGrid3d g3d( R_0-a, R_0+a, -a, a, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::DIR, dg::DIR, dg::PER);
    const dg::CylindricalGrid3d g3d( {R_0-a, R_0+a, n, Nx, dg::DIR}, {-a, a, n, Ny, dg::DIR}, {0, 2.*M_PI, n, Nz, dg::PER});
    dg::DVec w3d = dg::create::volume( g3d);

    dg::Elliptic3d<dg::CylindricalGrid3d, dg::DMatrix, dg::DVec> elliptic(g3d, dg::centered);
    const dg::geo::TokamakMagneticField mag = dg::geo::createGuenterField( R_0, I_0);
    const dg::geo::CylindricalVectorLvl0 bhat = dg::geo::createBHat(mag);
    dg::SparseTensor<dg::DVec> bb = dg::geo::createAlignmentTensor( bhat, g3d);
    elliptic.set_chi( bb);

    auto test = [&](const auto& x, auto& y){
                //  y = ( 1 - Delta) x
                dg::blas2::symv( elliptic, x, y);
                dg::blas1::axpby( 1., x, +1., y, y);
            };
    dg::PCG< dg::DVec > pcg( w3d, n*n*Nx*Ny);

    auto ff = dg::geo::TestFunctionDirNeu(mag);
    const dg::DVec sol = dg::evaluate( ff, g3d);
    dg::DVec b = dg::evaluate( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionDirNeu>(mag, ff), g3d);
    dg::DVec x = b;

    std::cout << "# --------- Alignment Tensor:\n";
    std::cout << "# For a precision of "<< eps<<" ..."<<std::endl;
    t.tic();
    std::cout << "# Number of pcg iterations "<< pcg.solve( test, x, b,
            1., w3d, eps)<<std::endl;
    t.toc();
    std::cout << "# ... on the device took "<< t.diff()<<"s\n";
    dg::DVec  error(  sol);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( w3d, error);
    double norm = dg::blas2::dot( w3d, sol);
    std::cout << "L2 Norm of relative error is:               " <<sqrt( normerr/norm)<<std::endl;
    dg::SparseTensor<dg::DVec> hh = dg::geo::createProjectionTensor( bhat, g3d);
    dg::Elliptic3d<dg::CylindricalGrid3d, dg::DMatrix, dg::DVec> ellipticP(g3d, dg::centered);
    ellipticP.set_chi( hh);
    dg::DVec one = dg::evaluate( dg::one, g3d);
    dg::blas1::copy( 3., one);
    ellipticP.set_chi( one);
    dg::blas1::copy( 1., one);
    ellipticP.set_chi( one);

    b = dg::evaluate( dg::geo::DPerpFunction<dg::geo::TestFunctionDirNeu>(mag,ff), g3d);

    std::cout << "# --------- Projection Tensor:\n";
    t.tic();
    std::cout << "# Number of pcg iterations " << pcg.solve( ellipticP, x, b,
            ellipticP.precond(), ellipticP.weights(), eps)<<std::endl;
    t.toc();
    std::cout << "# ... on the device took "<< t.diff()<<"s\n";
    dg::blas1::axpby( 1., x,+1.,sol,  error); //elliptic is negative

    normerr = dg::blas2::dot( w3d, error);
    norm = dg::blas2::dot( w3d, sol);
    std::cout << "L2 Norm of relative error is:               " <<sqrt( normerr/norm)<<std::endl;


    return 0;
}
