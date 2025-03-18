#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include "dg/algorithm.h"
#include "dg/file/file.h"

#include "solovev.h"
#include "taylor.h"
//#include "guenter.h"
#include "curvilinearX.h"
#include "ribeiroX.h"
#include "ds.h"

double sine( double x) {return sin(x);}
double cosine( double x) {return cos(x);}

thrust::host_vector<double> periodify( const thrust::host_vector<double>& in, const dg::GridX3d& g)
{
    assert( g.Nz() == 2);
    thrust::host_vector<double> out(g.size());
    for( unsigned s=0; s<g.Nz(); s++)
    for( unsigned i=0; i<g.Ny(); i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[(((s*g.Ny()+i)*g.n() + k)*g.Nx() + j)*g.n()+l] =
            in[((i*g.n() + k)*g.Nx() + j)*g.n()+l];

    //exchange two segments
    for( unsigned i=g.outer_Ny(); i<2*g.outer_Ny(); i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[(((1*g.Ny() + i)*g.n() + k)*g.Nx() + j)*g.n()+l] =
            in[(((i+g.inner_Ny())*g.n() + k)*g.Nx() + j)*g.n()+l];
    for( unsigned i=g.inner_Ny()+g.outer_Ny(); i<g.Ny(); i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[(((1*g.Ny() + i)*g.n() + k)*g.Nx() + j)*g.n()+l] =
            in[(((i-g.inner_Ny())*g.n() + k)*g.Nx() + j)*g.n()+l];
    if( g.outer_Ny() == 0)
    {
    //exchange two segments
    for( unsigned i=0; i<g.Ny()-1; i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[(((1*g.Ny() + i)*g.n() + k)*g.Nx() + j)*g.n()+l] =
            in[(((i+1)*g.n() + k)*g.Nx() + j)*g.n()+l];
    for( unsigned i=g.Ny()-1; i<g.Ny(); i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[(((1*g.Ny() + i)*g.n() + k)*g.Nx() + j)*g.n()+l] =
            in[(((0)*g.n() + k)*g.Nx() + j)*g.n()+l];
    }


    return out;
}

int main( int argc, char* argv[])
{
    std::cout << "Type n, Nx, Ny, Nz (Nx must be divided by 4 and Ny by 10) \n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    auto js = dg::file::file2Json( argc == 1 ? "geometry_params_Xpoint.json" : argv[1]);
    dg::geo::solovev::Parameters gp(js);
    dg::Timer t;
    std::cout << "Type psi_0 \n";
    double psi_0 = -16;
    std::cin >> psi_0;
    std::cout << "Type fx and fy ( fx*Nx and fy*Ny must be integer) \n";
    double fx_0=1./4., fy_0=1./22.;
    std::cin >> fx_0>> fy_0;
    gp.display( std::cout);
    std::cout << "Constructing orthogonal grid ... \n";
    t.tic();
    dg::geo::CylindricalFunctorsLvl2 psip = dg::geo::solovev::createPsip(gp);
    std::cout << "Psi min "<<psip.f()(gp.R_0, 0)<<"\n";
    double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    double Z_X = -1.1*gp.elongation*gp.a;
    dg::geo::findXpoint( psip, R_X, Z_X);

    double R0 = gp.R_0, Z0 = 0;
    dg::geo::RibeiroX generator(psip, psi_0, fx_0, R_X,Z_X, R0, Z0);
    dg::geo::CurvilinearProductGridX3d g3d(generator, fx_0, fy_0, n, Nx, Ny,Nz, dg::DIR, dg::NEU);
    dg::geo::CurvilinearGridX2d g2d(generator, fx_0, fy_0, n, Nx, Ny, dg::DIR, dg::NEU);
    t.toc();
    dg::GridX3d g3d_periodic(g3d.x0(), g3d.x1(), g3d.y0(), g3d.y1(), g3d.z0(), g3d.z1(), g3d.fx(), g3d.fy(), g3d.n(), g3d.Nx(), g3d.Ny(), 2);
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    dg::Grid1d g1d( g2d.x0(), g2d.x1(), g2d.n(), g2d.Nx());
    dg::HVec x_left = dg::evaluate( sine, g1d), x_right(x_left);
    dg::HVec y_left = dg::evaluate( cosine, g1d);
    dg::file::NcFile file( "ribeiroX.nc", dg::file::nc_clobber);
    file.defput_dim( "x", {{"axis", "X"}}, g3d_periodic.abscissas(0));
    file.defput_dim( "y", {{"axis", "Y"}}, g3d_periodic.abscissas(1));
    file.defput_dim( "z", {{"axis", "Z"}}, g3d_periodic.abscissas(2));
    file.defput_dim( "i", {{"axis", "X"}}, g1d.abscissas());
    file.defput_var( "x_left", {"i"}, {}, {g1d}, x_left);
    file.defput_var( "x_right", {"i"}, {}, {g1d}, x_right);
    file.defput_var( "y_left", {"i"}, {}, {g1d}, y_left);

    thrust::host_vector<double> psi_p = dg::pullback( psip.f(), g2d);
    g2d.display();
    file.defput_var( "psi", {"z", "y", "x"}, {}, {g3d_periodic.grid()}, periodify(
        psi_p, g3d_periodic));

    dg::DVec ones = dg::evaluate( dg::one, g2d);
    dg::DVec temp0( g2d.size()), temp1(temp0);
    dg::DVec w2d = dg::create::weights( g2d);

    file.defput_var( "xc", {"z", "y", "x"}, {}, {g3d_periodic.grid()},
        periodify( g2d.map()[0], g3d_periodic));
    file.defput_var( "yc", {"z", "y", "x"}, {}, {g3d_periodic.grid()},
        periodify( g2d.map()[1], g3d_periodic));

    dg::SparseTensor<dg::DVec> metric = g2d.metric();
    dg::DVec g_xx = metric.value(0,0), g_xy = metric.value(0,1), g_yy=metric.value(1,1);
    dg::DVec vol = dg::tensor::volume(metric);

    dg::blas1::pointwiseDivide( g_yy, g_xx, temp0);
    dg::blas1::axpby( 1., ones, -1., temp0, temp0);
    file.defput_var( "deformation", {"z", "y", "x"}, {}, {g3d_periodic.grid()},
        periodify( (dg::HVec)temp0, g3d_periodic));

    std::cout << "Construction successful!\n";

    //compute error in volume element (in conformal grid g^xx is the volume element)
    dg::blas1::pointwiseDot( g_xx, g_yy, temp0);
    dg::blas1::pointwiseDot( g_xy, g_xy, temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    dg::assign( g_xx,  temp1);
    dg::blas1::pointwiseDot( temp1, temp1, temp1);
    dg::blas1::axpby( 1., temp1, -1., temp0, temp0);
    double error = sqrt( dg::blas2::dot( temp0, w2d, temp0)/dg::blas2::dot( temp1, w2d, temp1));
    std::cout<< "Rel Error in Determinant is "<<error<<"\n";

    //compute error in determinant vs volume form
    dg::blas1::pointwiseDot( g_xx, g_yy, temp0);
    dg::blas1::pointwiseDot( g_xy, g_xy, temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    dg::blas1::transform( temp0, temp0, dg::SQRT<double>());
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    file.defput_var( "volume", {"z", "y", "x"}, {}, {g3d_periodic.grid()},
        periodify( (dg::HVec)temp0, g3d_periodic));
    dg::blas1::axpby( 1., temp0, -1., vol, temp0);
    error = sqrt(dg::blas2::dot( temp0, w2d, temp0)/dg::blas2::dot( vol, w2d, vol));
    std::cout << "Rel Consistency  of volume is "<<error<<"\n";

    //compare g^xx to volume form
    dg::assign( g_xx, temp0);
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    dg::blas1::axpby( 1., temp0, -1., vol, temp0);
    error=sqrt(dg::blas2::dot( temp0, w2d, temp0))/sqrt( dg::blas2::dot(vol, w2d, vol));
    std::cout << "Rel Error of volume form is "<<error<<"\n";

    std::cout << "TEST VOLUME IS:\n";
    dg::CartesianGrid2d g2dC( gp.R_0 -1.2*gp.a, gp.R_0 + 1.2*gp.a, -2.0*gp.a*gp.elongation, 1.2*gp.a*gp.elongation, 1, 5e3, 1e4, dg::PER, dg::PER);
    double psipmax = 0., psipmin = psi_0;
    auto iris = dg::compose( dg::Iris(  psipmin, psipmax), psip.f());
    dg::HVec vec  = dg::evaluate( iris, g2dC);
    dg::DVec cutter = dg::pullback( iris, g2d), cut_vol( cutter);
    dg::blas1::pointwiseDot(cutter, w2d, cut_vol);
    double volume = dg::blas1::dot( vol, cut_vol);
    dg::HVec g2d_weights = dg::create::volume( g2dC);
    double volumeRZP = dg::blas1::dot( vec, g2d_weights);
    std::cout << "volumeXYP is "<< volume<<std::endl;
    std::cout << "volumeRZP is "<< volumeRZP<<std::endl;
    std::cout << "relative difference in volume is "<<fabs(volumeRZP - volume)/volume<<std::endl;
    std::cout << "Note that the error might also come from the volume in RZP!\n";
    file.close();
    return 0;
}
