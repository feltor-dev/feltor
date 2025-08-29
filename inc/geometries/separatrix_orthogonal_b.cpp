#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include "dg/functors.h"

#include "dg/backend/timer.h"
#include "solovev.h"
#include "taylor.h"
//#include "guenter.h"
#include "dg/topology/transform.h"
#include "refined_curvilinearX.h"
#include "curvilinearX.h"
#include "separatrix_orthogonal.h"
#include "testfunctors.h"
#include "fluxfunctions.h"

#include "dg/file/file.h"


double sine( double x) {return sin(x);}
double cosine( double x) {return cos(x);}

thrust::host_vector<double> periodify( const thrust::host_vector<double>& in, const dg::GridX3d& g)
{
    //in is a 2d vector, out has a second layer on top
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
    std::cout << "Type n (3), Nx (8), Ny (176), Nz (1) \n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "Typed "<<n<<" "<<Nx<<" "<<Ny<<" "<<Nz<<"\n";
    auto js = dg::file::file2Json( argc == 1 ? "geometry_params_Xpoint.json" : argv[1]);
    //dg::geo::taylor::Parameters gp(js);
    //dg::geo::TokamakMagneticField mag = dg::geo::createTaylorField(gp);
    dg::geo::solovev::Parameters gp(js);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);
    double R_O = gp.R_0, Z_O = 0.;
    dg::geo::findOpoint( mag.get_psip(), R_O, Z_O);
    const double psipmin = mag.psip()(R_O, Z_O);
    std::cout << "Psi min "<<psipmin<<"\n";
    dg::Timer t;
    std::cout << "Type psi_0 (-15) \n";
    double psi_0 = -20;
    std::cin >> psi_0;
    std::cout << "Typed "<<psi_0<<"\n";
    std::cout << "Type fx and fy ( fx*Nx and fy*Ny must be integer) 0.25 0.04545454545454545 \n";
    double fx_0=1./4., fy_0=1./22.;
    std::cin >> fx_0>> fy_0;
    std::cout << "Typed "<<fx_0<<" "<<fy_0<<"\n";

    std::cout << "Type add_x and add_y \n";
    double add_x, add_y;
    std::cin >> add_x >> add_y;
    std::cout << "Typed "<<add_x<<" "<<add_y<<"\n";
    gp.display( std::cout);
    std::cout << "Constructing orthogonal grid ... \n";
    t.tic();
    double RX = gp.R_0-1.1*gp.triangularity*gp.a;
    double ZX = -1.1*gp.elongation*gp.a;
    dg::geo::findXpoint( mag.get_psip(), RX, ZX);
    const double psipX = mag.psip()( RX, ZX);
    //dg::geo::CylindricalSymmTensorLvl1 monitor_chi;
    dg::geo::CylindricalSymmTensorLvl1 monitor_chi = dg::geo::make_Xconst_monitor( mag.get_psip(), RX, ZX) ;
    //dg::geo::CylindricalSymmTensorLvl1 monitor_chi = dg::geo::make_Xbump_monitor( ag.get_psip(), RX, ZX, 100, 100) ;
    std::cout << "X-point set at "<<RX<<" "<<ZX<<" with Psi_p = "<<psipX<<"\n";

    //dg::geo::SeparatrixOrthogonal generator(mag.get_psip(), monitor_chi, psi_0, RX,ZX, mag.R0(), 0, 0, true);
    dg::geo::SeparatrixOrthogonal generator(mag.get_psip(), monitor_chi, psi_0, RX, ZX, mag.R0(), 0, 0, true);
    //dg::geo::SimpleOrthogonalX generator(mag.get_psip(), psi_0, RX,ZX, mag.R0(), 0,0);
    dg::EquidistXRefinement equi(add_x, add_y, 1,1);
    dg::geo::CurvilinearRefinedProductGridX3d g3d(equi, generator, fx_0, fy_0, n, Nx, Ny,Nz, dg::DIR, dg::NEU);
    dg::geo::CurvilinearRefinedGridX2d g2d(equi, generator, fx_0, fy_0, n, Nx, Ny,dg::DIR, dg::NEU);
    t.toc();
    dg::GridX3d g3d_periodic(g3d.x0(), g3d.x1(), g3d.y0(), g3d.y1(), g3d.z0(), g3d.z1(), g3d.fx(), g3d.fy(), g3d.n(), g3d.Nx(), g3d.Ny(), 2);
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    double psi_1 = -fx_0/(1.-fx_0)*psi_0;
    std::cout << "psi 1 is          "<<psi_1<<"\n";
    dg::Grid1d g1d( g2d.x0(), g2d.x1(), g2d.n(), g2d.Nx());
    g1d.display( std::cout);
    dg::HVec x_left = dg::evaluate( sine, g1d), x_right(x_left);
    dg::HVec y_left = dg::evaluate( cosine, g1d);
    std::vector<std::tuple<std::string, dg::HVec, std::string> > map2d;
    dg::HVec psi_p = dg::pullback( mag.psip(), g2d);
    map2d.emplace_back( "Psi_p", psi_p, "Poloidal flux function");
    //test if interpolation onto grid works
    dg::HVec psip_X(psi_p);
    {
        double Rmin=gp.R_0-1.2*gp.a;
        double Zmin=-1.2*gp.a*gp.elongation;
        double Rmax=gp.R_0+1.2*gp.a;
        double Zmax=1.2*gp.a*gp.elongation;
        dg::Grid2d grid2d(Rmin,Rmax,Zmin,Zmax, 3,200,200);
        std::vector<dg::HVec> coordsX = g2d.map();
        dg::IHMatrix grid2gX2d = dg::create::interpolation( coordsX[0], coordsX[1], grid2d);
        dg::HVec psipog2d   = dg::evaluate( mag.psip(), grid2d);
        dg::blas2::symv( grid2gX2d, psipog2d, psip_X);
    }
    map2d.emplace_back( "Psi_p_interpolated", psip_X, "Poloidal flux function");
    g2d.display();
    dg::HVec X( g2d.map()[0]), Y(g2d.map()[1]), P = dg::evaluate( dg::zero, g2d);
    map2d.emplace_back( "xc", X, "X-coordinate Cartesian");
    map2d.emplace_back( "yc", Y, "Y-coordinate Cartesian");
    map2d.emplace_back( "zc", P, "Z-coordinate Cartesian");

    dg::HVec ones = dg::evaluate( dg::one, g2d);
    dg::HVec temp0( g2d.size()), temp1(temp0);
    dg::HVec w2d = dg::create::weights( g2d);

    dg::SparseTensor<dg::HVec> metric = g2d.metric();
    dg::HVec vol = dg::tensor::volume(metric);
    map2d.emplace_back( "volume", vol, "Volume element");
    map2d.emplace_back( "g_xx", metric.value(0,0), "Metric element");
    map2d.emplace_back( "g_xy", metric.value(0,1), "Metric element");
    map2d.emplace_back( "g_yy", metric.value(1,1), "Metric element");

    X = dg::pullback( dg::geo::FuncDirNeu(mag, psi_0, psi_1, 480, -300, 70., 1. ), g2d);
    map2d.emplace_back( "FuncDirNeu1", X, "FuncDirNeu");
    X = dg::pullback( dg::geo::FuncDirNeu(mag, psi_0, psi_1, 420, -470, 50.,1.), g2d);
    map2d.emplace_back( "FuncDirNeu2", X, "FuncDirNeu");
    std::cout << "OPEN FILE orthogonalX.nc ...\n";
    dg::file::NcFile file( "orthogonalX.nc", dg::file::nc_clobber);
    file.defput_dim( "x", {{"axis", "X"}}, g3d_periodic.abscissas(0));
    file.defput_dim( "y", {{"axis", "Y"}}, g3d_periodic.abscissas(1));
    file.defput_dim( "z", {{"axis", "Z"}}, g3d_periodic.abscissas(2));

    for(auto tp : map2d)
    {
        file.defput_var( std::get<0>(tp), {"z", "y", "x"},
            {{"long_name", std::get<2>(tp)}}, {g3d_periodic.grid()},
            periodify( std::get<1>(tp), g3d_periodic));
    }
    file.close();
    std::cout << "FILE orthogonalX.nc CLOSED AND READY TO USE NOW!\n" <<std::endl;

    std::cout << "TEST VOLUME IS:\n";
    dg::CartesianGrid2d g2dC( gp.R_0 -1.2*gp.a, gp.R_0 + 1.2*gp.a, ZX, 1.2*gp.a*gp.elongation, 1, 5e2, 5e2, dg::PER, dg::PER);
    double psipmax = 0., psipmin_iris = psi_0;
    auto iris = dg::compose( dg::Iris( psipmin_iris, psipmax), mag.psip());
    dg::HVec vec  = dg::evaluate( iris, g2dC);
    dg::HVec g2d_weights = dg::create::volume( g2dC);
    double volumeRZP = dg::blas1::dot( vec, g2d_weights);

    dg::HVec cutter = dg::pullback( iris, g2d), vol_cut( cutter);
    dg::geo::ZCutter cut(ZX);
    dg::HVec zcutter = dg::pullback( cut, g2d);
    w2d = dg::create::weights( g2d);//make weights w/o refined weights
    dg::blas1::pointwiseDot(cutter, w2d, vol_cut);
    dg::blas1::pointwiseDot(zcutter, vol_cut, vol_cut);
    double volume = dg::blas1::dot( vol, vol_cut);
    std::cout << "volumeXYP is "<< volume<<std::endl;
    std::cout << "volumeRZP is "<< volumeRZP<<std::endl;
    std::cout << "relative difference in volume is "<<fabs(volumeRZP - volume)/volume<<std::endl;
    std::cout << "Note that the error might also be because the regions in the RZ grid and the orthogonal grid are not the same!\n";
    std::cout << "Note that the error might also come from the volume in RZP!\n";

    return 0;
}
