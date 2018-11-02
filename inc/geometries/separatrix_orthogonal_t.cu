#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include "json/json.h"

#include "dg/functors.h"

#include "dg/backend/timer.h"
#include "solovev.h"
//#include "taylor.h"
//#include "guenther.h"
#include "dg/topology/transform.h"
#include "refined_curvilinearX.h"
#include "curvilinearX.h"
#include "separatrix_orthogonal.h"
#include "init.h"
#include "testfunctors.h"

#include "file/nc_utilities.h"


struct ZCutter
{
    ZCutter(double ZX): Z_X(ZX){}
    double operator()(double R, double Z) const {
        if( Z> Z_X)
            return 1;
        return 0;
    }
    private:
    double Z_X;
};
double sine( double x) {return sin(x);}
double cosine( double x) {return cos(x);}
//typedef dg::FieldAligned< dg::CurvilinearGridX3d<dg::HVec> , dg::IHMatrix, dg::HVec> HFA;

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
    std::cout << "Type n (3), Nx (8), Ny (176), Nz (1) \n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "Typed "<<n<<" "<<Nx<<" "<<Ny<<" "<<Nz<<"\n";
    Json::Value js;
    if( argc==1)
    {
        //std::ifstream is("geometry_params_Xpoint_taylor.js");
        std::ifstream is("geometry_params_Xpoint.js");
        is >> js;
    }
    else
    {
        std::ifstream is(argv[1]);
        is >> js;
    }
    //dg::geo::taylor::Parameters gp(js);
    dg::geo::solovev::Parameters gp(js);
    dg::Timer t;
    std::cout << "Type psi_0 (-15) \n";
    double psi_0 = -20;
    std::cin >> psi_0;
    std::cout << "Typed "<<psi_0<<"\n";
    //std::cout << "Type fx and fy ( fx*Nx and fy*Ny must be integer) \n";
    double fx_0=1./4., fy_0=1./22.;
    //std::cin >> fx_0>> fy_0;
    std::cout << "Typed "<<fx_0<<" "<<fy_0<<"\n";

    std::cout << "Type add_x and add_y \n";
    double add_x, add_y;
    std::cin >> add_x >> add_y;
    std::cout << "Typed "<<add_x<<" "<<add_y<<"\n";
    gp.display( std::cout);
    std::cout << "Constructing orthogonal grid ... \n";
    t.tic();
    //dg::geo::TokamakMagneticField c = dg::geo::createTaylorField(gp);
    dg::geo::TokamakMagneticField c = dg::geo::createSolovevField(gp);
    std::cout << "Psi min "<<c.psip()(gp.R_0, 0)<<"\n";
    double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    double Z_X = -1.1*gp.elongation*gp.a;
    //dg::geo::CylindricalSymmTensorLvl1 monitor_chi;
    dg::geo::CylindricalSymmTensorLvl1 monitor_chi = dg::geo::make_Xconst_monitor( c.get_psip(), R_X, Z_X) ;
    //dg::geo::CylindricalSymmTensorLvl1 monitor_chi = dg::geo::make_Xbump_monitor( c.get_psip(), R_X, Z_X, 100, 100) ;
    std::cout << "X-point set at "<<R_X<<" "<<Z_X<<"\n";

    double R0 = gp.R_0, Z0 = 0;
    dg::geo::SeparatrixOrthogonal generator(c.get_psip(), monitor_chi, psi_0, R_X,Z_X, R0, Z0,0, true);
    //dg::geo::SimpleOrthogonalX generator(c.get_psip(), psi_0, R_X,Z_X, R0, Z0,0);
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
    int ncid;
    file::NC_Error_Handle err;
    err = nc_create( "orthogonalX.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim3d[3], dim1d[1];
    err = file::define_dimensions(  ncid, dim3d, g3d_periodic.grid());
    //err = file::define_dimensions(  ncid, dim3d, g2d.grid());
    err = file::define_dimension(  ncid, "i", dim1d, g1d);
    int coordsID[2], defID, volID, divBID, gxxID, gyyID, gxyID;
    err = nc_def_var( ncid, "psi", NC_DOUBLE, 3, dim3d, &gxyID);
    err = nc_def_var( ncid, "deformation", NC_DOUBLE, 3, dim3d, &defID);
    err = nc_def_var( ncid, "volume", NC_DOUBLE, 3, dim3d, &volID);
    err = nc_def_var( ncid, "divB", NC_DOUBLE, 3, dim3d, &divBID);
    err = nc_def_var( ncid, "num_solution", NC_DOUBLE, 3, dim3d, &gxxID);
    err = nc_def_var( ncid, "ana_solution", NC_DOUBLE, 3, dim3d, &gyyID);
    err = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 3, dim3d, &coordsID[0]);
    err = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 3, dim3d, &coordsID[1]);

    thrust::host_vector<double> psi_p = dg::pullback( c.psip(), g2d);
    g2d.display();
    //err = nc_put_var_double( ncid, onesID, periodify(psi_p, g3d_periodic).data());
    //err = nc_put_var_double( ncid, onesID, periodify(g2d.g(), g3d_periodic).data());
    dg::HVec X( g2d.size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g2d.size(); i++)
    {
        X[i] = g2d.map()[0][i];
        Y[i] = g2d.map()[1][i];
    }
    err = nc_put_var_double( ncid, coordsID[0], periodify(X, g3d_periodic).data());
    err = nc_put_var_double( ncid, coordsID[1], periodify(Y, g3d_periodic).data());

    dg::HVec ones = dg::evaluate( dg::one, g2d);
    dg::HVec temp0( g2d.size()), temp1(temp0);
    dg::HVec w2d = dg::create::weights( g2d);

    dg::SparseTensor<dg::HVec> metric = g2d.metric();
    dg::HVec g_xx = metric.value(0,0), g_xy = metric.value(0,1), g_yy=metric.value(1,1);
    dg::HVec vol = dg::tensor::volume(metric);
    dg::blas1::transfer( vol, X);
    err = nc_put_var_double( ncid, volID, periodify(X, g3d_periodic).data());
    dg::blas1::transfer( g_xx, X);
    err = nc_put_var_double( ncid, gxxID, periodify(X, g3d_periodic).data());
    dg::blas1::transfer( g_xy, X);
    err = nc_put_var_double( ncid, gxyID, periodify(X, g3d_periodic).data());
    dg::blas1::transfer( g_yy, X);
    err = nc_put_var_double( ncid, gyyID, periodify(X, g3d_periodic).data());

    std::cout << "Construction successful!\n";

    X = dg::pullback( dg::geo::FuncDirNeu(c, psi_0, psi_1, 480, -300, 70., 1. ), g2d);
    err = nc_put_var_double( ncid, divBID, periodify(X, g3d_periodic).data());
    X = dg::pullback( dg::geo::FuncDirNeu(c, psi_0, psi_1, 420, -470, 50.,1.), g2d);
    err = nc_put_var_double( ncid, defID, periodify(X, g3d_periodic).data());
    err = nc_close( ncid);

    std::cout << "TEST VOLUME IS:\n";
    dg::CartesianGrid2d g2dC( gp.R_0 -1.2*gp.a, gp.R_0 + 1.2*gp.a, Z_X, 1.2*gp.a*gp.elongation, 1, 5e3, 5e3, dg::PER, dg::PER);
    gp.psipmax = 0., gp.psipmin = psi_0;
    dg::geo::Iris iris( c.psip(), gp.psipmin, gp.psipmax);
    dg::HVec vec  = dg::evaluate( iris, g2dC);
    dg::HVec g2d_weights = dg::create::volume( g2dC);
    double volumeRZP = dg::blas1::dot( vec, g2d_weights);

    dg::HVec cutter = dg::pullback( iris, g2d), vol_cut( cutter);
    ZCutter cut(Z_X);
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
