#include <iostream>

#include "file/nc_utilities.h"

#include "dg/geometry/refined_gridX.h"
#include "dg/backend/timer.cuh"
#include "dg/backend/grid.h"
#include "dg/backend/gridX.h"
#include "dg/backend/derivativesX.h"
#include "dg/backend/evaluationX.cuh"
#include "dg/cg.h"
#include "dg/elliptic.h"

#include "solovev.h"
#include "taylor.h"
//#include "ribeiroX.h"
#include "curvilinearX.h"
#include "separatrix_orthogonal.h"
#include "testfunctors.h"

using namespace dg::geo::taylor;
const char* parameters = "geometry_params_Xpoint_taylor.js";

//using namespace dg::geo::solovev;
//const char* parameters = "geometry_params_Xpoint.js";

int main(int argc, char**argv)
{
    std::cout << "Type n, Nx (fx = 1./4.), Ny (fy = 1./22.)\n";
    unsigned n, Nx, Ny;
    std::cin >> n>> Nx>>Ny;   
    std::cout << "Type psi_0 (-100)! \n";
    double psi_0, psi_1;
    std::cin >> psi_0;
    Json::Reader reader;
    Json::Value js;
    if( argc==1)
    {
        std::ifstream is(parameters);
        reader.parse(is,js,false);
    }
    else
    {
        std::ifstream is(argv[1]);
        reader.parse(is,js,false);
    }
    GeomParameters gp(js);
    gp.display( std::cout);
    dg::Timer t;
    std::cout << "Constructing grid ... \n";
    t.tic();

    ////////////////construct Generator////////////////////////////////////
    dg::geo::TokamakMagneticField c = dg::geo::createTaylorField(gp);
    std::cout << "Psi min "<<c.psip()(gp.R_0, 0)<<"\n";
    double R0 = gp.R_0, Z0 = 0;
    //double R_X = gp.R_0-1.4*gp.triangularity*gp.a;
    //double Z_X = -1.0*gp.elongation*gp.a;
    double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    double Z_X = -1.1*gp.elongation*gp.a;
    std::cout << "X-point at "<<R_X <<" "<<Z_X<<"\n";
    dg::geo::SeparatrixOrthogonal generator(c.get_psip(), psi_0, R_X,Z_X, R0, Z0,0);
    dg::CurvilinearGridX2d g2d( generator, 0.25, 1./22., n, Nx, Ny, dg::DIR, dg::NEU);
    dg::Elliptic<dg::CurvilinearGridX2d, dg::Composite<dg::DMatrix>, dg::DVec> pol( g2d, dg::not_normed, dg::forward);
    double fx = 0.25;
    psi_1 = -fx/(1.-fx)*psi_0;
    std::cout << "psi 1 is          "<<psi_1<<"\n";

    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s\n";
    std::cout << "Computing on "<<n<<" x "<<Nx<<" x "<<Ny<<"\n";
    ///////////////////////////////////////////////////////////////////////////
    int ncid;
    file::NC_Error_Handle ncerr;
    ncerr = nc_create( "testX.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim2d[2];
    ncerr = file::define_dimensions(  ncid, dim2d, g2d.grid());
    int coordsID[2], psiID, functionID, function2ID;
    ncerr = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim2d, &coordsID[0]);
    ncerr = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim2d, &coordsID[1]);
    ncerr = nc_def_var( ncid, "error", NC_DOUBLE, 2, dim2d, &psiID);
    ncerr = nc_def_var( ncid, "num_solution", NC_DOUBLE, 2, dim2d, &functionID);
    ncerr = nc_def_var( ncid, "ana_solution", NC_DOUBLE, 2, dim2d, &function2ID);

    dg::HVec X( g2d.size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g2d.size(); i++)
    {
        X[i] = g2d.map()[0][i];
        Y[i] = g2d.map()[1][i];
    }
    ncerr = nc_put_var_double( ncid, coordsID[0], X.data());
    ncerr = nc_put_var_double( ncid, coordsID[1], Y.data());
    dg::DVec x =    dg::evaluate( dg::zero, g2d);
    ////////////////////////blob solution////////////////////////////////////////
    //const dg::DVec b =        dg::pullback( dg::geo::EllipticBlobDirNeuM<MagneticField>(c,psi_0, psi_1, 450, -340, 40.,1.), g2d);
    //const dg::DVec chi  =  dg::pullback( dg::ONE(), g2d);
    //const dg::DVec solution =     dg::pullback( dg::geo::FuncDirNeu<MagneticField>(c, psi_0, psi_1, 450, -340, 40., 1. ), g2d);
    //////////////////////////blob solution on X-point/////////////////////////////
    //const dg::DVec b =        dg::pullback( dg::geo::EllipticBlobDirNeuM<MagneticField>(c,psi_0, psi_1, 480, -420, 40.,1.), g2d);
    //const dg::DVec chi  =  dg::pullback( dg::ONE(), g2d);
    //const dg::DVec solution =     dg::pullback( dg::geo::FuncDirNeu<MagneticField>(c, psi_0, psi_1, 480, -420, 40., 1. ), g2d);
    ////////////////////////////laplace psi solution/////////////////////////////
    //const dg::DVec b =        dg::pullback( c.laplacePsip);
    //const dg::DVec chi =      dg::evaluate( dg::one, g2d);
    //const dg::DVec solution =     dg::pullback( c.psip, g2d);
    /////////////////////////////Dir/////FIELALIGNED SIN///////////////////
    const dg::DVec b =    dg::pullback( dg::geo::EllipticXDirNeuM(c, psi_0, psi_1), g2d);
    dg::DVec chi  =  dg::pullback( dg::geo::Bmodule(c), g2d);
    dg::blas1::plus( chi, 1e4);
    //const dg::DVec chi =  dg::pullback( dg::ONE(), g2d);
    const dg::DVec solution = dg::pullback( dg::geo::FuncXDirNeu(c, psi_0, psi_1 ), g2d);
    ////////////////////////////////////////////////////////////////////////////

    const dg::DVec vol2d = dg::create::volume( g2d);
    const dg::DVec inv_vol2d = dg::create::inv_volume( g2d);
    const dg::DVec v2d = dg::create::inv_weights( g2d);
    pol.set_chi( chi);
    //compute error
    dg::DVec error( solution);
    const double eps = 1e-11;
    std::cout << "eps \t # iterations \t error \t hx_max\t hy_max \t time/iteration \n";
    std::cout << eps<<"\t";
    t.tic();
    dg::Invert<dg::DVec > invert( x, n*n*Nx*Ny, eps);
    //unsigned number = invert(pol, x,b, vol2d, inv_vol2d );
    unsigned number = invert(pol, x,b, vol2d, v2d ); //inv weights are better preconditioners
    std::cout <<number<<"\t";
    t.toc();
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( vol2d, error);
    const double norm = dg::blas2::dot( vol2d, solution);
    std::cout << sqrt( err/norm) << "\t";
    ///////////////////////////////////metric//////////////////////
    dg::SparseTensor<dg::DVec> metric = g2d.metric();
    dg::DVec gyy = metric.value(1,1), gxx = metric.value(0,0), vol = dg::tensor::volume(metric).value(); 
    dg::blas1::transform( gxx, gxx, dg::SQRT<double>());
    dg::blas1::transform( gyy, gyy, dg::SQRT<double>());
    dg::blas1::pointwiseDot( gxx, vol, gxx);
    dg::blas1::pointwiseDot( gyy, vol, gyy);
    dg::blas1::scal( gxx, g2d.hx());
    dg::blas1::scal( gyy, g2d.hy());
    double hxX = dg::interpolate( 0, 0, gxx, g2d);
    double hyX = dg::interpolate( 0, 0, gyy, g2d);
    std::cout << *thrust::max_element( gxx.begin(), gxx.end()) << "\t";
    std::cout << *thrust::max_element( gyy.begin(), gyy.end()) << "\t";
    std::cout << hxX << "\t";
    std::cout << hyX << "\t";
    std::cout<<t.diff()/(double)number<<"s"<<std::endl;

    dg::blas1::transfer( error, X);
    ncerr = nc_put_var_double( ncid, psiID, X.data());
    dg::blas1::transfer( x, X);
    ncerr = nc_put_var_double( ncid, functionID, X.data());
    dg::blas1::transfer( solution, Y);
    //dg::blas1::axpby( 1., X., -1, Y);
    ncerr = nc_put_var_double( ncid, function2ID, Y.data());
    ncerr = nc_close( ncid);


    return 0;
}
