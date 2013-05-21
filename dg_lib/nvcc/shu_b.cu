#include <iostream>
#include <iomanip>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "timer.cuh"

#include "cuda_texture.cuh"
#include "functors.cuh"

#include "arrvec2d.cuh"
#include "evaluation.cuh"
#include "shu.cuh"
#include "rk.cuh"



using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 32; 
const unsigned Ny = 32; 
const double lx = 1.;
const double ly = 1.;

const unsigned k = 3;
const double D = 0.0;
const double U = 1; //the dipole doesn't move with this velocity because box is not infinite
const double R = 0.2*lx;
const double T = 1.;//0.6;
const unsigned NT =  (unsigned)(T*n*Nx/0.05/lx);
const double eps = 1e-3; //CG method
const unsigned N = 3; //only output every Nth step 

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrVec;
typedef dg::ArrVec2d< double, n, DVec>  DArrVec;

typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;

typedef cusp::device_memory Memory;

using namespace std;

int main()
{
    Timer t;
    const double hx = lx/ (double)Nx;
    const double hy = ly/ (double)Ny;
    const double dt = T/(double)NT;
    /////////////////////////////////////////////////////////////////
    //create CUDA context that uses OpenGL textures in Glfw window
    HostWindow w( 600, 600);
    glfwSetWindowTitle( "Navier Stokes");
    ////////////////////////////////////////////////////////////
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Timestep                    " << dt << endl;
    //cout << "# of timesteps              " << NT << endl;
    cout << "Diffusion                   " << D <<endl;
    dg::Lamb lamb( 0.5*lx, 0.8*ly, R, U);
    HArrVec omega = expand< dg::Lamb, n> ( lamb, 0, lx, 0, ly, Nx, Ny);
    DArrVec stencil = expand< double(&)(double, double), n> ( one, 0, lx, 0, ly, Nx, Ny);
    dg::Lamb lamb2( 0.5*lx, 0.8*ly-0.9755*U*T, R, U);
    HArrVec solh = expand< dg::Lamb, n> ( lamb2, 0, lx, 0, ly, Nx, Ny);
    DVec sol = solh.data();
    DVec y0( omega.data()), y1( y0);
    //make solver and stepper
    Shu<double, n, DVec, Memory> test( Nx, Ny, hx, hy, D, eps);
    RK< k, Shu<double, n, DVec, Memory> > rk( y0);

    t.tic();
    test( y0, y1);
    t.toc();
    cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";
    double vorticity = blas2::dot( stencil.data(), S2D<double, n>(hx, hy), y0);
    double enstrophy = 0.5*blas2::dot( y0, S2D<double, n>(hx, hy), y0);
    double energy =    0.5*blas2::dot( y0, S2D<double, n>(hx, hy), test.potential()) ;

    double time = 0;
    ////////////////////////////////glfw//////////////////////////////
    //create equidistant backward transformation
    dg::Operator<double, n> backwardeq( dg::DLT<n>::backwardEQ);
    dg::Operator<double, n*n> backward2d = dg::tensor( backwardeq, backwardeq);
    HMatrix hbackward = dg::tensor( Nx*Ny, backward2d);
    DMatrix backward = hbackward;
    //create visualisation vectors
    int running = GL_TRUE;
    DVec visual( n*n*Nx*Ny), visual2( visual);
    HVec hvisual( n*n*Nx*Ny);
    thrust::device_vector<int> map = dg::makePermutationMap<n>( Nx, Ny);
    dg::ColorMapRedBlueExt colors( 1.);
    cout << "Press any key to start!\n";
    double x; 
    cin >> x;
    while (running && time < T)
    {
        //transform field to an equidistant grid
        //t.tic();
        //cout << "Total vorticity           is: "<<blas2::dot( stencil.data(), S2D<double, n>(hx, hy), y0) << "\n";
        //cout << "Relative enstrophy error  is: "<<(0.5*blas2::dot( S2D<double, n>(hx, hy), y0) - enstrophy)/enstrophy<<"\n";
        //test( y0, y1); //get the potential ready
        //cout << "Relative energy error     is: "<<(0.5*blas2::dot( test.potential(), S2D<double, n>(hx, hy), y0) - energy)/energy<<"\n";
        //t.toc();
        dg::blas2::symv( backward, y0, visual2);
        cudaThreadSynchronize();
        thrust::scatter( visual2.begin(), visual2.end(), map.begin(), visual.begin());
        cudaThreadSynchronize();
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
        //std::cout << "Color scale " << colors.scale() <<"\n";
        //draw and swap buffers
        hvisual = visual;
        cudaThreadSynchronize();
        w.draw( hvisual, n*Nx, n*Ny, colors);
        //step 
        t.tic();
        for( unsigned i=0; i<N; i++)
        {
            rk( test, y0, y1, dt);
            thrust::swap(y0, y1);
        }
        t.toc();
        //cout << "Timer for one step: "<<t.diff()/N<<"s\n";
        cout << "Simulation Time "<<time<< " \ttook "<<t.diff()/(double)N<<"\t per step"<<endl;
        time += N*dt;

        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    ////////////////////////////////////////////////////////////////////
    cout << "Analytic formula enstrophy "<<lamb.enstrophy()<<endl;
    cout << "Analytic formula energy    "<<lamb.energy()<<endl;
    cout << "Total vorticity           is: "<<blas2::dot( stencil.data(), S2D<double, n>(hx, hy), y0) << "\n";
    cout << "Relative enstrophy error  is: "<<(0.5*blas2::dot( S2D<double, n>(hx, hy), y0) - enstrophy)/enstrophy<<"\n";
    test( y0, y1); //get the potential ready
    cout << "Relative energy error     is: "<<(0.5*blas2::dot( test.potential(), S2D<double, n>(hx, hy), y0) - energy)/energy<<"\n";

    blas1::axpby( 1., y0, -1, sol);
    cout << "Distance to solution: "<<sqrt(blas2::dot( S2D<double,n>(hx,hy), sol ))<<endl;

    cout << "Press any key to quit!\n";
    cin >> x;
    return 0;

}
