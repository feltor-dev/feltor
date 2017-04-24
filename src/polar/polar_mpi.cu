#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits.h>  // UINT_MAX is needed in cusp (v0.5.1) but limits.h is not included
#include <mpi.h>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "dg/backend/timer.cuh"
#include "dg/backend/evaluation.cuh"
#include "dg/backend/xspacelib.cuh"
#include "dg/runge_kutta.h"
#include "dg/multistep.h"
#include "dg/helmholtz.h"
#include "dg/backend/typedefs.cuh"
#include "dg/functors.h"

//#include "geometries/solovev.h"
//#include "geometries/conformal.h"
#include "geometries/mpi_orthogonal.h"

#ifdef OPENGL_WINDOW
#include "draw/host_window.h"
#endif

#include "file/read_input.h"

#include "ns.h"
#include "parameters.h"

using namespace std;
using namespace dg;
const unsigned k = 3;

//typedef CartesianGrid2d Grid;
//#define Grid OrthogonalGrid2d<DVec>
// local container for the grid
#define Grid OrthogonalMPIGrid2d<DVec>

struct PolarGenerator
{
    private:
        double r_min, r_max;

    public:

    PolarGenerator(double _r_min, double _r_max) : r_min(_r_min), r_max(_r_max) {}

    void operator()( 
         const thrust::host_vector<double>& zeta1d, 
         const thrust::host_vector<double>& eta1d, 
         thrust::host_vector<double>& x, 
         thrust::host_vector<double>& y, 
         thrust::host_vector<double>& zetaX, 
         thrust::host_vector<double>& zetaY, 
         thrust::host_vector<double>& etaX, 
         thrust::host_vector<double>& etaY) {

        int size_r   = zeta1d.size();
        int size_phi = eta1d.size();
        int size     = size_r*size_phi;

        x.resize(size); y.resize(size);
        zetaX.resize(size); zetaY.resize(size);
        etaX.resize(size); etaY.resize(size);

        // the first coordinate has stride=1
        for(int j=0;j<size_phi;j++)
            for(int i=0;i<size_r;i++) {
                double r   = zeta1d[i] + r_min;
                double phi = eta1d[j];

                x[i+size_r*j] = r*cos(phi);
                y[i+size_r*j] = r*sin(phi);

                zetaX[i+size_r*j] = cos(phi);
                zetaY[i+size_r*j] = sin(phi);
                etaX[i+size_r*j] = -sin(phi)/r;
                etaY[i+size_r*j] =  cos(phi)/r;
            }

    }
   
    double width() const{return r_max-r_min;}
    double height() const{return 2*M_PI;}
    bool isOrthogonal() const{return true;}
    bool isConformal()  const{return false;}
};

struct func_r {
    double r_min;
    func_r(double _r_min) : r_min(_r_min) {}

    double operator()(double x, double y) {
        return r_min+x;
    }
};


struct LogPolarGenerator
{
    private:
        double r_min, r_max;

    public:

    LogPolarGenerator(double _r_min, double _r_max) : r_min(_r_min), r_max(_r_max) {}

    void operator()(
         const thrust::host_vector<double>& zeta1d,
         const thrust::host_vector<double>& eta1d,
         thrust::host_vector<double>& x,
         thrust::host_vector<double>& y,
         thrust::host_vector<double>& zetaX,
         thrust::host_vector<double>& zetaY,
         thrust::host_vector<double>& etaX,
         thrust::host_vector<double>& etaY) {

        int size_r   = zeta1d.size();
        int size_phi = eta1d.size();
        int size     = size_r*size_phi;

        x.resize(size); y.resize(size);
        zetaX.resize(size); zetaY.resize(size);
        etaX.resize(size); etaY.resize(size);

        // the first coordinate has stride=1
        for(int j=0;j<size_phi;j++)
            for(int i=0;i<size_r;i++) {
                double l   = zeta1d[i] + log(r_min);
                double phi = eta1d[j];

                x[i+size_r*j] = exp(l)*cos(phi);
                y[i+size_r*j] = exp(l)*sin(phi);

                zetaX[i+size_r*j] = cos(phi)*exp(-l);
                zetaY[i+size_r*j] = sin(phi)*exp(-l);
                etaX[i+size_r*j] = -sin(phi)*exp(-l);
                etaY[i+size_r*j] =  cos(phi)*exp(-l);
            }

    }

    double width() const{return log(r_max)-log(r_min);}
    double height() const{return 2*M_PI;}
    bool isOrthogonal() const{return true;}
    bool isConformal()  const{return true;}
};

#ifdef LOG_POLAR
    #define Generator LogPolarGenerator
#else
    #define Generator PolarGenerator
#endif

int main()
{
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    Timer t;
    const Parameters p( file::read_input( "input.txt"));
    p.display();
    if( p.k != k)
    {
        std::cerr << "Time stepper needs recompilation!\n";
        return -1;
    }

    //Grid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    Generator generator(p.r_min, p.r_max); // Generator is defined by the compiler
    Grid grid( generator, p.n, p.Nx, p.Ny, dg::DIR); // second coordiante is periodic by default

    MDVec w2d( create::weights(grid));

#ifdef OPENGL_WINDOW
    //create CUDA context that uses OpenGL textures in Glfw window
    std::stringstream title;
    GLFWwindow* w = draw::glfwInitAndCreateWindow(600, 600, "");
    draw::RenderHostData render( 1,1);
#endif

    dg::Lamb lamb( p.posX, p.posY, p.R, p.U);
    MHVec omega = evaluate ( lamb, grid);
#if LOG_POLAR
    MDVec stencil = evaluate( one, grid);
#else
    MDVec stencil = evaluate( func_r(p.r_min), grid);
#endif
    MDVec y0( omega ), y1( y0);

    //make solver and stepper
    Shu<Grid, MDMatrix, MDVec> shu( grid, p.eps);
    Diffusion<Grid, MDMatrix, MDVec> diffusion( grid, p.D);
    Karniadakis< MDVec > ab( y0, y0.size(), 1e-9);

    t.tic();
    shu( y0, y1);
    t.toc();
    cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";

    double vorticity = blas2::dot( stencil , w2d, y0);
    MDVec ry0(stencil);
    blas1::pointwiseDot( stencil, y0, ry0);
    double enstrophy = 0.5*blas2::dot( ry0, w2d, y0);
    double energy =    0.5*blas2::dot( ry0, w2d, shu.potential()) ;
    cout << "Total vorticity:  "<<vorticity<<"\n";
    cout << "Total enstrophy:  "<<enstrophy<<"\n";
    cout << "Total energy:     "<<energy<<"\n";

    double time = 0;
#ifdef OPENGL_WINDOW
    //create visualisation vectors
    MDVec visual( grid.size());
    HVec hvisual( grid.size());
    //transform vector to an equidistant grid
    dg::IDMatrix equidistant = dg::create::backscatter( grid );
    draw::ColorMapRedBlueExt colors( 1.);
#endif

    ab.init( shu, diffusion, y0, p.dt);
    ab( shu, diffusion, y0); //make potential ready

    t.tic();
    while (time < p.maxout*p.itstp*p.dt)
    {
#ifdef OPENGL_WINDOW
        if(glfwWindowShouldClose(w))
            break;

        dg::blas2::symv( equidistant, ab.last(), visual);
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
        //draw and swap buffers
        dg::blas1::transfer( visual, hvisual);
        render.renderQuad( hvisual, p.n*p.Nx, p.n*p.Ny, colors);
        title << "Time "<<time<< " \ttook "<<t.diff()/(double)p.itstp<<"\t per step";
        glfwSetWindowTitle(w, title.str().c_str());
        title.str("");
        glfwPollEvents();
        glfwSwapBuffers(w);
#endif

        //step 
        for( unsigned i=0; i<p.itstp; i++)
        {
            ab( shu, diffusion, y0 );
        }
        time += p.itstp*p.dt;

    }
    t.toc();

#ifdef OPENGL_WINDOW
    glfwTerminate();
#endif


    double vorticity_end = blas2::dot( stencil , w2d, ab.last());
    blas1::pointwiseDot( stencil, ab.last(), ry0);
    double enstrophy_end = 0.5*blas2::dot( ry0, w2d, ab.last());
    double energy_end    = 0.5*blas2::dot( ry0, w2d, shu.potential()) ;
    cout << "Vorticity error           :  "<<vorticity_end-vorticity<<"\n";
    cout << "Enstrophy error (relative):  "<<(enstrophy_end-enstrophy)/enstrophy<<"\n";
    cout << "Energy error    (relative):  "<<(energy_end-energy)/energy<<"\n";

    cout << "Runtime: " << t.diff() << endl;

    return 0;
}
