#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits.h>  // UINT_MAX is needed in cusp (v0.5.1) but limits.h is not included
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

#include "geometries/solovev.h"
#include "geometries/conformal.h"
#include "geometries/orthogonal.h"

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
#define Grid OrthogonalGrid2d<DVec>

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

        for(int i=0;i<size_r;i++)
            cout << zeta1d[i] << " ";
        cout << endl;

        for(int i=0;i<size_phi;i++)
            cout << eta1d[i] << " ";
        cout << endl;

        // the first coordinate has stride=1 (according to inc/dg/backend/evaluation.cuh:61)
        for(int j=0;j<size_phi;j++)
            for(int i=0;i<size_r;i++) {
                double r   = zeta1d[i] + r_min;
                double phi = eta1d[j];
                x[i+size_r*j] = r*cos(phi);
                y[i+size_r*j] = r*sin(phi);

                zetaX[i+size_r*j] = cos(phi);
                zetaY[i+size_r*j] = sin(phi);
                etaX[i+size_r*j] = -sin(phi)/r;
                etaY[i+size_r*j] = -cos(phi)/r;
            }

    }
   
    double width() const{return r_max-r_min;}
    double height() const{return 2*M_PI;}
    bool isOrthogonal() const{return true;}
    bool isConformal()  const{return false;}
};

int main()
{
    double r_min = 0.2;
    double r_max = 2.0;

    Timer t;
    const Parameters p( file::read_input( "input.txt"));
    p.display();
    if( p.k != k)
    {
        std::cerr << "Time stepper needs recompilation!\n";
        return -1;
    }
    //Grid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);

    PolarGenerator generator(r_min, r_max);
    Grid grid( generator, p.n, p.Nx, p.Ny, dg::DIR); // second coordiante is periodic by default

    DVec w2d( create::weights(grid));

#ifdef OPENGL_WINDOW
    //create CUDA context that uses OpenGL textures in Glfw window
    std::stringstream title;
    GLFWwindow* w = draw::glfwInitAndCreateWindow(600, 600, "");
    draw::RenderHostData render( 1,1);
#endif

    //dg::Lamb lamb( p.posX*p.lx, p.posY*p.ly, p.R, p.U);
    dg::Lamb lamb( 0.9, M_PI, 0.5, p.U);

    HVec omega = evaluate ( lamb, grid);
    DVec stencil = evaluate( one, grid);
    DVec y0( omega ), y1( y0);
    //subtract mean mass 
    if( p.bc_x == dg::PER && p.bc_y == dg::PER)
    {
        double meanMass = dg::blas2::dot( y0, w2d, stencil)/(double)(p.lx*p.ly);
        dg::blas1::axpby( -meanMass, stencil, 1., y0);
    }
    //make solver and stepper
    Shu<Grid, DMatrix, DVec> shu( grid, p.eps);
    Diffusion<Grid, DMatrix, DVec> diffusion( grid, p.D);
    Karniadakis< DVec > ab( y0, y0.size(), 1e-9);

    t.tic();
    shu( y0, y1);
    t.toc();
    cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";
    double vorticity = blas2::dot( stencil , w2d, y0);
    double enstrophy = 0.5*blas2::dot( y0, w2d, y0);
    double energy =    0.5*blas2::dot( y0, w2d, shu.potential()) ;
    
    std::cout << "Total energy:     "<<energy<<"\n";
    std::cout << "Total enstrophy:  "<<enstrophy<<"\n";
    std::cout << "Total vorticity:  "<<vorticity<<"\n";

    double time = 0;
#ifdef OPENGL_WINDOW
    //create visualisation vectors
    DVec visual( grid.size());
    HVec hvisual( grid.size());
    //transform vector to an equidistant grid
    dg::IDMatrix equidistant = dg::create::backscatter( grid );
    draw::ColorMapRedBlueExt colors( 1.);
#endif

    ab.init( shu, diffusion, y0, p.dt);
    ab( shu, diffusion, y0); //make potential ready

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
        t.tic();
        for( unsigned i=0; i<p.itstp; i++)
        {
            ab( shu, diffusion, y0 );
        }
        t.toc();
        //cout << "Timer for one step: "<<t.diff()/N<<"s\n";
        time += p.itstp*p.dt;

    }

#ifdef OPENGL_WINDOW
    glfwTerminate();
#endif

    cout << "Analytic formula enstrophy "<<lamb.enstrophy()<<endl;
    cout << "Analytic formula energy    "<<lamb.energy()<<endl;
    cout << "Total vorticity          is: "<<blas2::dot( stencil , w2d, ab.last()) << "\n";
    cout << "Relative enstrophy error is: "<<(0.5*blas2::dot( w2d, ab.last()) - enstrophy)/enstrophy<<"\n";
    cout << "Relative energy error    is: "<<(0.5*blas2::dot( shu.potential(), w2d, ab.last()) - energy)/energy<<"\n";

    return 0;
}
