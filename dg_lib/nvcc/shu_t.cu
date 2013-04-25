#include <iostream>
#include <iomanip>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

//#include "cuda_texture.cuh"

#include "arrvec2d.cuh"
#include "evaluation.cuh"
#include "shu.cuh"
#include "rk.cuh"


using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 20;
const unsigned Ny = 20;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const unsigned k = 3;
const double D = 0.01;
const double T = 2.;
const unsigned NT = (unsigned)(D*T*n*n*Nx*Nx/0.01/lx/lx);

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrVec;
typedef dg::ArrVec2d< double, n, DVec>  DArrVec;


/*
double amplitude = 100;
double x00 = 0.5*lx;
double y00 = 0.5*ly;
double sigma_x = 0.2;
double sigma_y = 0.2;
double gaussian( double x, double y)
{
    return  amplitude*
                   exp( -(double)((x-x00)*(x-x00)/2./sigma_x/sigma_x+
                                  (y-y00)*(y-y00)/2./sigma_y/sigma_y) );
}
*/

double initial( double x, double y){return 2.*sin(x)*sin(y);}
double solution( double x, double y) {return 2.*sin(x)*sin(y)*exp( -2.*T*D);}
double one( double x, double y){ return 1.;}


using namespace std;

int main()
{
    const double hx = lx/ (double)Nx;
    const double hy = ly/ (double)Ny;
    const double dt = T/(double)NT;
    /*
    /////////////////////////////////////////////////////////////////
    //create CUDA context that uses OpenGL textures in Glfw window
    cudaGlfwInit( 600, 600);
    glfwSetWindowTitle( "Texture test");
    glClearColor( 0.f, 0.f, 1.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);
    ////////////////////////////////////////////////////////////
    */
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Timestep                    " << dt << endl;
    cout << "# of timesteps              " << NT << endl;
    HArrVec omega = expand< double(&)(double, double), n> ( initial, 0, lx, 0, ly, Nx, Ny);
    DArrVec stencil = expand< double(&)(double, double), n> ( one, 0, lx, 0, ly, Nx, Ny);
    DArrVec sol = expand< double(&)(double, double), n> ( solution, 0, lx, 0, ly, Nx, Ny);
    DVec visual( Nx*Ny);
    DVec y0( omega.data()), y1( y0);
    Shu<double, n, DVec, cusp::device_memory> test( Nx, Ny, hx, hy, D);
    RK< 3, Shu<double, n, DVec, cusp::device_memory> > rk( y0);
    for( unsigned i=0; i<NT; i++)
    {
        rk( test, y0, y1, dt);
        thrust::swap(y0, y1);
        cout << "Total vorticity is: "<< blas2::dot( stencil.data(), S2D<double, n>(hx, hy), y0) << "\n";
        cout << "Total enstrophy  is "<<blas2::dot( y0, S2D<double, n>(hx, hy), y0)<<"\n";
    }


    cout << "Total vorticity is: "<< blas2::dot( stencil.data(), S2D<double, n>(hx, hy), y0) << "\n";
    cout << "Total enstrophy  is "<<blas2::dot( y0, S2D<double, n>(hx, hy), y0)<<"\n";
    blas1::axpby( 1., sol.data(), -1., y0);
    cudaThreadSynchronize();
    cout << "Distance to solution "<<sqrt(blas2::dot( S2D<double, n>(hx, hy), y0))<<endl; //don't forget sqrt when comuting errors

    /*
    ////////////////////////////////glfw//////////////////////////////
    int running = GL_TRUE;
    ColorMapRedBlueExt colors( 2.);
    cudaGraphicsResource* resource = allocateCudaGlBuffer( 3*Nx*Ny); 
    while( running)
    {
        cout << "Total vorticity is: "<< blas2::dot( stencil.data(), S2D<double, n>(hx, hy), y0) << "\n";
        //copy only the 00 index
        thrust::remove_copy_if( y0.begin(), y0.end(), stencil.data().begin(), visual.begin(), thrust::logical_not<double>());
        //generate a texture and draw the picture
        glClear( GL_COLOR_BUFFER_BIT);
        colors.scale() =  thrust::reduce( visual.begin(), visual.end(), 0, thrust::maximum<float>() );
        mapColors( resource, visual, colors);
        loadTexture( Ny, Nx);
        drawQuad( -1., 1., -1., 1.);
        glfwSwapBuffers();
        glfwWaitEvents(); //wait for user input

        //Timestep
        rk( test, y0, y1, dt);
        thrust::swap(y0, y1);

        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    freeCudaGlBuffer( resource);
    glfwTerminate();
    */

    ////////////////////////////////////////////////////////////////////

    return 0;

}
