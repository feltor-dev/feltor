#include <iostream>
#include <iomanip>
#include <vector>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "cuda_texture.cuh"

#include "arrvec2d.cuh"
#include "evaluation.cuh"
#include "toefl.cuh"
#include "rk.cuh"


using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 20;
const unsigned Ny = 20;
const double lx = 128.;
const double ly = 128.;

const unsigned k = 3;
const double dt = 0.005;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrVec;
typedef dg::ArrVec2d< double, n, DVec>  DArrVec;


double amplitude = 1.;
double x00 = 0.5*lx;
double y00 = 0.5*ly;
double sigma_x = 0.06*lx;
double sigma_y = 0.06*ly;
double gaussian( double x, double y)
{
    return  amplitude*
                   exp( -(double)((x-x00)*(x-x00)/2./sigma_x/sigma_x+
                                  (y-y00)*(y-y00)/2./sigma_y/sigma_y) );
}

double one( double x, double y){ return 1.;}


using namespace std;

int main()
{
    const double hx = lx/ (double)Nx;
    const double hy = ly/ (double)Ny;
    /////////////////////////////////////////////////////////////////
    //create CUDA context that uses OpenGL textures in Glfw window
    cudaGlfwInit( 600, 600);
    glfwSetWindowTitle( "Texture test");
    glClearColor( 0.f, 0.f, 1.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);
    ////////////////////////////////////////////////////////////
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Timestep                    " << dt << endl;
    HArrVec ne = expand< double(&)(double, double), n> ( gaussian, 0, lx, 0, ly, Nx, Ny);
    DArrVec stencil = expand< double(&)(double, double), n> ( one, 0, lx, 0, ly, Nx, Ny), vorticity( stencil);
    DVec visual( Nx*Ny);
    vector<DVec> y0(2, ne.data()), y1(y0);
    Toefl<double, n, DVec, cusp::device_memory> test( Nx, Ny, hx, hy, 0.,  0.5, 1);
    RK< 3, Toefl<double, n, DVec, cusp::device_memory> > rk( y0);
    //cout<< fixed <<setprecision(2);

    ////////////////////////////////glfw//////////////////////////////
    int running = GL_TRUE;
    ColorMapRedBlueExt colors( amplitude);
    cudaGraphicsResource* resource = allocateCudaGlBuffer( 3*Nx*Ny); 
    while( running)
    {
        blas1::axpby( -1., y0[0], 0., vorticity.data());
        cudaThreadSynchronize();
        blas1::axpby(  1., y0[1], 1., vorticity.data());
        cudaThreadSynchronize();
        cout << "Total charge is: "<< blas2::dot( stencil.data(), S2D<double, n>(hx, hy), vorticity.data()) << "\n";
        //copy only the 00 index
        //thrust::remove_copy_if( y0[1].begin(), y0[1].end(), stencil.data().begin(), visual.begin(), thrust::logical_not<double>() );
        thrust::remove_copy_if( vorticity.data().begin(), vorticity.data().end(), stencil.data().begin(), visual.begin(), thrust::logical_not<double>() );
        cudaThreadSynchronize();
        //generate a texture and draw the picture
        glClear( GL_COLOR_BUFFER_BIT);
        colors.scale() =  thrust::reduce( visual.begin(), visual.end(), -1., thrust::maximum<float>() );
        cout << "Scale is " <<colors.scale() <<"\n";
        mapColors( resource, visual, colors);
        loadTexture( Ny, Nx);
        drawQuad( -1., 1., -1., 1.);
        glfwSwapBuffers();
        //glfwWaitEvents();

        rk( test, y0, y1, dt);
        cudaThreadSynchronize();
        for( unsigned j=0; j<2; j++)
            thrust::swap( y0[j], y1[j]);
        cudaThreadSynchronize();

        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    freeCudaGlBuffer( resource);
    glfwTerminate();

    ////////////////////////////////////////////////////////////////////

    return 0;

}
