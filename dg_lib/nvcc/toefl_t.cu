#include <iostream>
#include <vector>

#include "cuda_texture.cuh"

#include "evaluation.cuh"
#include "toefl.cuh"
#include "rk.cuh"
#include "arrvec2d.cuh"


using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 50;
const unsigned Ny = 50;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const unsigned k = 3;
const double nu = 0.01;
const double T = 0.8;
const unsigned NT = (unsigned)(nu*T*n*n*Nx*Nx/0.01/lx/lx);

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrVec;
typedef dg::ArrVec2d< double, n, DVec>  DArrVec;


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

double one( double x, double y){ return 1.;}


using namespace std;

int main()
{
    const double hx = lx/ (double)Nx;
    const double hy = ly/ (double)Ny;
    const double dt = T/(double)NT;
    /////////////////////////////////////////////////////////////////
    //create CUDA context that uses OpenGL textures in Glfw window
    cudaGlfwInit( 300, 300);
    glfwSetWindowTitle( "Texture test");
    glClearColor( 0.f, 0.f, 1.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);
    ////////////////////////////////////////////////////////////
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Timestep                    " << dt << endl;
    cout << "# of timesteps              " << NT << endl;
    HArrVec ne = expand< double(&)(double, double), n> ( gaussian, 0, lx, 0, ly, Nx, Ny);
    HArrVec stencil = expand< double(&)(double, double), n> ( one, 0, lx, 0, ly, Nx, Ny);
    vector<DVec> y0(2, ne.data()), y1(y0);
    Toefl<double, n, DVec, cusp::device_memory> test( Nx, Ny, hx, hy, 1., 1., 0.005,  0.5, 1);
    RK< 3, Toefl<double, n, DVec, cusp::device_memory> > rk( y0);
    for( unsigned i=0; i<NT; i++)
    {
        rk( test, y0, y1, dt);
        for( unsigned j=0; j<2; j++)
            thrust::swap(y0[j], y1[j]);
    }





    DVec visual( Nx*Ny);
    ArrVec2d_View<double, n, DVec> neview( y0[0], Nx);
    for( unsigned i=0; i<Ny; i++)
        for( unsigned j=0; j<Nx; j++)
            visual[i*Nx+j] = neview(i,j, 0,0 );
    ////////////////////////////////glfw//////////////////////////////
    int running = GL_TRUE;
    ColorMapRedBlueExt colors( amplitude);
    cudaGraphicsResource* resource = allocateCudaGlBuffer( 3*Nx*Ny); 
    while( running)
    {
        //generate a texture
        glClear(GL_COLOR_BUFFER_BIT);
        mapColors( resource, visual, colors);
        loadTexture( Ny, Nx);
        drawQuad( -1., 1., -1., 1.);
        glfwSwapBuffers();
        glfwWaitEvents();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    freeCudaGlBuffer( resource);
    glfwTerminate();

    ////////////////////////////////////////////////////////////////////

    return 0;

}
