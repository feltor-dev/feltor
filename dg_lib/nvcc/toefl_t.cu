#include <iostream>
#include <vector>
#include <cuda_gl_interop.h>

#include <GL/glfw.h>
#include "../../lib/texture.h"


#include "toefl.cuh"
#include "rk.cuh";
#include "arrvec2d.cuh"


using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 10;
const unsigned Ny = 10;
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
double sigma_x = 2;
double sigma_y = 2;
double gaussian( double x, double y)
{
    return  amplitude*
                   exp( -(double)((x-x00)*(x-x00)/2./sigma_x/sigma_x+
                                  (y-y00)*(y-y00)/2./sigma_y/sigma_y) );
}

//N should be 3*Nx*Ny
cudaGrahicsResource_t registerCudaOpenGLBuffer( unsigned N )
{
    int device;
    cudaGraphicsResource_t* resource;
    GLuint bufferID;
    cudaGetDevice( &device);
    cudaGLSetGLDevice(device ); 
    error = cudaGetLastError();
    if( error != cudaSuccess){
        cout << cudaGetErrorString( error); return 1;}
    glGenBuffers( 1, &bufferID);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, bufferID);
    // the buffer shall contain a texture 
    glBufferData( GL_PIXEL_UNPACK_BUFFER, N*sizeof(float), NULL, GL_DYNAMIC_DRAW);

    //register the resource i.e. tell CUDA and OpenGL that buffer is used by both
    error = cudaGraphicsGLRegisterBuffer( resource, bufferID, cudaGraphicsRegisterFlagsWriteDiscard); 
    if( error != cudaSuccess){
        cout << cudaGetErrorString( error); return 1;}
    return resource;
}


int main()
{
    HArrVec ne = expand< double(&)(double, double), n> ( gaussian, 0, lx, 0, ly, Nx, Ny);
    vector<DVec> y0(2, ne.data()), y1(y0);

    toefl::Matrix<double> visual( Nx, Ny, 0);
    for( unsigned i=0; i<Ny; i++)
        for( unsigned j=0; j<Nx; j++)
            visual(i,j) = ne(i,j, 0,0 );
    ////////////////////////////////glfw//////////////////////////////
    int running = GL_TRUE;
    if( !glfwInit()) { cerr << "ERROR: glfw couldn't initialize.\n";}
    if( !glfwOpenWindow( 300, 300,  0,0,0,  0,0,0, GLFW_WINDOW))
    { 
        cerr << "ERROR: glfw couldn't open window!\n";
    }
    glfwSetWindowTitle( "Texture test");
    //////////////////////////////////////////////////////////////////
    Texture_RGBf tex( Ny, Nx);
    glEnable(GL_TEXTURE_2D);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    while( running)
    {
        //generate a texture
        gentexture_RGBf( tex, visual, amplitude);
        glLoadIdentity();
        glClearColor(0.f, 0.f, 1.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT);
        // image comes from texarray on host
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex.cols(), tex.rows(), 0, GL_RGB, GL_FLOAT, tex.getPtr());
        glLoadIdentity();
        //Draw a textured quad
        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, -1.0);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0, 1.0);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0, 1.0);
        glEnd();
        glfwSwapBuffers();
        glfwWaitEvents();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    glfwTerminate();
    ////////////////////////////////////////////////////////////////////

    return 0;

}
