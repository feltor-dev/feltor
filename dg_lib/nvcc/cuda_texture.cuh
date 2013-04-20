#ifndef _DG_CUDA_OPENGL_
#define _DG_CUDA_OPENGL_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <GL/glew.h>
#include <GL/glfw.h>
#include <cuda_gl_interop.h>

#include <thrust/transform.h>

#include "colormap.cuh"

void cudaGlfwInit( int width, int height)
{
    //initialize glfw
    if( !glfwInit()) { std::cerr << "ERROR: glfw couldn't initialize.\n";}
    int major, minor, rev;
    glfwGetVersion( &major, &minor, &rev);
    std::cout << "Using GLFW version   "<<major<<"."<<minor<<"."<<rev<<"\n";
    // create window and OpenGL context bound to it
    if( !glfwOpenWindow( width, height,  0,0,0,  0,0,0, GLFW_WINDOW))
    { 
        std::cerr << "ERROR: glfw couldn't open window!\n";
    } 
    std::cout << "Using OpenGL version "
        <<glfwGetWindowParam( GLFW_OPENGL_VERSION_MAJOR)<<"."
        <<glfwGetWindowParam( GLFW_OPENGL_VERSION_MINOR)<<"\n";
    //initialize glew
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
          /* Problem: glewInit failed, something is seriously wrong. */
        std::cerr << "Error: " << glewGetErrorString(err) << "\n";
        return;
    }
    std::cout << "Using GLEW version   " << glewGetString(GLEW_VERSION) <<"\n";

    int device;
    cudaGetDevice( &device);
    std::cout << "Using device number  "<<device<<"\n";
    cudaGLSetGLDevice( device ); 

    cudaError_t error;
    error = cudaGetLastError();
    if( error != cudaSuccess){
        std::cout << cudaGetErrorString( error);}

    glEnable(GL_TEXTURE_2D);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}
//N should be 3*Nx*Ny
cudaGraphicsResource* allocateCudaGlBuffer( unsigned N )
{
    cudaGraphicsResource* resource;
    GLuint bufferID;
    glGenBuffers( 1, &bufferID);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, bufferID);
    // the buffer shall contain a texture 
    glBufferData( GL_PIXEL_UNPACK_BUFFER, N*sizeof(float), NULL, GL_DYNAMIC_DRAW);

    //register the resource i.e. tell CUDA and OpenGL that buffer is used by both
    cudaError_t error;
    error = cudaGraphicsGLRegisterBuffer( &resource, bufferID, cudaGraphicsRegisterFlagsWriteDiscard); 
    if( error != cudaSuccess){
        std::cout << cudaGetErrorString( error); return NULL;}
    return resource;
}

void freeCudaGlBuffer( cudaGraphicsResource* resource)
{
    cudaGraphicsUnregisterResource( resource);
}


template< class ThrustVector>
void mapColors( cudaGraphicsResource* resource, const ThrustVector& x, const dg::ColorMapRedBlueExt& map)
{
    dg::Color* d_buffer;
    size_t size;
    //Map resource into CUDA memory space
    cudaGraphicsMapResources( 1, &resource, 0);
    // get a pointer to the mapped resource
    cudaGraphicsResourceGetMappedPointer( (void**)&d_buffer, &size, resource);
#ifdef DG_DEBUG
    assert( x.size() == size);
#endif //DG_DEBUG
    thrust::transform( x.begin(), x.end(), thrust::device_pointer_cast( d_buffer), map);
    //unmap the resource before OpenGL uses it
    cudaGraphicsUnmapResources( 1, &resource, 0);
}

void loadTexture( int rows, int cols)
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rows, cols, 0, GL_RGB, GL_FLOAT, NULL);
}

void drawQuad( double x0, double x1, double y0, double y1)
{
    // image comes from UNPACK_BUFFER on device
    glLoadIdentity();
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f( x0, y0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( x1, y0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( x1, y1);
        glTexCoord2f(0.0f, 1.0f); glVertex2f( x0, y1);
    glEnd();
}

#endif //_DG_CUDA_OPENGL_
