#ifndef _DG_CUDA_OPENGL_
#define _DG_CUDA_OPENGL_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG
#include <sstream>

#include <GL/glew.h>
#include <GL/glfw.h>
#include <cuda_gl_interop.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "colormap.cuh"

namespace dg
{

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
    //initialize glew (needed for GLbuffer allocation)
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

GLuint allocateGlBuffer( unsigned N)
{
    GLuint bufferID;
    glGenBuffers( 1, &bufferID);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, bufferID);
    // the buffer shall contain a texture 
    glBufferData( GL_PIXEL_UNPACK_BUFFER, N*sizeof(float), NULL, GL_DYNAMIC_DRAW);
    return bufferID;
}
//N should be 3*Nx*Ny
cudaGraphicsResource* allocateCudaGlBuffer( unsigned N )
{
    cudaGraphicsResource* resource;
    GLuint bufferID = allocateGlBuffer( N);
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

template< class T>
void mapColors( const dg::ColorMapRedBlueExt& map, const thrust::device_vector<T>& x, cudaGraphicsResource* resource)
{
    dg::Color* d_buffer;
    size_t size;
    //Map resource into CUDA memory space
    cudaGraphicsMapResources( 1, &resource, 0);
    // get a pointer to the mapped resource
    cudaGraphicsResourceGetMappedPointer( (void**)&d_buffer, &size, resource);
#ifdef DG_DEBUG
    assert( x.size() == size/3/sizeof(float));
#endif //DG_DEBUG
    thrust::transform( x.begin(), x.end(), thrust::device_pointer_cast( d_buffer), map);
    cudaThreadSynchronize();
    //unmap the resource before OpenGL uses it
    cudaGraphicsUnmapResources( 1, &resource, 0);
}

void loadTexture( int width, int height)
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
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

void GLFWCALL WindowResize( int w, int h)
{
    // map coordinates to the whole window
    glViewport( 0, 0, (GLsizei) w, h);
    //std::cout << "Resize\n";
    // map coordinates to the whole window
    //double win_ratio = (double)w/(double)h;
    //GLint ww = (win_ratio<field_ratio) ? w : h*field_ratio ;
    //GLint hh = (win_ratio<field_ratio) ? w/field_ratio : h;
    //glViewport( 0, 0, (GLsizei) ww, hh);
    //width = w;
    //height = h;
}

struct Window
{
    Window( int width, int height) { 
        resource = NULL;
        Nx_ = Ny_ = 0;
        bufferID = 0;
        cudaGlfwInit( width, height);
        glClearColor( 0.f, 0.f, 1.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSetWindowSizeCallback( WindowResize);
    }
    ~Window( ) {
        if( resource != NULL){
            cudaGraphicsUnregisterResource( resource); 
            //free the opengl buffer
            glDeleteBuffers( 1, &bufferID);
        }
        //terminate glfw
        glfwTerminate();
    }
    template< class T>
    void draw( const thrust::device_vector<T>& x, unsigned Nx, unsigned Ny, dg::ColorMapRedBlueExt& map)
    {
        if( Nx != Nx_ || Ny != Ny_) {
            Nx_ = Nx; Ny_ = Ny;
            cudaGraphicsUnregisterResource( resource);
            std::cout << "Allocate resources for drawing!\n";
            //free opengl buffer
            GLint id; 
            glGetIntegerv( GL_PIXEL_UNPACK_BUFFER_BINDING, &id);
            bufferID = (GLuint)id;
            glDeleteBuffers( 1, &bufferID);
            //allocate new buffer
            resource = allocateCudaGlBuffer( 3*Nx*Ny);
            glGetIntegerv( GL_PIXEL_UNPACK_BUFFER_BINDING, &id);
            bufferID = (GLuint)id;
        }

        mapColors( map, x, resource);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, Nx, Ny, 0, GL_RGB, GL_FLOAT, NULL);
        glLoadIdentity();
        float x0 = -1, x1 = 1, y0 = -1, y1 = 1;
        glBegin(GL_QUADS); 
            glTexCoord2f(0.0f, 0.0f); glVertex2f( x0, y0);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( x1, y0);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( x1, y1);
            glTexCoord2f(0.0f, 1.0f); glVertex2f( x0, y1);
        glEnd();
        glfwSwapBuffers();
    }
  private:
    Window( const Window&);
    Window& operator=( const Window&);
    GLuint bufferID;
    cudaGraphicsResource* resource;  
    unsigned Nx_, Ny_;
};

struct HostWindow
{
    HostWindow( int width, int height){
        Nx_ = Ny_ = 0;
        // create window and OpenGL context bound to it
        if( !glfwInit()) { std::cerr << "ERROR: glfw couldn't initialize.\n";}
        if( !glfwOpenWindow( width, height,  0,0,0,  0,0,0, GLFW_WINDOW))
        { 
            std::cerr << "ERROR: glfw couldn't open window!\n";
        }
        glfwSetWindowSizeCallback( WindowResize);
        int major, minor, rev;
        glfwGetVersion( &major, &minor, &rev);
        std::cout << "Using GLFW version   "<<major<<"."<<minor<<"."<<rev<<"\n";
        //enable textures
        glEnable(GL_TEXTURE_2D);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        window_str << "Host Window\n";
    }
    ~HostWindow() { glfwTerminate();}
    template< class T>
    void draw( const thrust::host_vector<T>& x, unsigned Nx, unsigned Ny, dg::ColorMapRedBlueExt& map)
    {
        //gehört das hier rein??
        glfwSetWindowTitle( (window_str.str()).c_str() );
        window_str.str(""); //clear title string
        glClear(GL_COLOR_BUFFER_BIT);
        if( Nx != Nx_ || Ny != Ny_) {
            Nx_ = Nx; Ny_ = Ny;
            std::cout << "Allocate resources for drawing!\n";
            resource.resize( Nx*Ny);
        }
#ifdef DG_DEBUG
        assert( x.size() == resource.size());
#endif //DG_DEBUG
        //map colors
        thrust::transform( x.begin(), x.end(), resource.begin(), map);
        //load texture
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, Nx, Ny, 0, GL_RGB, GL_FLOAT, resource.data());
        glLoadIdentity();
        float x0 = -1, x1 = 1, y0 = -1, y1 = 1;
        glBegin(GL_QUADS); 
            glTexCoord2f(0.0f, 0.0f); glVertex2f( x0, y0);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( x1, y0);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( x1, y1);
            glTexCoord2f(0.0f, 1.0f); glVertex2f( x0, y1);
        glEnd();
        glfwSwapBuffers();
    }
    void set_multiplot( unsigned i, unsigned j);
    template< class T>
    void draw( const thrust::host_vector<T>& x, unsigned Nx, unsigned Ny, dg::ColorMapRedBlueExt& map, unsigned i, unsigned j);
    std::stringstream& title() { return window_str;}
  private:
    HostWindow( const HostWindow&);
    HostWindow& operator=( const HostWindow&);
    unsigned Nx_, Ny_;
    thrust::host_vector<Color> resource;
    std::stringstream window_str;  //window name
};

}

#endif //_DG_CUDA_OPENGL_
