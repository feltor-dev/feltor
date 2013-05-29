#ifndef _HOST_WIDNOW_H_
#define _HOST_WIDNOW_H_

#include <cassert>

#include <algorithm>  //transform 
#include <vector>
#include <sstream>
#include <GL/glfw.h>

#include "colormap.h"

//maybe in future Qt is an alternative
namespace draw
{

void GLFWCALL WindowResize( int w, int h)
{
    // map coordinates to the whole window
    glViewport( 0, 0, (GLsizei) w, h);
}

/**
 * @brief A window for 2d scientific plots 
 *
 * The intention of this class is to provide an interface to make 
 * the plot of a 2D vector during computations as simple as possible. 
 * To use it simply use
 * @code
 * draw::HostWindow w( 400, 400);
 * draw::ColorMapRedBlueExt map( 1.);
 * w.draw( v, 100, 100, map);
 * @endcode
 */
struct HostWindow
{
	/**
	 * @brief Open window
	 *
	 * @param width in pixels
	 * @param height in pixels
	 */
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
    /**
     * @brief Close window and OpenGL context
     */
    ~HostWindow() { glfwTerminate();}
    //Vector has to be useable in std functions
    /**
     * @brief Draw a 2D field in the open window
     *
     * @tparam Vector The container class of your elements
     * @param x Elements to be drawn
     * @param Nx # of x points to be used ( the width)
     * @param Ny # of y points to be used ( the height)
     * @param map The colormap used to compute color from elements
     */
    template< class Vector>
    void draw( const Vector& x, unsigned Nx, unsigned Ny, draw::ColorMapRedBlueExt& map)
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
        std::transform( x.begin(), x.end(), resource.begin(), map);
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
    template< class Vector>
    void draw( const Vector& x, unsigned Nx, unsigned Ny, draw::ColorMapRedBlueExt& map, unsigned i, unsigned j);
    std::stringstream& title() { return window_str;}
  private:
    HostWindow( const HostWindow&);
    HostWindow& operator=( const HostWindow&);
    unsigned Nx_, Ny_;
    std::vector<Color> resource;
    std::stringstream window_str;  //window name
};
} //namespace draw

#endif//_HOST_WIDNOW_H_
