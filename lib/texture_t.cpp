#include <iostream>
#include <GL/glfw.h>
#include <cmath>
#include "texture.h"
#include "matrix.h"

const double R = 500;
const unsigned nz = 10;
const unsigned nx = 10;

//Draws a simple textured square in an open window 

using namespace std;
int main()
{
    
    toefl::Matrix<double> field( nz, nx);

    field.zero();
    ////////////////////////////////glfw//////////////////////////////
    int running = GL_TRUE;
    if( !glfwInit()) { cerr << "ERROR: glfw couldn't initialize.\n";}
    if( !glfwOpenWindow( 300, 300,  0,0,0,  0,0,0, GLFW_WINDOW))
    { 
        cerr << "ERROR: glfw couldn't open window!\n";
    }
    glfwSetWindowTitle( "Texture test");
    //////////////////////////////////////////////////////////////////
    Texture_RGBf tex( nz, nx);
    int scale_z = 1.0;
    glEnable(GL_TEXTURE_2D);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    while( running)
    {
        //generate a texture
        gentexture_RGBf_temp( tex, field, R);
        glLoadIdentity();
        glClearColor(0.f, 0.f, 0.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT);
        // image comes from texarray on host
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex.cols(), tex.rows(), 0, GL_RGB, GL_FLOAT, tex.getPtr());
        glLoadIdentity();
        //Draw a textured quad
        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0*scale_z);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, -1.0*scale_z);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0,  scale_z);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0,  scale_z);
        glEnd();
        glfwSwapBuffers();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    glfwTerminate();
    //////////////////////////////////////////////////////////////////
    return 0;
}
