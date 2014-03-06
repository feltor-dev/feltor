#ifndef _TL_UTILITY_
#define _TL_UTILITY_
//DEPRECATED: use draw library instead!!

#include <GLFW/glfw3.h>
namespace toefl
{

/**
* @brief compute the absolute maximum of a given matrix
*
* @ingroup utilities
* @tparam M Matrix class
* @param field The matrix
*
* @return  The maximum value of the matrix
*/
template< class M>
double abs_max( const M& field)
{
    double temp = 0;
    for( unsigned i=0; i<field.rows(); i++)
        for( unsigned j=0; j<field.cols(); j++)
            if( fabs(field(i,j)) > temp) temp = fabs(field(i,j));
    return temp;
}

/**
* @brief draw a textured quad from a matrix on the screen
*
* @ingroup utilities
* @tparam M The matrix class
* @param field The field to be plotted
* @param max The value that corresponds to the highest value in the colormap
* @param x0 x-position of lower left corner
* @param x1 x-position of upper right corner
* @param y0 y-position of lower left corner
* @param y1 y-position of upper right corner
*/
template<class M>
void drawTexture( const M& field, double max, double x0, double x1, double y0, double y1)
{
    static Texture_RGBf tex( field.rows(), field.cols());
    gentexture_RGBf( tex, field, max);
    // image comes from texarray on host
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex.cols(), tex.rows(), 0, GL_RGB, GL_FLOAT, tex.getPtr());
    glLoadIdentity();
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f( x0, y0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( x1, y0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( x1, y1);
        glTexCoord2f(0.0f, 1.0f); glVertex2f( x0, y1);
    glEnd();
}
template<class M>
void drawTemperature( const M& field, double ray, double x0, double x1, double y0, double y1)
{
    static Texture_RGBf temp_tex( field.rows(), field.cols());
    gentexture_RGBf_temp( temp_tex, field, ray);
    // image comes from texarray on host
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, temp_tex.cols(), temp_tex.rows(), 0, GL_RGB, GL_FLOAT, temp_tex.getPtr());
    glLoadIdentity();
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f( x0, y0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( x1, y0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( x1, y1);
        glTexCoord2f(0.0f, 1.0f); glVertex2f( x0, y1);
    glEnd();
}
} //namespace toefl

#endif //_TL_UTILITY_
