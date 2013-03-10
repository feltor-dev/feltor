#ifndef _TL_UTILITY_
#define _TL_UTILITY_

#include <GL/glfw.h>
namespace toefl
{

template< class M>
double abs_max( const M& field)
{
    double temp = 0;
    for( unsigned i=0; i<field.rows(); i++)
        for( unsigned j=0; j<field.cols(); j++)
            if( fabs(field(i,j)) > temp) temp = fabs(field(i,j));
    return temp;
}

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
} //namespace toefl

#endif //_TL_UTILITY_
