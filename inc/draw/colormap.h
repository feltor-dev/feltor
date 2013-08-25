/*! \file
 * \author Christian Knapp and Matthias Wiesenberger; colormap designed by Josef Peer
 * \date 14.03.2012
 *
 * Originally written for convection code and adapted for C++.
 */
#ifndef _DG_TEXTURE_
#define _DG_TEXTURE_

#include <cmath>

namespace draw{

/*! @brief POD that contains RGB values for 256 colors*/
struct Color
{
    float r; //!< R contains red value for color 
    float g; //!< G contains green value for color 
    float b; //!< B contains blue value for color 
};

/**
 * @brief A Colormap from black - blue  over white to red and gold
 */
struct ColorMapRedBlueExt
{
/*! @brief Create an extended redblue colormap
    
    the extra colors are black beyond the blue and gold in the infrared
    @param scale The scale specifies which value corresponds to red
     	(-scale corresponds to blue, 0 to white)
 */
    ColorMapRedBlueExt( float scale = 1.);
    //maps [-scale, scale] to a color
    /**
     * @brief map a value to a color
     *
     * @param x value
     *
     * @return corresponding color according to the colormap
     */
    Color operator()( float x);
    /**
     * @brief Set the scale
     *
     * @return reference to the scale value
     */
    float& scale( ) { return scale_;}
    /**
     * @brief The scale currently in use
     *
     * @return The scale currently in use
     */
    float scale() const { return scale_;} 
  private:
    float scale_; 
    Color M[384];
};


ColorMapRedBlueExt::ColorMapRedBlueExt( float scale): scale_(scale)
{
    float scal = 1.0/64.0;
    for ( int i=0; i < 64; i++) {
        M[i].r = 0.0;
        M[i].g = 0.0;
        M[i].b = 0.5*scal*(float)i;
    }
    for ( int i=0; i < 64; i++) {
        M[i+64].r = 0.0;
        M[i+64].g = 0.25*scal*(float)i;
        M[i+64].b = 0.5 + (0.5*scal*(float)i);
    }
    for ( int i=0; i < 64; i++) {
        M[i+128].r = scal*(float)i;
        M[i+128].g = 0.25 + (0.75*scal*(float)i);
        M[i+128].b = 1.0;
    }
    for ( int i=0; i < 64; i++) {
        M[i+192].r = 1.0;
        M[i+192].g = 1.0 - (scal*(float)i);
        M[i+192].b = 1.0 - (scal*(float)i);
    }
    for ( int i=0; i < 64; i++) {
        M[i+256].r = 1.0 - (0.5*scal*(float)i);
        M[i+256].g = 0.0;
        M[i+256].b = 0.0;
    }
    for ( int i=0; i < 64; i++) {
        M[i+320].r = 0.5 + (0.5*scal*(float)i);
        M[i+320].g = scal*(float)i;
        M[i+320].b = 0.0;
    }
    M[383].b = 5.0;
}


// Maps scale to Red and -scale to Blue, > scale to gold and < -scale to black
//on device direct evaluation is faster (probably map remains on host??)
/*
Color ColorMapRedBlueExt::operator()( float x)
{
    float scalefact = 127./scale_;
    x = scalefact*x + 192; // +192 instead of +128 due to extended colormap
    x = x<0 ? 0 : ( x>383 ? 383 : x ); //clip values
    Color c;
    x/= 64.;
    if( x < 1.)        { c.r = 0.; c.g = 0.; c.b = 0.5*x;}
    else if( x < 2. ) { x-= 1.; c.r = 0.; c.g = 0.25*x; c.b = 0.5 + (0.5*x);}
    else if( x < 3. ) { x-= 2.; c.r = x; c.g = 0.25 + 0.75*x; c.b = 1.;}
    else if( x < 4. ) { x-= 3.; c.r = 1.0; c.g = 1.0 - x; c.b = 1. - x;}
    else if( x < 5. ) { x-= 4.; c.r = 1.0 - 0.5*x; c.g = 0.; c.b = 0.;}
    else if( x < 6. ) { x-= 5.; c.r = 0.5 + 0.5*x; c.g = x; c.b = 0.;}
    return c;
}
*/
Color ColorMapRedBlueExt::operator()( float x)
{
    Color c;
    float scalefact = 127./scale_;
    int k;
    k = (int)floor(scalefact*x) + 192; // +192 instead of +128 due to extended colormap
    k = k<0 ? 0 : ( k>383 ? 383 : k ); //clip values
    c.r = M[k].r;
    c.g = M[k].g;
    c.b = M[k].b;
    return c;
}



} //namespace draw
#endif // _DG_TEXTURE_
