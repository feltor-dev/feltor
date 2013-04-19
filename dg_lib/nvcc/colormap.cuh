/*! \file
 * \author Christian Knapp and Matthias Wiesenberger; colormap designed by Josef Peer
 * \date 14.03.2012
 *
 * Originally written for convection code and adapted for C++.
 */
#ifndef _DG_TEXTURE_
#define _DG_TEXTURE_

namespace dg{
/*! @addtogroup utilities
 * @{
 */

/*! @brief POD that contains RGB values for 256 colors*/
struct Color
{
    float r; //!< R contains red value for color 
    float g; //!< G contains green value for color 
    float b; //!< B contains blue value for color 
};

struct ColorMapRedBlueExt
{
    ColorMapRedBlueExt( float scale = 1.);
    //maps [-scale, scale] to a color
    __host__ __device__
    Color operator()( float x);
    float& scale( ) { return scale_;}
    float scale() const { return scale_;} 
  private:
    float scale_; 
    Color M[384];
};


/*! @brief Create an extended redblue colormap with 384 floats
    
    the extra colors are black beyond the blue and gold in the infrared
 */
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
__host__ __device__ Color ColorMapRedBlueExt::operator()( float x)
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


///@}

} //namespace dg 
#endif // _DG_TEXTURE_
