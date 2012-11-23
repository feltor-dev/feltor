/*! \file
 * \author Christian Knapp and Matthias Wiesenberger; colormap designed by Josef Peer
 * \date 14.03.2012
 *
 * Originally written for convection code and adapted for C++.
 */
#ifndef _TL_TEXTURE_
#define _TL_TEXTURE_
#include <iostream>
#include <float.h>

/*! @brief POD that contains RGB values for 256 colors*/
struct colormap_f
{
    float R[256]; //!< R[i] contains red value for color i
    float G[256]; //!< G[i] contains green value for color i
    float B[256]; //!< B[i] contains blue value for color i
};

/*! @brief Create a redblue colormap with 256 floats
    
    The values R[i], G[i], B[i] lead with increasing i from blue to red!
    @return A struct that contains 256 float-values for each red, green and blue
 */
colormap_f redblue_f()
{
    int i;
    colormap_f M;
    float scale = 1.0/64.0;
    
    for (i=0; i < 64; i++) {
        M.R[i] = 0.0;
        M.G[i] = 0.25*scale*(float)i;
        M.B[i] = 0.5 + (0.5*scale*(float)i);
    }
    for (i=0; i < 64; i++) {
        M.R[i+64] = scale*(float)i;
        M.G[i+64] = 0.25 + (0.75*scale*(float)i);
        M.B[i+64] = 1.0;
    }
    for (i=0; i < 64; i++) {
        M.R[i+128] = 1.0;
        M.G[i+128] = 1.0 - (scale*(float)i);
        M.B[i+128] = 1.0 - (scale*(float)i);
    }
    for (i=0; i < 64; i++) {
        M.R[i+192] = 1.0 - (0.5*scale*(float)i);
        M.G[i+192] = 0.0;
        M.B[i+192] = 0.0;
    }
    
    return M;
}


/*! @brief POD that contains RGB values for 384 colors*/
typedef struct colormap_ext
{
    float R[384];//!< R[i] contains red value for color i
    float G[384];//!< G[i] contains green value for color i
    float B[384];//!< B[i] contains blue value for color i
}colormap_ext;

/*! @brief Create an extended redblue colormap with 384 floats
    
    the extra colors are black beyond the blue and gold in the infrared
    @return A struct that contains 384 float-values for each red, green and blue
 */
colormap_ext redblue_ext()
{
    int i;
    colormap_ext M;
    float scale = 1.0/64.0;
    
    for (i=0; i < 64; i++) {
        M.R[i] = 0.0;
        M.G[i] = 0.0;
        M.B[i] = 0.5*scale*(float)i;
    }
    for (i=0; i < 64; i++) {
        M.R[i+64] = 0.0;
        M.G[i+64] = 0.25*scale*(float)i;
        M.B[i+64] = 0.5 + (0.5*scale*(float)i);
    }
    for (i=0; i < 64; i++) {
        M.R[i+128] = scale*(float)i;
        M.G[i+128] = 0.25 + (0.75*scale*(float)i);
        M.B[i+128] = 1.0;
    }
    for (i=0; i < 64; i++) {
        M.R[i+192] = 1.0;
        M.G[i+192] = 1.0 - (scale*(float)i);
        M.B[i+192] = 1.0 - (scale*(float)i);
    }
    for (i=0; i < 64; i++) {
        M.R[i+256] = 1.0 - (0.5*scale*(float)i);
        M.G[i+256] = 0.0;
        M.B[i+256] = 0.0;
    }
    for (i=0; i < 64; i++) {
        M.R[i+320] = 0.5 + (0.5*scale*(float)i);
        M.G[i+320] = scale*(float)i;
        M.B[i+320] = 0.0;
    }
    
    M.B[383] = 5.0;
    
    return M;
}

typedef Matrix<std::array<float,3>, TL_NONE> Texture_RGBf;


/*! @brief updates a texture with a given field for use of the glTexImage2D() function

    @param field a matrix containing the field to be plotted
    @param draw_h the height of the field
    @param draw_w the width of the field
    @param maxabs the absolute of the maximum value of the field
    @return a copy the static texture 
*/
template< class M>
void gentexture_RGBf( Texture_RGBf& tex, const M& field, const double maxabs)
{
    
    const static colormap_ext cm = redblue_ext(); // extended colormap
    
    double scalefact = 127/maxabs;
    int k, l, value;
    
    //store the min and max values of the field
    double min = DBL_MAX; // test range DBL_MAX is the maximal floating point value
    double max = (-1.)*DBL_MAX; // test range 
    
    for (int i=0; i < tex.height; i++)
        for (int j=0; j < tex.width; j++)
        {
            value = field[i][j];
            // test range
            if (min > value) min = value;
            if (max < value) max = value;
                
            k = (int)floor(scalefact*value) + 192; // +192 instead of +128 due to extended colormap
            k = k<0 ? 0 : ( k>383 ? 383 : k );
            
            l = 3 * (j + tex.width*i);
            tex.texarray[l]   = cm.R[k];
            tex.texarray[l+1] = cm.G[k];
            tex.texarray[l+2] = cm.B[k];
        }
    }
    //std::cout <<"Minimum / Maximum value:"  <<min << " "<< max;//test
    
    return tex;
}

/*! @brief updates a texture with a given field (special function for the temperature field)

    A texture struct and a colormap are statically allocated and reused at every entry to the function. 
    The maximum of the temperature field is known to be the rayleigh number
    @param draw_h the height of the temperature field
    @param draw_w the width of the temperature field
    @return a copy the static texture 
*/
void gentexture_RGBf_temp( Texture_RGBf& tex, const M& field, const double ray)
{
    static colormap_ext cm = redblue_ext(); // extended colormap
    
    double scalefact = 255./ray;
    double temp;
    int k, l;
    
    //double min = DBL_MAX; // test range
    //double max = (-1.)*DBL_MAX; // test range
    
    for (int i=0; i < tex.height; i++)
        for (int j=0; j < tex.width; j++)
        {
            // field.temp_0 = deltatemp, but we want absolute temp!
            temp = ( field.temp_0[i][j] + ray*(1.0-(double)i/(double)(grid.nz-1)) );
            
            k = ((int)(scalefact * temp)) + 64; // extended colormap
            k = k<0 ? 0 : ( k>383 ? 383 : k );
            
            l = 3 * (j + tex.width*i);
            tex.texarray[l]   = cm.R[k];
            tex.texarray[l+1] = cm.G[k];
            tex.texarray[l+2] = cm.B[k];
        }
    return;
}

#endif // _TL_TEXTURE_
