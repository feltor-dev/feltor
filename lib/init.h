/*! \file
 * @brief Implementation of some matrix initialisation routines 
 * @author Matthias Wiesenberger
 *  Matthias.Wiesenberger@uibk.ac.at
 */
#ifndef _TL_INIT_
#define _TL_INIT_

#include <time.h>
#include <math.h>

namespace toefl
{
/*! @brief Adds a gaussian to a given matrix
 *
 * The function interprets the given points as inner, cell centered points of a 
 * square box. [0,1]x[0,1]
 * , where the first index is the y and the second index is the x point. 
 * (0,0) corresponds  to the lower left corner.
 * It adds the values of the smooth function
 * \f[
   f(x,y) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}} 
   \f]
   where A is a constant and \f$ x,y = 0...1 \f$.
 * \param m the matrix
 * @param x0 x-position of maximum 0<x0<1
 * @param y0 y-position of maximum 0<y0<1
 * @param sigma_x Varianz in x (FWHM = 2.35*sigma_x)
 * @param sigma_y Varianz in y (FWHM = 2.35*sigma_y)
 * @param amplitude Value of maximum
 */
template< class M>
void init_gaussian( M& m, const double x0, const double y0, 
                          const double sigma_x, const double sigma_y,
                          const double amplitude)
{
    const size_t rows = m.rows(), cols = m.cols();
    const double hx = 1./(double)(cols), hy = 1./(double)(rows); 
    double x,y;
    for( unsigned i=0; i<rows; i++)
        for( unsigned j=0; j<cols; j++)
        {
            x = (j+0.5)*hx;
            y = (i+0.5)*hy;
            m(i,j) += amplitude*
                   exp( -(double)((x-x0)*(x-x0)/2./sigma_x/sigma_x+
                                  (y-y0)*(y-y0)/2./sigma_y/sigma_y) );
        }
}

/*! @brief Adds a gaussian column to a given matrix
 *
 * The function interprets the given points as inner, cell centered points of a 
 * square box. [0,1]x[0,1]
 * , where the first index is the y and the second index is the x point. 
 * (0,0) corresponds  to the lower left corner.
 * It adds the values of the smooth function
 * \f[
   f(x,y) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2} }
   \f]
   where A is a constant and \f$ x,y = 0...1 \f$.
 * \param m the matrix
 * @param x0 x-position of maximum 0<x0<1
 * @param sigma_x Varianz in x (FWHM = 2.35*sigma_x)
 * @param amplitude Value of maximum
 */
template< class M>
void init_gaussian_column( M& m, const double x0, const double sigma_x, const double amplitude)
{
    const size_t rows = m.rows(), cols = m.cols();
    const double hx = 1./(double)(cols); 
    double x;
    for( unsigned i=0; i<rows; i++)
        for( unsigned j=0; j<cols; j++)
        {
            x = (j+0.5)*hx;
            m(i,j) += amplitude*
                   exp( -(double)((x-x0)*(x-x0)/2./sigma_x/sigma_x));
        }
}
        



/*! @brief Functor for turbulent bath coefficients
 * 
 */
class TurbulentBath{
public:
    /*! @brief Init amplitude
     * 
     * Inits the random number generator
     * @param amp The fourier amplitude
     */
    TurbulentBath( double amp):amplitude(amp){srand(time(0));}
    /*! @brief Generate a fourier amplitude
     * 
     * The function is
     * \f[
       f(k_x, k_y) = z(k_x, k_y)\frac{A}{\sqrt{1+3.125(k_x^2 + k_y^2)^4}}
       \f]
       where z is a random number between 0 and 1.
     * @param laplace The laplacian in fourier space
     * @return The function value
     */
    double operator()(const double laplace)
    {
        double zuf = (double)rand();
        return cos(zuf)*amplitude/sqrt( 1.+pow(laplace,4));
    }
private:
    const double amplitude;

};

} // namespace toefl

#endif // _TL_INIT_
