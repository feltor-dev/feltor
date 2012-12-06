#include <iostream>
#include <iomanip>
#include "dft_dft.h"



using namespace std;
using namespace toefl;

//why there are two coefficients in y direction
/*
 * The r2c yields one coefficient for a mode
 * The c2c trafo afterwards yields two: at k and N-k for one mode.
 */
unsigned rows = 7, cols = 10;
complex<double> I = {0,1};

/*! @brief Adds a gaussian to a given matrix
 *
 * The function interprets the given points as inner, cell centered points of a 
 * square box. [0,1]x[0,1]
 * , where the first index is the y and the second index is the x point. 
 * (0,0) corresponds  to the lower left corner.
 * It adds the values of the smooth function
 * \f[
   f(x) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}} 
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
int main()
{
    Matrix<double, TL_DFT> m1{rows, cols};
    Matrix<complex<double> >   m1_{ rows, cols/2 + 1}, m1_der{m1_};
    DFT_DFT dft_dft( rows,cols);
    double dx = 1./(double)cols, dy = 1./(double)rows;
    init_gaussian( m1, 0.5,0.5, 0.2,0.2, 1);
    //for( size_t i = 0; i < m1.rows(); i++)
    //    for ( size_t j=0; j < m1.cols(); j++)
    //        m1(i, j) = (double)j*dx; //f(x,y) = y;
    cout << setprecision(2) << fixed;
    cout <<"The original matrix\n"<<m1<<endl;
    dft_dft.r2c( m1, m1_);
    cout << "The transformed matrix\n"<<m1_<<endl;
    //multiply coefficients
    int ik;
    for( unsigned i=0; i<rows; i++)
        for( unsigned j=0; j<cols/2+1; j++)
        {
            ik = (i>rows/2)?(i-rows):i;
            m1_der(i,j) = {0, 2.*M_PI*ik};
            m1_der(i,j) *=m1_(i,j)/(double)(rows*cols);
            m1_(i,j) *= -(double)(ik*ik+j*j)*2.*M_PI*2.*M_PI/(double)(rows*cols);
        }
    m1_(rows-1, cols/2) = {0,0};
    cout << "The multiplied matrix\n"<<m1_<<endl;
    cout << "The multiplied matrix\n"<<m1_der<<endl;
    try{
        dft_dft.c2r( m1_, m1);
    }catch( Message& m){m.display();}
    cout << "The laplacian of the original\n"<<m1<<endl;
    try{
        dft_dft.c2r( m1_der, m1);
    }catch( Message& m){m.display();}
    cout << "The y-deriv of the original\n"<<m1<<endl;




    return 0;
}
