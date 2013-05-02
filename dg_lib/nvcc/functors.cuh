#ifndef _DG_FUNCTORS_CUH_
#define _DG_FUNCTORS_CUH_


namespace dg
{
struct Gaussian
{
    Gaussian( double x0, double y0, double sigma_x, double sigma_y, double amp)
        : x00(x0), y00(y0), sigma_x(sigma_x), sigma_y(sigma_y), amplitude(amp){}
    double operator()(double x, double y)
    {
        return  amplitude*
                   exp( -((x-x00)*(x-x00)/2./sigma_x/sigma_x +
                          (y-y00)*(y-y00)/2./sigma_y/sigma_y) );
    }
  private:
    double  x00, y00, sigma_x, sigma_y, amplitude;

};

}

#endif //_DG_FUNCTORS_CUH
