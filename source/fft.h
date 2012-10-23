#ifndef _TL_FFT_
#define _TL_FFT_

#include "matrix.h"
#include "fftw3.h"

namespace toefl{

template <typename Complex>
class FFT_MANY_1D
{
  private:
    fftw_plan forward_plan;
    fftw_plan backward_plan;
#ifdef TL_DEBUG
    size_t nrows;
    size_t nelements;
#endif
  public:
    FFT_MANY_1D( Matrix<double, TL_FFT_1D>& m)
    {
#ifdef TL_DEBUG
        nrows = m.rows();
        nelements = TotalNumberOf<TL_FFT_1D>::elements( m.rows(), m.cols());
        if(m.isVoid())
            throw Message( "Cannot initialize a plan for a void Matrix!\n", ping);
#endif
        int n[] = { (int)m.cols()}; //length of each transform
        forward_plan = fftw_plan_many_dft_r2c(  1,  //dimension 1D
                                    n,  //size of each dimension
                                    m.rows(), //number of transforms
                                    &m(0,0), //input
                                    NULL, //embed
                                    1, //stride in units of double
                                    TotalNumberOf<TL_FFT_1D>::cols( m.cols()), //distance between trafos
                                    reinterpret_cast<fftw_complex*>(&m(0,0)),
                                    NULL,
                                    1, //stride in units of fftw_complex
                                    m.cols()/2 + 1, //distance between trafos
                                    FFTW_MEASURE);

        backward_plan = fftw_plan_many_dft_c2r(  1,  //dimension 1D
                                    n,  //size of each dimension (in complex)
                                    m.rows(), //number of transforms
                                    reinterpret_cast<fftw_complex*>(&m(0,0)), //input
                                    NULL, //embed
                                    1, //stride in units of complex
                                    m.cols()/2 + 1, //distance between trafos (in complex)
                                    &m(0,0),
                                    NULL,
                                    1, //stride in units of double
                                    TotalNumberOf<TL_FFT_1D>::cols(m.cols()), //distance between trafos (in double)
                                    FFTW_MEASURE);
    }
    //the address of m(0,0) might have changed due to the swap routine
    void r2c( Matrix<double, TL_FFT_1D>& m)
    { 
#ifdef TL_DEBUG
        if(nrows != m.rows() || nelements != TotalNumberOf<TL_FFT_1D>::elements( m.rows(), m.cols())) 
            throw Message( "Size doesn't match size of plan\n", ping);
        if(m.isVoid())
            throw Message( "Cannot use plan on a void Matrix!\n", ping);
#endif
        fftw_execute_dft_r2c( forward_plan, &m(0,0), reinterpret_cast<fftw_complex*>(&m(0,0)));
    }

    void c2r( Matrix<Complex, TL_NONE>& m)
    { 
#ifdef TL_DEBUG
        if(nrows != m.rows() || nelements != 2*m.cols()*m.rows()) 
            throw Message( "Size doesn't match size of plan\n", ping);
        if(m.isVoid())
            throw Message( "Cannot use plan on a void Matrix!\n", ping);
#endif
        fftw_execute_dft_c2r( backward_plan, reinterpret_cast<fftw_complex*>(&m(0,0)), reinterpret_cast<double*>(&m(0,0)));
    }

    ~FFT_MANY_1D()
    {
        fftw_destroy_plan( forward_plan);
        fftw_destroy_plan( backward_plan);

    }

};


class FFT_MANY_1D_SINE
{
  private:
    fftw_plan plan;
#ifdef TL_DEBUG
    size_t nrows;
    size_t nelements;
#endif
  public:
    FFT_MANY_1D_SINE( Matrix<double, TL_NONE>& m)
    {
#ifdef TL_DEBUG
        nrows = m.rows();
        nelements = m.rows()*m.cols();
        if(m.isVoid())
            throw Message( "Cannot initialize a plan for a void Matrix!\n", ping);
#endif
        int n[] = { (int)m.cols()}; //length of each transform
        fftw_r2r_kind kind[] = {FFTW_RODFT00};
        plan = fftw_plan_many_r2r(  1,  //dimension 1D
                                    n,  //size of each dimension
                                    m.rows(), //number of transforms
                                    &m(0,0), //input
                                    NULL, //embed
                                    1, //stride in units of double
                                    TotalNumberOf<TL_NONE>::cols( m.cols()), //distance between trafos
                                    &m(0,0), //output array (the same)
                                    NULL,
                                    1, //stride in units of double
                                    TotalNumberOf<TL_NONE>::cols( m.cols()), //distance between trafos
                                    kind, //odd around j = -1 and j = n
                                    FFTW_MEASURE);
    }

    void r2r( Matrix<double, TL_NONE>& m)
    { 
#ifdef TL_DEBUG
        if(nrows != m.rows() || nelements != m.cols()*m.rows()) 
            throw Message( "Size doesn't match size of plan\n", ping);
        if(m.isVoid())
            throw Message( "Cannot use plan on a void Matrix!\n", ping);
#endif
        fftw_execute_r2r( plan, &m(0,0), &m(0,0));
    }

    ~FFT_MANY_1D_SINE()
    {
        fftw_destroy_plan( plan);
    }

};

}
#endif //_TL_FFT_
