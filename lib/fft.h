#ifndef _TL_FFT_
#define _TL_FFT_

#include <complex>
#include "matrix.h"
#include "fftw3.h"
#include <algorithm>

/* fftw guru interface
 *
 * fftw_iodim{int n,  // Größe der Dimension des Index 
 *            int is, //in stride
 *            int os} //out strid
 * rank, fftw_iodim dims[rank] //describe how you come to the next point inside a trafo for every index i.e. dims[0] describes the first index of the matrix m[i0][i1]...[i_rank-1]
 * howmany_rank, fftw_iodim howmany_dims[howmany_rank] //describe how you come to the first point of the next trafo
 */
// was nicht geprüft wird ist, ob der Plan in der execute Funktion den richtigen "Typ" hat und (evtl macht das die fftw selbst)
// ob der Plan für die Größe der Matrix passt (das macht die fftw aber auch nicht)
namespace toefl{

    //1d and 2d r2r plans and execute functions
    fftw_plan plan_dst_1d( Matrix<double, TL_NONE>& inout); //implemented
    fftw_plan plan_dst_1d( Matrix<double, TL_NONE>& in, Matrix<double, TL_NONE>& out);
    fftw_plan plan_dst_2d( Matrix<double, TL_NONE>& inout); 
    fftw_plan plan_dst_2d( Matrix<double, TL_NONE>& in, Matrix<double, TL_NONE>& out); 

    void execute_dst_1d( const fftw_plan plan, Matrix<double, TL_NONE>& inout); //implemented
    void execute_dst_1d( const fftw_plan plan, Matrix<double, TL_NONE>& in, Matrix<double, TL_NONE>& out);
    void execute_dst_2d( const fftw_plan plan, Matrix<double, TL_NONE>& inout);
    void execute_dst_2d( const fftw_plan plan, Matrix<double, TL_NONE>& in, Matrix<double, TL_NONE>& out);

    //1d dft_c2c plans and execute functions
    fftw_plan plan_dft_1d( Matrix<std::complex<double>, TL_NONE>& inout); 
    fftw_plan plan_dft_1d( Matrix<std::complex<double>, TL_NONE>& in, Matrix<std::complex<double>, TL_NONE>& out);

    void execute_dft_1d( const fftw_plan plan, Matrix<std::complex<double>, TL_NONE>& inout); 
    void execute_dft_1d( const fftw_plan plan, Matrix<std::complex<double>, TL_NONE>& in, Matrix<std::complex<double>, TL_NONE>& out);

    //1d dft_r2c plans and execute functions
    fftw_plan plan_dft_1d_r2c( Matrix<double, TL_DFT_1D>& inout);//implemented
    template< typename Complex>
    fftw_plan plan_dft_1d_r2c( Matrix<double, TL_NONE>& in, Matrix<Complex, TL_NONE>& out);

    template< typename Complex>
    void execute_dft_1d_r2c( const fftw_plan plan, Matrix< double, TL_DFT_1D>& inout, Matrix< Complex, TL_NONE>& swap);//implemented
    template< typename Complex>
    void execute_dft_1d_r2c( const fftw_plan plan, Matrix< double, TL_NONE>& in, Matrix< Complex, TL_NONE>& out);

    //1d dft_c2r plans and execute functions
    template< typename Complex>
    fftw_plan plan_dft_1d_c2r( Matrix<Complex, TL_NONE>& inout,  bool odd); //init with n%2//implemented
    template< typename Complex>
    fftw_plan plan_dft_1d_c2r( Matrix<Complex, TL_NONE>& in,     Matrix<double, TL_NONE>& out); 

    template< typename Complex>
    void execute_dft_1d_c2r( const fftw_plan plan, Matrix< Complex, TL_NONE>& inout, Matrix< double, TL_DFT_1D>& swap);//implemented
    template< typename Complex>
    void execute_dft_1d_c2r( const fftw_plan plan, Matrix< Complex, TL_NONE>& in,    Matrix< double, TL_NONE>& out);
    
    //2d dft_r2c plans and execute functions
    fftw_plan plan_dft_2d_r2c( Matrix<double, TL_DFT_2D>& inout);
    template< typename Complex>
    fftw_plan plan_dft_2d_r2c( Matrix<double, TL_NONE>& in, Matrix<Complex, TL_NONE>& out);

    template< typename Complex>
    void execute_dft_2d_r2c( const fftw_plan plan, Matrix< double, TL_DFT_2D>& inout, Matrix< Complex, TL_NONE>& swap );
    template< typename Complex>
    void execute_dft_2d_r2c( const fftw_plan plan, Matrix< double, TL_NONE>& in, Matrix< Complex, TL_NONE>& out);

    //2d dft_c2r plans and execute functions
    template< typename Complex>
    fftw_plan plan_dft_2d_c2r( Matrix<Complex, TL_NONE>& inout, bool odd);
    template< typename Complex>
    fftw_plan plan_dft_2d_c2r( Matrix<Complex, TL_NONE>& in, Matrix<double, TL_NONE>& out);

    template< typename Complex>
    void execute_dft_2d_c2r( const fftw_plan plan, Matrix< Complex, TL_NONE>& inout, Matrix< double, TL_DFT_2D>& swap);
    template< typename Complex>
    void execute_dft_2d_c2r( const fftw_plan plan, Matrix< Complex, TL_NONE>& in, Matrix< double, TL_NONE>& out);
    

/////////////////////Definitions/////////////////////////////////////////////////////////
    
    
    fftw_plan plan_dst_1d( Matrix<double, TL_NONE>& m)
    {
#ifdef TL_DEBUG
        if(m.isVoid())
            throw Message( "Cannot initialize a plan for a void Matrix!\n", ping);
#endif
        int n[] = { (int)m.cols()}; //length of each transform
        fftw_r2r_kind kind[] = {FFTW_RODFT00};
        fftw_plan plan = fftw_plan_many_r2r(  1,  //dimension 1D(rank)
                                    n,  //size of each dimension (# of elements)
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
        return plan;
    }

    void execute_dst_1d( const fftw_plan plan, Matrix<double, TL_NONE>& m)
    {
#ifdef TL_DEBUG
        if( m.isVoid() == true)
            throw Message("Matrix is void!\n", ping);
#endif
        fftw_execute_r2r( plan, &m(0,0), &m(0,0));
    }



    fftw_plan plan_dft_1d_r2c( Matrix<double, TL_DFT_1D>& m)
    {
#ifdef TL_DEBUG
        if(m.isVoid())
            throw Message( "Cannot initialize a plan for a void Matrix!\n", ping);
#endif
        int n[] = { (int)m.cols()}; //length of each transform
        fftw_plan plan = fftw_plan_many_dft_r2c(  1,  //dimension 1D
                                    n,  //size of each dimension
                                    m.rows(), //number of transforms
                                    &m(0,0), //input
                                    NULL, //embed
                                    1, //stride in units of double
                                    TotalNumberOf<TL_DFT_1D>::cols( m.cols()), //distance between trafos
                                    reinterpret_cast<fftw_complex*>(&m(0,0)),
                                    NULL,
                                    1, //stride in units of fftw_complex
                                    m.cols()/2 + 1, //distance between trafos
                                    FFTW_MEASURE);
        return plan;
    }

    template< typename Complex>
    void execute_dft_1d_r2c( const fftw_plan plan, Matrix< double, TL_DFT_1D>& m, Matrix< Complex, TL_NONE>& swap)
    {
#ifdef TL_DEBUG
        if( swap.isVoid() == false)
            throw Message( "Swap matrix is not void!\n", ping);
        if(m.isVoid())
            throw Message( "Cannot use plan on a void Matrix!\n", ping);
#endif
        fftw_execute_dft_r2c( plan, &m(0,0), reinterpret_cast<fftw_complex*>(&m(0,0)));
        swap_fields( m, swap); //checkt, wenn swap nicht geht
    }

    template< typename Complex>
    fftw_plan plan_dft_1d_c2r( Matrix<Complex, TL_NONE>& m,  bool odd)
    {
#ifdef TL_DEBUG
        if(m.isVoid())
            throw Message( "Cannot initialize a plan for a void Matrix!\n", ping);
#endif
        int n[] ={2*(int)m.cols() - ((odd==true)?1:2) }; //{ (int)m.cols()};  //length of each transform (double)
        fftw_plan plan = fftw_plan_many_dft_c2r(  1,  //dimension 1D
                                    n,  //size of each dimension (in complex)
                                    m.rows(), //number of transforms
                                    reinterpret_cast<fftw_complex*>(&m(0,0)), //input
                                    NULL, //embed
                                    1, //stride in units of complex
                                    m.cols(), //distance between trafos (in complex)
                                    reinterpret_cast<double*>(&m(0,0)),
                                    NULL,
                                    1, //stride in units of double
                                    2*(int)m.cols(), //distance between trafos (in double)
                                    FFTW_MEASURE);
        return plan;
    }

    template< typename Complex>
    void execute_dft_1d_c2r( const fftw_plan plan, Matrix< Complex, TL_NONE>& m, Matrix< double, TL_DFT_1D>& swap)
    {
#ifdef TL_DEBUG
        if( swap.isVoid() == false)
            throw Message( "Swap matrix is not void!\n", ping);
        if(m.isVoid())
            throw Message( "Cannot use plan on a void Matrix!\n", ping);
#endif
        fftw_execute_dft_c2r( plan, reinterpret_cast<fftw_complex*>(&m(0,0)), reinterpret_cast<double*>(&m(0,0)));
        swap_fields( m, swap);
    }


}
#endif //_TL_FFT_
