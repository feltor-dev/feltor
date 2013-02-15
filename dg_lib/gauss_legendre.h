#ifndef _DG_GAUSS_LEGENDRE_
#define _DG_GAUSS_LEGENDRE_

//obsolete (now in DLT in dlt.h)
namespace dg
{

template< size_t n>
struct Weights_and_Abscissas
{
    const static double weight[n];
    const static double abscissa[n];
};

//values taken from http://processingjs.nihongoresources.com/bezierinfo/legendre-gauss-values.php
const double Weights_and_Abscissas<2>::weight[2] = 
{ 1., 1.}
const double Weights_and_Abscissas<2>::abscissa[2] = 
{ -0.5773502691896257, 0.5773502691896257 };
const double Weights_and_Abscissas<3>::weight = 
{ 0.5555555555555556, 0.8888888888888888, 0.5555555555555556};
const double Weights_and_Abscissas<3>::abscissa = 
{-0.7745966692414834, 0.0, 0.7745966692414834 };
const double Weights_and_Abscissas<4>::weight = 
{ 0.3478548451374538, 0.6521451548625461, 0.6521451548625461,0.3478548451374538} ;
const double Weights_and_Abscissas<4>::abscissa = 
{ -0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
const double Weights_and_Abscissas<5>::weight = 
{0.2369268850561891,    0.4786286704993665,0.5688888888888889,0.4786286704993665,0.2369268850561891};
const double Weights_and_Abscissas<5>::abscissa = 
{ -0.9061798459386640,-0.5384693101056831,0.0,0.5384693101056831,0.9061798459386640};

}

#endif // _DG_GAUSS_LEGENDRE_
