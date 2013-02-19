#ifndef _DG_DLT_
#define _DG_DLT_

namespace dg
{

template<size_t n>
struct DLT
{
    static double const forward[n][n];
    static double const backward[n][n];
    const static double weight[n];
    const static double abscissa[n];
};

//values taken from http://processingjs.nihongoresources.com/bezierinfo/legendre-gauss-values.php
template<>
const double DLT<1>::weight[1] = 
{2.};
template <>
const double DLT<1>::abscissa[1] = 
{0.}; 
template<>
const double DLT<2>::weight[2] = 
{ 1., 1.};
template<>
const double DLT<2>::abscissa[2] = 
{ -0.5773502691896257, 0.5773502691896257 };
template<>
const double DLT<3>::weight[3] = 
{ 0.5555555555555556, 0.8888888888888888, 0.5555555555555556};
template<>
const double DLT<3>::abscissa[3] = 
{-0.7745966692414834, 0.0, 0.7745966692414834 };
template<>
const double DLT<4>::weight[4] = 
{ 0.3478548451374538, 0.6521451548625461, 0.6521451548625461,0.3478548451374538} ;
template<>
const double DLT<4>::abscissa[4] = 
{ -0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
template<>
const double DLT<5>::weight[5] = 
{0.2369268850561891,    0.4786286704993665,0.5688888888888889,0.4786286704993665,0.2369268850561891};
template<>
const double DLT<5>::abscissa[5] = 
{ -0.9061798459386640,-0.5384693101056831,0.0,0.5384693101056831,0.9061798459386640};

template<>
const double DLT<1>::forward[1][1] = {
    {1.}
};
template<>
const double DLT<1>::backward[1][1] = {
    {1.}
};
template<>
const double DLT<2>::forward[2][2] = {
    { 0.5                , 0.5}, 
    {-0.86602540378443865, 0.86602540378443865} 
}; 

template<>
const double DLT<2>::backward[2][2] = {
    {1., -0.57735026918962576},
    {1.,  0.57735026918962576}
}; 

template<>
const double DLT<3>::forward[3][3] = {
    { 0.27777777777777778, 0.44444444444444444, 0.27777777777777778},
    {-0.64549722436790283, 0.0                , 0.64549722436790283},
    { 0.55555555555555556,-1.1111111111111111 , 0.55555555555555556}
}; 

template<>
const double DLT<3>::backward[3][3] = {
    {1., -0.77459666924148338,  0.4},
    {1.,  0.0                , -0.5},
    {1.,  0.77459666924148338,  0.4}
}; 
template<>
const double DLT<4>::forward[4][4] = {
    {0.17392742256872693,  0.32607257743127307,  0.32607257743127307, 0.17392742256872693},
    {-0.44932565746768106, -0.33257548547846420,  0.33257548547846420, 0.44932565746768106},
    {0.53250804201891148, -0.53250804201891152, -0.53250804201891152, 0.53250804201891148},
    {-0.37102700340194724,  0.93977247037775300, -0.93977247037775300, 0.37102700340194724}
}; 

template<>
const double DLT<4>::backward[4][4] = {
    {1., -.86113631159405256,  .61233362071871377, -.30474698495520613},
    {1., -.33998104358485625, -.32661933500442811, 0.41172799967289956},
    {1.,  .33998104358485625, -.32661933500442811, -.41172799967289956},
    {1.,  .86113631159405256,  .61233362071871377, .30474698495520613}
}; 
}
#endif //_DG_DLT_
