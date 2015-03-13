#include <iostream>
#include <vector>

#include "blas1.h"


//test program that calls every blas1 function for every specialization

int main()
{
    thrust::device_vector<double> v1( 5, 2), v2( 5, 3);
    thrust::device_vector<double> v3(5);
    std::cout << "5*(2*3) = "<<dg::blas1::dot( v1, v2) << std::endl;
    dg::blas1::axpby( 2., v1, 3., v2, v3);
    std::cout << "2*2+ 3*3 = " << v3[0] << std::endl;
    dg::blas1::axpby( 0., v1, 3., v2, v3);
    std::cout << "0*2+ 3*3 = " << v3[0] << std::endl;
    dg::blas1::axpby( 2., v1, 0., v2, v3);
    std::cout << "2*2+ 0*3 = " << v3[0] << std::endl;
    dg::blas1::pointwiseDot( v1, v2, v3);
    std::cout << "2*3 = "<<v3[0]<<std::endl;
    dg::blas1::axpby( 2., v1, 3., v2);
    std::cout << "2*2+ 3*3 = " << v2[0] << std::endl;
    dg::blas1::axpby( 2., v1, 0., v2);
    std::cout << "2*2+ 0 = " << v2[0] << std::endl;
    dg::blas1::axpby( 0., v1, 1., v2);
    std::cout << "4 = " << v2[0] << std::endl;


    std::cout << "Test std::vector \n";
    std::vector<thrust::device_vector<double> > w1( 2, v1), w2(2, v2), w3( w2);
    dg::blas1::axpby( 2., w1, 3., w2, w3);
    std::cout << " 2. * 2 + 3.*4 = " <<w3[0][0] <<std::endl;
    dg::blas1::pointwiseDot( w1, w2, w3);
    std::cout << " 2 * 4 = " <<w3[0][0] <<std::endl;
    dg::blas1::axpby( 2., w1, 3., w2);
    std::cout << " 2. * 2 + 3.*4 = " <<w2[0][0] <<std::endl;


    return 0;

}
