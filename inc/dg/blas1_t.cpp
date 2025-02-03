//#define CUSP_DEVICE_BLAS_SYSTEM CUSP_DEVICE_BLAS_CUBLAS
#include <iostream>
#include <vector>
#include <array>

#ifdef WITH_MPI
#include <mpi.h>
#include "backend/mpi_init.h"
#endif

#include "backend/typedefs.h"
#include "blas1.h"
#include "functors.h"


#include "catch2/catch.hpp"
//test program that (should ) call every blas1 function for every specialization

//using Vector = std::array<double,2>;
//using Vector = thrust::host_vector<double>;
using Vector = dg::x::DVec;
using value_type = double;
//using Vector = cusp::array1d<double, cusp::device_memory>;

template<class Vector, class T>
bool equal( const Vector& vec, T result)
{
    for( unsigned i=0; i<vec.size(); i++)
    {
        if( fabs( vec[i] - result) > 1.2e-16)
        {
            UNSCOPED_INFO( "Element "<<i<<" "<< vec[i] << " "<<result);
            UNSCOPED_INFO( "Difference "<<vec[i]-result);
            return false;
        }
    }
    return true;
}
template<class Vector>
bool equal64( const Vector& vec, int64_t result)
{
    dg::exblas::udouble ud;
    for( unsigned i=0; i<vec.size(); i++)
    {
        ud.d = vec[i];

        if( abs(ud.i - result) >2)
        {
            UNSCOPED_INFO( "Element "<<i<<" "<< ud.i << " "<<result);
            return false;
        }
    }
    return true;
}
template<class Vector>
auto result( const Vector& vec) { return vec[0];}
#ifdef WITH_MPI
template<class Vector, class T>
bool equal( const dg::MPI_Vector<Vector>& vec, T result)
{
    return equal( vec.data(), result);
}
template<class Vector>
bool equal64( const dg::MPI_Vector<Vector>& vec, int64_t result)
{
    return equal64( vec.data(), result);
}
template<class Vector>
auto result( const dg::MPI_Vector<Vector>& vec) { return vec.data()[0];}
#endif

template<class Vector, size_t Nd>
auto result( const std::array<Vector,Nd>& vec) { return result( vec[0]);}

template<class Vector, class T>
bool equal_rec( const Vector& vec, T result)
{
    for( const auto& v : vec)
        if( not equal ( v, result))
            return false;
    return true;
}


TEST_CASE("blas1")
{
#ifdef WITH_MPI
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm comm, commX, commY;
    comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {0,0}, false);
#endif
    // "This program tests the blas1 functions up to binary reproducibility
    // with the exception of the dot function, which is tested in the
    // dg/topology/evaluation_t program\n";
    //Vector v1( 5, 2.0002), v2( 5, 3.00003), v3(5,5.0005), v4(5,4.00004), v5(v4);
    //Vector v1( {2,2.0002}), v2({3,3.00003}), v3({5,5.0005}), v4({4,4.00004}), v5(v4); //std::array
    SECTION( "Shared vector")
    {
        dg::DVec v1p( 500, 2.0002), v2p( 500, 3.00003), v3p(500,5.0005), v4p(500,4.00004);
#ifdef WITH_MPI
        Vector v1(v1p, comm), v2(v2p, comm), v3(v3p, comm), v4(v4p, comm);
        Vector nanvec( v3p, comm);
        nanvec.data()[0] = 1./0.;
#else
        Vector v1(v1p), v2(v2p), v3(v3p), v4(v4p);
        Vector nanvec( v3p);
        nanvec[0] = 1./0.;
#endif

        Vector res = nanvec;
        REQUIRE( not equal64( res, 4617316080911554445));
        dg::blas1::copy( v3, res);
        INFO("copy (x=x)");
        CHECK( equal64( res, 4617316080911554445));

        dg::blas1::scal( v3, 3e-10);
        INFO("scal (x=ax)");
        CHECK( equal64( v3, 4474825110624711575));

        dg::blas1::plus( v3, 3e-10);
        INFO( "plus (x=x+a)");
        CHECK( equal64( v3, 4476275821608249130));

        dg::blas1::axpby( 3e+10, v3, 1 , v4);
        INFO( "fma (y=ax+y)");
        CHECK( equal64( v4, 4633360230582305548));

        dg::blas1::axpby( 3e-10, v1, -2e-10 , v2);
        INFO( "axpby (y=ax+by)");
        CHECK(equal64( v2, 4408573477492505937));

        auto v5 = nanvec; //we test here if nan breaks code
        dg::blas1::axpby( 3e-10, v1, -2. , v2, v5);
        INFO( "axpby (z=ax+by)");
        CHECK( equal64( v5, 4468869610430797025));

        dg::blas1::axpbypgz( 2.5, v1, 7.e+10, v2, -0.125, v3);
        INFO( "axpbypgz (y=ax+by+gz)");
        CHECK( equal64( v3, 4617320336812948958));

        v3 = nanvec; //we test here if nan breaks code
        dg::blas1::pointwiseDot( v1, v2, v3);
        INFO( "pDot (z=xy)");
        CHECK( equal64( v3, 4413077932784031586));

        dg::blas1::pointwiseDot( 0.2, v1, v2, +0.4e10, v3);
        INFO( "pDot ( z=axy+bz)");
        CHECK( equal64 ( v3, 4556605413983777388));

#ifdef WITH_MPI
        v5 = {v4p, comm};
#else
        v5 = v4p;
#endif
        dg::blas1::pointwiseDot( -0.2, v1, v2, 0.4, v3, v4, 0.1, v5);
        INFO( "pDot (z=axy+bfh+gz)");
        CHECK( equal64( v5,4601058031075598447));

        dg::blas1::pointwiseDot( 0.2, v1, v2,v4, 0.4, v3);
        INFO( "pDot (z=awxy + bz)");
        CHECK( equal64( v3,4550507856334720009));

        v5 = nanvec; //we test here if nan breaks code
        dg::blas1::pointwiseDivide( v1,v2,v5);
        INFO( "pDivide (z=x/y)");
        CHECK(equal64 ( v5, 4810082017219139146));

        dg::blas1::pointwiseDivide( 5.,v1,v2,-1,v3);
        INFO( "pDivide (z=ax/y+bz)");
        CHECK( equal64(v3,  4820274520177585116));

        v3 = nanvec; //we test here if nan breaks code
        dg::blas1::transform( v1, v3, dg::EXP<>());
        INFO( "transform y=exp(x)");
        CHECK( equal64( v3, 4620007020034741378));

        // host transform (checks if nvcc generates warnings, it should be suppressed)
        dg::x::HVec v1h ( v1), v3h(v3);
        dg::blas1::transform( v1h, v3h, [](double x){return exp(x);});
        INFO( "h_transform y=exp(x)");
        CHECK( equal64( v3h, 4620007020034741378));

#ifdef WITH_MPI
        std::vector<double> xs{1,2,3};
        std::vector<double> ys{10*(double)rank + 10};
        std::vector<thrust::complex<double>> sol =
            {{1111+10.*rank,1}, {1112+10.*rank,1}, {1113+10.*rank,1}};
        dg::x::cDVec sold( sol, comm);
#else
        std::vector<double> xs{1,2,3};
        std::vector<double> ys{10,20,30,40};
        std::vector<thrust::complex<double>> sol = {{1111,1}, {1112,1},
        {1113,1}, {1121,1}, {1122,1}, {1123,1}, {1131,1}, {1132,1}, {1133,1},
        {1141,1}, {1142,1}, {1143,1}};
        dg::cDVec sold( sol);
#endif
        double zs{100};
        double ws{1000};
        std::vector<thrust::complex<double>> y(xs.size()*ys.size());
#ifdef WITH_MPI
        commX = dg::mpi_cart_sub( comm, {1,0});
        commY = dg::mpi_cart_sub( comm, {0,1});
        dg::x::DVec xsd(xs, commX), ysd(ys, commY);
        dg::x::cDVec yd( y, comm);
#else
        dg::DVec xsd(xs), ysd(ys);
        dg::cDVec yd(y.size());
#endif
        dg::blas1::kronecker( yd, dg::equals(), []DG_DEVICE(double x, double y,
                    double z, double u){ return thrust::complex<double>{x+y+z+u,1};}, xsd, ysd, zs, ws);
        INFO( "Kronecker test (X ox Y)");
#ifdef WITH_MPI
        CHECK( yd.data() == sold.data());
        CHECK( yd.communicator() == sold.communicator());
#else
        CHECK( yd == sold);
#endif

        auto ydd = dg::kronecker( []DG_DEVICE( double x, double y, double z, double
            u){ return thrust::complex<double>{x+y+z+u,1};}, xsd, ysd, zs, ws);
        INFO( "Kronecker test (X ox Y)");
#ifdef WITH_MPI
        CHECK( ydd.data() == sold.data());
        CHECK( ydd.communicator() == sold.communicator());
#else
        CHECK( ydd == sold);
#endif
    }

    //v1 = 2, v2 = 3
    SECTION( "Recursive vector")
    {
        using ArrVec = std::array<Vector,2>;
        dg::DVec v1p( 500, 2.), v2p( 500, 3.), v3p(500,5.), v4p(500,4.);
#ifdef WITH_MPI
        Vector v1(v1p, comm), v2(v2p, comm), v3(v3p, comm), v4(v4p, comm);
#else
        Vector v1(v1p), v2(v2p), v3(v3p), v4(v4p);
#endif
        ArrVec w1( dg::construct<ArrVec>(v1)),
            w2({v2,v2}), w3({v3,v3}), w4({v4,v4});

        dg::blas1::axpby( 2., w1, 3., w2, w3);
        INFO( "2*2+ 3*3 = " << result(w3) <<" (13)");
        CHECK( equal_rec<ArrVec,value_type>( w3, 13));
        dg::blas1::axpby( 0., w1, 3., w2, w3);
        INFO( "0*2+ 3*3 = " << result(w3) <<" (9)");
        CHECK( equal_rec<ArrVec,value_type>( w3, 9));
        dg::blas1::axpby( 2., w1, 0., w2, w3);
        INFO( "2*2+ 0*3 = " << result(w3) <<" (4)");
        CHECK( equal_rec<ArrVec,value_type>( w3, 4));
        dg::blas1::pointwiseDot( w1, w2, w3);
        INFO( "2*3 = "<<result(w3)<<" (6)");
        CHECK( equal_rec<ArrVec,value_type>( w3, 6));
        dg::blas1::pointwiseDot( 2., w1, w2, -4., w3);
        INFO( "2*2*3 -4*6 = "<<result(w3)<<" (-12)");
        CHECK( equal_rec<ArrVec,value_type>( w3, -12));
        dg::blas1::pointwiseDot( 2., w1, w2,w4, -4., w3);
        INFO( "2*2*3*4 -4*(-12) = "<<result(w3)<<" (96)");
        CHECK( equal_rec<ArrVec,value_type>( w3, 96));
        dg::blas1::pointwiseDot( 2., w1, w2, -4., w1, w2, 0., w2);
        INFO( "2*2*3 -4*2*3 = "<<result(w2)<<" (-12)");
        CHECK( equal_rec<ArrVec,value_type>( w2, -12));
        dg::blas1::axpby( 2., w1, 3., w2);
        INFO( "2*2+ 3*(-12) = " << result(w2) <<" (-32)");
        CHECK( equal_rec<ArrVec,value_type>( w2, -32));
        dg::blas1::axpby( 2.5, w1, 0., w2);
        INFO( "2.5*2+ 0 = " << result(w2) <<" (5)");
        CHECK( equal_rec<ArrVec,value_type>( w2, 5));
        dg::blas1::axpbypgz( 2.5, w1, 2., w2, -0.125, w3);
        INFO( "2.5*2+ 2.*5-0.125*96 = " << result(w3) <<" (3)");
        CHECK( equal_rec<ArrVec,value_type>( w3, 3));
        dg::blas1::pointwiseDivide( 5.,w1,5.,-1,w3);
        INFO( "5*2/5-1*3 = " << result(w3) <<" (-1)");
        CHECK( equal_rec<ArrVec,value_type>( w3, -1));
        dg::blas1::pointwiseDivide( w1,5.,w3);
        INFO( "2/5 = " << result(w3) <<" (0.4)");
        CHECK( equal_rec<ArrVec,value_type>( w3, 0.4));
        dg::blas1::copy( w2, w1);
        INFO( "5 = " << result(w1) <<" (5)");
        CHECK( equal_rec<ArrVec,value_type>( w1, 5));
        dg::blas1::scal( w1, 0.4);
        INFO( "5*0.5 = " << result(w1) <<" (2)");
        CHECK( equal_rec<ArrVec,value_type>( w1, 2));
        dg::blas1::evaluate( w4, dg::equals(),dg::AbsMax<>(), w1, w2);
        INFO( "absMax( 2, 5) = " << result(w4) <<" (5)");
        CHECK( equal_rec<ArrVec,value_type>( w4, 5));
        dg::blas1::transform( w1, w3, dg::EXP<>());
        INFO( "e^2 = " << result(w3) <<" (7.389056...)");
        CHECK( equal_rec<ArrVec,value_type>( w3, exp(2)));
        dg::blas1::plus( w3, -7.0);
        INFO( "e^2-7 = " << result(w3) <<" (0.389056...)");
        CHECK( equal_rec<ArrVec,value_type>( w3, exp(2)-7.0));

        // Recursive Recursive
        w1 = dg::construct<ArrVec>(v1), w2 = {v2,v2}, w3 = {v3,v3}, w4 = {v4,v4};
        std::array<ArrVec,2> w11( dg::construct<std::array<ArrVec,2>>(v1)),
            w22({w2,w2}), w33({w3,w3}), w44({w4,w4});
        dg::blas1::axpby( 2., w11, 3., w22);
        INFO( "2*2+ 3*3 = " << result( w22) <<" (13)");
        CHECK( equal_rec( w22[0], value_type(13)));
        CHECK( equal_rec( w22[1], value_type(13)));
    }
}

