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


#include "catch2/catch_all.hpp"
//test program that (should ) call every blas1 function for every specialization

//using Vector = std::array<double,2>;
//using Vector = thrust::host_vector<double>;
using Vector = dg::x::DVec;
using value_type = double;

template<class Vector, class T>
bool equal( const Vector& vec, T result)
{
    for( unsigned i=0; i<vec.size(); i++)
    {
        // norm works for complex values
        using namespace std; // make ADL work
        if( sqrt(norm( T(vec[i]) - result)) > 1.2e-16)
        {
            UNSCOPED_INFO( "Element "<<i<<" "<< T(vec[i]) << " "<<result);
            UNSCOPED_INFO( "Difference "<<T(vec[i])-result);
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

TEST_CASE( "Complex algebra")
{
    dg::cDVec v1p( 500, {2.,2.}), v2p( 500, {3.,3});
    dg::DVec v3p(500, 3.), v4p(500, 4.);
#ifdef WITH_MPI
    dg::x::cDVec c1(v1p, MPI_COMM_WORLD), c2(v2p, MPI_COMM_WORLD);
    dg::x::DVec v3(v3p, MPI_COMM_WORLD), v4(v4p, MPI_COMM_WORLD);
#else
    dg::x::cDVec c1(v1p), c2(v2p);
    dg::x::DVec v3(v3p), v4(v4p);
#endif
    SECTION( "Axpby")
    {
        dg::blas1::axpby( 2, c1, 3., c2);
        INFO( "2*{2,2}+ 3*{3,3} = " << result(c2) <<" (13,13)");
        CHECK( equal<dg::cDVec,thrust::complex<double>>( c2, {13,13}));

        thrust::complex<double> a1 = {2,0}, a2 = {3.,0};
        dg::blas1::axpby( a1, c1, a2, c2);
        INFO( "{2,0}*{2,2}+ {3,0}*{13,13} = " << result(c2) <<" (43,43)");
        CHECK( equal<dg::cDVec,thrust::complex<double>>( c2, {43,43}));
    }
    SECTION( "pointwiseDot")
    {
        // Multiply double vector with complex vector
        dg::blas1::pointwiseDot( v3, c1, c2);
        INFO( "3*{2,2} = " << result(c2) <<" (6,6)");
        CHECK( equal<dg::cDVec,thrust::complex<double>>( c2, {6,6}));
    }
}

#ifndef WITH_MPI
TEST_CASE( "Blas1 documentation")
{
    SECTION( "vdot")
    {
        //! [vdot]
        // A fun way to compute the size of a vector
        dg::DVec v( 100,2);
        unsigned size = dg::blas1::vdot( []DG_DEVICE(double x){ return 1u;},
                v);
        CHECK( size == 100u); //100*1
        //! [vdot]

        //! [vcdot]
        // Compute the weighted norm of a complex vector
        std::vector<double> ww( 100, 42.);
        std::vector<std::complex<double>> cc( 100, {1,1});
        // Use auto to allow changing std::complex to thrust::complex
        // Use norm instead of std::norm and let ADL work
        double nrm = dg::blas1::vdot([](double w, auto z){ return w*norm(z);},
            ww, cc);
        CHECK( nrm == 100*42*2.0);
        //! [vcdot]
    }
    SECTION( "dot")
    {
        //! [dot]
        dg::DVec two( 100,2.0), three(100,3.0);
        double result = dg::blas1::dot(two, three);
        CHECK( result == 600.0); //100*(2*3)
        //! [dot]

        //! [cdot]
        std::vector<thrust::complex<double>> c0( 100, {1,1}), c1(100, {1,-1});
        auto cresult = dg::blas1::dot(c0, c1);
        // Result is of complex type
        CHECK( cresult == thrust::complex<double>{200,0});
        //! [cdot]


    }
    SECTION( "reduce")
    {
        //! [reduce nan]
        //Check if a vector contains Inf or NaN
        thrust::device_vector<double> x( 100, NAN );
        bool hasnan = false;
        hasnan = dg::blas1::reduce( x, false, thrust::logical_or<bool>(),
            dg::ISNFINITE<double>());
        CHECK( hasnan == true);
        //! [reduce nan]
        //! [reduce min]
        // Find minimum and maximum of a vector
        for( int u=0; u<100; u++)
            x[u] = (double)(u-10)*(u-10);
        // Notice the zero elements of the min and max functions
        double min = dg::blas1::reduce( x, +1e308, thrust::minimum<double>());
        double max = dg::blas1::reduce( x, -1e308, thrust::maximum<double>());
        CHECK( min == 0);
        CHECK( max == 89*89);
        //! [reduce min]
    }
    SECTION( "copy")
    {
        //! [copy]
        dg::DVec two( 100,2), two_copy(100);
        dg::blas1::copy( two, two_copy);
        CHECK( two_copy == two);
        //! [copy]
    }
    SECTION( "scal")
    {
        //! [scal]
        dg::DVec two( 100,2);
        dg::blas1::scal( two, 0.5);
        CHECK( two == dg::DVec( 100, 1));
        //! [scal]
    }
    SECTION( "plus")
    {
        //! [plus]
        dg::DVec two( 100,2);
        dg::blas1::plus( two, 3.);
        CHECK( two == dg::DVec( 100, 5.));
        //! [plus]
    }
    SECTION( "axpby")
    {
        //! [axpby]
        dg::DVec two( 100,2), three(100,3);
        dg::blas1::axpby( 2, two, 3., three);
        CHECK( three == dg::DVec( 100, 13.)); // 2*2+3*3
        //! [axpby]
    }
    SECTION( "axpbypgz")
    {
        //! [axpbypgz]
        dg::DVec two(100,2), five(100,5), result(100, 12);
        dg::blas1::axpbypgz( 2.5, two, 2., five, -3.,result);
        CHECK( result == dg::DVec( 100, -21.)); // 2.5*2+2*5-3*12
        //! [axpbypgz]
    }
    SECTION( "axpbyz")
    {
        //! [axpbyz]
        dg::DVec two( 100,2), three(100,3), result(100);
        dg::blas1::axpby( 2, two, 3., three, result);
        CHECK( result == dg::DVec( 100, 13.)); // 2*2+3*3
        //! [axpbyz]
    }
    SECTION( "pointwiseDot")
    {
        //! [pointwiseDot]
        dg::DVec two( 100,2), three( 100,3), result(100,6);
        dg::blas1::pointwiseDot(2., two,  three, -4., result );
        CHECK( result == dg::DVec( 100, -12.)); // 2*2*3-4*6
        //! [pointwiseDot]
    }
    SECTION( "pointwiseDot 2")
    {
        //! [pointwiseDot 2]
        dg::DVec two( 100,2), three( 100,3), result(100);
        dg::blas1::pointwiseDot( two,  three, result );
        CHECK( result == dg::DVec( 100, 6.)); // 2*3
        //! [pointwiseDot 2]
    }
    SECTION( "pointwiseDot 3")
    {
        //! [pointwiseDot 3]
        dg::DVec two( 100,2), three( 100,3), four(100,4), result(100,6);
        dg::blas1::pointwiseDot(2., two,  three, four, -4., result );
        CHECK( result == dg::DVec( 100, 24.)); // 2*2*3*4-4*6
        //! [pointwiseDot 3]
    }
    SECTION( "pointwiseDot 4")
    {
        //! [pointwiseDot 4]
        dg::DVec two(100,2), three(100,3), four(100,4), five(100,5), result(100,6);
        dg::blas1::pointwiseDot(2., two,  three, -4., four, five, 2., result );
        CHECK( result == dg::DVec( 100, -56.)); // 2*2*3-4*4*5+2*6
        //! [pointwiseDot 4]
    }
    SECTION( "pointwiseDivide")
    {
        //! [pointwiseDivide]
        dg::DVec two( 100,2), three( 100,3), result(100,1);
        dg::blas1::pointwiseDivide( 3, two,  three, 5, result );
        CHECK( result == dg::DVec( 100, 7.)); // 3*2/3+5*1
        //! [pointwiseDivide]
    }
    SECTION( "pointwiseDivide 2")
    {
        //! [pointwiseDivide 2]
        dg::DVec two( 100,2), three( 100,3), result(100);
        dg::blas1::pointwiseDivide( two,  three, result );
        CHECK( result == dg::DVec( 100, 2./3.));
        //! [pointwiseDivide 2]
    }
    SECTION( "transform")
    {
        //! [transform]
        dg::DVec two( 100,2), result(100);
        dg::blas1::transform( two, result, dg::EXP<double>());
        CHECK( result == dg::DVec( 100, exp(2.)));
        //! [transform]
    }
    SECTION( "evaluate")
    {
        //! [evaluate]
        auto function = [](double x, double y){
            return sin(x)*sin(y);
        };
        dg::HVec pi2(20, M_PI/2.), pi3( 20, 3*M_PI/2.), result(20, 0);
        dg::blas1::evaluate( result, dg::equals(), function, pi2, pi3);
        CHECK( result == dg::HVec( 20, -1.)); // sin(M_PI/2.)*sin(3*M_PI/2.) = -1
        //! [evaluate]
    }
    SECTION( "subroutine")
    {
        //! [subroutine]
        dg::DVec two( 100,2), four(100,4);
        dg::blas1::subroutine( []DG_DEVICE( double x, double y, double& z){
            z = 7*x + y +z;
        }, two, 3., four);
        CHECK( four == dg::HVec( 100, 21.)); // 7*2+3+4
        //! [subroutine]
    }
    SECTION( "kronecker")
    {
        //! [kronecker]
        auto function = []( double x, double y) {
            return x+y;
        };
        std::vector<double> xs{1,2,3,4}, ys{ 10,20,30,40}, y(16, 0);
        dg::blas1::kronecker( y, dg::equals(), function, xs, ys);
        // Note how xs varies fastest in result:
        std::vector<double> result = { 11,12,13,14,21,22,23,24,31,32,33,34,41,42,43,44};
        CHECK( y == result);

        // Note that the following code is equivalent
        // but that dg::blas1::kronecker has a performance advantage since
        // it never explicitly forms XS or YS
        std::vector<double> XS(16), YS(16);
        for( unsigned i=0; i<4; i++)
        for( unsigned k=0; k<4; k++)
        {
            XS[i*4+k] = xs[k];
            YS[i*4+k] = ys[i];
        }
        dg::blas1::evaluate( y, dg::equals(), function, XS, YS);
        CHECK( y == result);

        // Finally, we could also write
        dg::blas1::kronecker( XS, dg::equals(), []( double x, double y){ return x;}, xs, ys);
        dg::blas1::kronecker( YS, dg::equals(), []( double x, double y){ return y;}, xs, ys);
        dg::blas1::evaluate( y, dg::equals(), function, XS, YS);
        CHECK( y == result);
        //! [kronecker]
    }
    SECTION( "dg kronecker")
    {
        //! [dg kronecker]
        auto function = []( double x, double y) {
            return x+y;
        };
        std::vector<double> xs{1,2,3,4}, ys{ 10,20,30,40};
        auto y = dg::kronecker( function, xs, ys);
        // Note how xs varies fastest in result:
        std::vector<double> result = { 11,12,13,14,21,22,23,24,31,32,33,34,41,42,43,44};
        CHECK( y == result);
        //! [dg kronecker]
    }
    SECTION( "assign")
    {
        //! [assign]
        //Assign a host vector to a device vector
        dg::HVec host( 100, 1.);
        dg::DVec device(100);
        dg::assign( host, device );
        CHECK( device == dg::DVec( 100, 1.));

        //Assign a host vector to all elements of a std::vector of 3 dg::DVec
        std::vector<dg::DVec> device_vec;
        dg::assign( host, device_vec, 3);
        REQUIRE( device_vec.size() == 3);
        for( unsigned u=0; u<3; u ++)
            CHECK( device_vec[u] == device);
        //! [assign]
    }
    SECTION( "construct")
    {
        //! [construct]
        //Construct a device vector from host vector
        dg::HVec host( 100, 1.);
        auto device = dg::construct<dg::DVec>( host );
        CHECK( device == dg::DVec( 100, 1.));

        // Construct an array of vectors
        auto device_arr = dg::construct<std::array<dg::DVec, 3>>( host );
        for( unsigned u=0; u<3; u ++)
            CHECK( device_arr[u] == device);

        //Construct a std::vector of 3 dg::DVec from a host vector
        auto device_vec = dg::construct<std::vector<dg::DVec>>( host, 3);
        REQUIRE( device_vec.size() == 3);
        for( unsigned u=0; u<3; u++)
            CHECK( device_vec[u] == device);

        // Can also be used to change value type, e.g. complex:
        auto complex_dev = dg::construct<dg::cDVec>( host );
        CHECK( complex_dev == dg::cDVec( 100, thrust::complex<double>(1.)));
        //! [construct]
    }
}
TEST_CASE( "Aliases and special numbers")
{
    SECTION( "copy")
    {
        dg::DVec two( 100,2);
        dg::blas1::copy( two, two);
        CHECK( two == dg::DVec( 100, 2));
    }
    SECTION( "scal")
    {
        dg::DVec two( 100,2);
        dg::blas1::scal( two, 1.0);
        CHECK( two == dg::DVec( 100, 2));
    }
    SECTION( "plus")
    {
        dg::DVec two( 100,2);
        dg::blas1::plus( two, 0.);
        CHECK( two == dg::DVec( 100, 2.));
    }
    SECTION( "axpby")
    {
        dg::DVec two( 100,2), three(100,3);
        dg::blas1::axpby( 0, two, 3., three);
        CHECK( three == dg::DVec( 100, 9.));
        dg::blas1::axpby( 2, two, 0., three);
        CHECK( three == dg::DVec( 100, 4.));
        dg::blas1::axpby( 2, three, 0., three);
        CHECK( three == dg::DVec( 100, 8.));
        dg::blas1::axpby( 2, three, 1., three);
        CHECK( three == dg::DVec( 100, 24.));
    }
    SECTION( "axpbypgz")
    {
        dg::DVec two(100,2), five(100,5), result(100, 3);
        dg::blas1::axpbypgz( 0, two, 2., five, -3.,result);
        CHECK( result == dg::DVec( 100, 1.)); // 2*5-3*3
        dg::blas1::axpbypgz( 2, two, 0., five, 1.,result);
        CHECK( result == dg::DVec( 100, 5.)); // 2*2+1*1
        dg::blas1::axpbypgz( 2, two, 1., two, 0.,result);
        CHECK( result == dg::DVec( 100, 6.)); // 2*2+1*2
        dg::blas1::axpbypgz( 2, two, 1., result, 0.,result);
        CHECK( result == dg::DVec( 100, 10.)); // 2*2+1*6
        dg::blas1::axpbypgz( 2, result, 1., five, 0.,result);
        CHECK( result == dg::DVec( 100, 25.)); // 2*10+1*5
    }
    SECTION( "pointwiseDot")
    {
        dg::DVec two( 100,2), three( 100,3), result(100,6);
        dg::blas1::pointwiseDot(0., two,  three, -0.5, result );
        CHECK( result == dg::DVec( 100, -3.)); // -0.5*6

        dg::blas1::pointwiseDot(-1., result,  three, -0., result );
        CHECK( result == dg::DVec( 100, 9.)); // 3*3
        dg::blas1::pointwiseDot(0.25, two,  result, -0., result );
        CHECK( result == dg::DVec( 100, 4.5)); // 2*3
    }
    SECTION( "pointwiseDot 2")
    {
        dg::DVec two(100,2), three(100,3), four(100,4), five(100,5), result(100,6);
        dg::blas1::pointwiseDot(0., two,  three, -4., four, five, 2., result );
        CHECK( result == dg::DVec( 100, -68.)); // 0*2*3-4*4*5+2*6
        dg::blas1::pointwiseDot(2., two,  three, -0., four, five, 2., result );
        CHECK( result == dg::DVec( 100, -124.)); // 2*2*3-0*4*5+2*6
    }
    SECTION( "pointwiseDot 3")
    {
        dg::DVec two( 100,2), three( 100,3), four(100,4), result(100,6);
        dg::blas1::pointwiseDot(0., two,  three, four, -4., result );
        CHECK( result == dg::DVec( 100, -24.)); // -4*6
    }
    SECTION( "pointwiseDivide")
    {
        dg::DVec two( 100,2), three( 100,3), result(100,1);
        dg::blas1::pointwiseDivide( 0, two,  three, 6, result );
        CHECK( result == dg::DVec( 100, 6.)); // 6*1
        dg::blas1::pointwiseDivide( 1, result,  three, 0, result );
        CHECK( result == dg::DVec( 100, 2.)); // 6/3
    }
}
#endif

