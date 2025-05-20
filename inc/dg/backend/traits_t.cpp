#include <iostream>

#include "predicate.h"
#include "tensor_traits.h"
#include "tensor_traits_scalar.h"
#include "tensor_traits_thrust.h"
#include "tensor_traits_std.h"

#include "catch2/catch_all.hpp"

struct MyOwnScalar;
namespace dg
{
// TensorTraits are necessary only for std::vector
// thrust::device_vector assumes a scalar by default
template<>
struct TensorTraits<MyOwnScalar>
{
    using value_type = MyOwnScalar;
    using tensor_category = dg::AnyScalarTag;
    using execution_policy = dg::AnyPolicyTag;
};
}
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    using execution_policy  = dg::CudaTag ;
#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
    using execution_policy  = dg::OmpTag ;
#else
    using execution_policy  = dg::SerialTag ;
#endif

// This programs PASSES if it compiles!!

TEST_CASE( "Tensor traits std::vector")
{
    // Test for equality of Tags:
    STATIC_REQUIRE( std::is_same_v< dg::get_tensor_category<
            std::vector<std::complex<double>> >, dg::ThrustVectorTag
            >);
    STATIC_REQUIRE( std::is_same_v< dg::get_tensor_category<
            std::vector<double> >, dg::ThrustVectorTag
            >);
    STATIC_REQUIRE( std::is_same_v< dg::get_tensor_category<
            std::vector<std::vector<double>> >, dg::RecursiveVectorTag
            >);
    STATIC_REQUIRE( std::is_same_v< dg::get_tensor_category<
            std::vector<MyOwnScalar> >, dg::ThrustVectorTag
            >);

    STATIC_REQUIRE( dg::has_policy_v< std::vector<std::complex<double>>,
        dg::SerialTag >);
    STATIC_REQUIRE( dg::has_policy_v< std::vector<double>, dg::SerialTag >);
    STATIC_REQUIRE( dg::has_policy_v< std::vector<std::vector<double>>,
        dg::SerialTag >);
    STATIC_REQUIRE( dg::has_policy_v< std::vector<MyOwnScalar>, dg::SerialTag >);

    STATIC_REQUIRE( std::is_same_v< dg::get_value_type<
            std::vector<std::complex<double>> >, std::complex<double>
            >);
    STATIC_REQUIRE( std::is_same_v< dg::get_value_type<
            std::vector<double> >, double
            >);
    STATIC_REQUIRE( std::is_same_v< dg::get_value_type<
            std::vector<std::vector<double>> >, double
            >);
    STATIC_REQUIRE( std::is_same_v< dg::get_value_type<
            std::vector<MyOwnScalar> >, MyOwnScalar
            >);
}
TEST_CASE( "thrust::device_vector")
{
    STATIC_REQUIRE( std::is_same_v< dg::get_tensor_category<
            thrust::device_vector<std::complex<double>> >,
            dg::ThrustVectorTag >);
    STATIC_REQUIRE( std::is_same_v< dg::get_tensor_category<
            thrust::device_vector<double> >,
            dg::ThrustVectorTag >);
    STATIC_REQUIRE( std::is_same_v< dg::get_tensor_category<
            thrust::device_vector<MyOwnScalar> >,
            dg::ThrustVectorTag >);

    STATIC_REQUIRE( dg::has_policy_v<
            thrust::device_vector<std::complex<double>>,
            execution_policy >);
    STATIC_REQUIRE( dg::has_policy_v<
            thrust::device_vector<double>,
            execution_policy >);
    STATIC_REQUIRE( dg::has_policy_v<
            thrust::device_vector<MyOwnScalar>,
            execution_policy >);

    STATIC_REQUIRE( std::is_same_v< dg::get_value_type<
            thrust::device_vector<thrust::complex<double>> >,
            thrust::complex<double> >);
    STATIC_REQUIRE( std::is_same_v< dg::get_value_type<
            thrust::device_vector<double> >,
            double >);
    STATIC_REQUIRE( std::is_same_v< dg::get_value_type<
            thrust::device_vector<MyOwnScalar> >,
            MyOwnScalar >);
}

TEST_CASE( "is_scalar_v, is_vector_v, is_matrix_v, has_policy_v")
{
    STATIC_REQUIRE( dg::is_scalar_v< MyOwnScalar>);
    STATIC_REQUIRE( dg::is_vector_v< thrust::device_vector<double>,
        dg::SharedVectorTag>);
    STATIC_REQUIRE( !dg::is_scalar_v< thrust::host_vector<int>>);
    STATIC_REQUIRE( dg::has_policy_v< thrust::device_vector<double>,
        execution_policy>);
    STATIC_REQUIRE( !dg::has_policy_v< std::vector<std::complex<double>>,
        dg::CudaTag >);
}
// ... continue

