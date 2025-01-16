#include <iostream>

#include "backend/predicate.h"
#include "backend/tensor_traits.h"
#include "backend/tensor_traits_scalar.h"
#include "backend/tensor_traits_thrust.h"
#include "backend/tensor_traits_cusp.h"
#include "backend/tensor_traits_std.h"

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

int main()
{
    // Test for equality of Tags:
    { // std::vector
    static_assert( std::is_same_v< dg::get_tensor_category<
            std::vector<std::complex<double>> >, dg::ThrustVectorTag
            >, " std ");
    static_assert( std::is_same_v< dg::get_tensor_category<
            std::vector<double> >, dg::ThrustVectorTag
            >, " std ");
    static_assert( std::is_same_v< dg::get_tensor_category<
            std::vector<std::vector<double>> >, dg::RecursiveVectorTag
            >, " std ");
    static_assert( std::is_same_v< dg::get_tensor_category<
            std::vector<MyOwnScalar> >, dg::ThrustVectorTag
            >, " std ");

    static_assert( dg::has_policy_v< std::vector<std::complex<double>>,
        dg::SerialTag >, " std ");
    static_assert( dg::has_policy_v< std::vector<double>, dg::SerialTag >,
        " std ");
    static_assert( dg::has_policy_v< std::vector<std::vector<double>>,
        dg::SerialTag >, " std ");
    static_assert( dg::has_policy_v< std::vector<MyOwnScalar>, dg::SerialTag >,
        " std ");

    static_assert( std::is_same_v< dg::get_value_type<
            std::vector<std::complex<double>> >, std::complex<double>
            >, " std ");
    static_assert( std::is_same_v< dg::get_value_type<
            std::vector<double> >, double
            >, " std ");
    static_assert( std::is_same_v< dg::get_value_type<
            std::vector<std::vector<double>> >, double
            >, " std ");
    static_assert( std::is_same_v< dg::get_value_type<
            std::vector<MyOwnScalar> >, MyOwnScalar
            >, " std ");
    }
    { // thrust::device_vector
    static_assert( std::is_same< dg::get_tensor_category<
            thrust::device_vector<std::complex<double>> >, dg::ThrustVectorTag
            >::value, " thrust ");
    static_assert( std::is_same< dg::get_tensor_category<
            thrust::device_vector<double> >, dg::ThrustVectorTag
            >::value, " thrust ");
    static_assert( std::is_same< dg::get_tensor_category<
            thrust::device_vector<MyOwnScalar> >, dg::ThrustVectorTag
            >::value, " thrust ");

    static_assert( dg::has_policy_v<
            thrust::device_vector<std::complex<double>>, execution_policy
            >, " thrust ");
    static_assert( dg::has_policy_v<
            thrust::device_vector<double>, execution_policy
            >, " thrust ");
    static_assert( dg::has_policy_v<
            thrust::device_vector<MyOwnScalar>, execution_policy
            >, " thrust ");

    static_assert( std::is_same_v< dg::get_value_type<
            thrust::device_vector<thrust::complex<double>> >, thrust::complex<double>
            >, " thrust ");
    static_assert( std::is_same_v< dg::get_value_type<
            thrust::device_vector<double> >, double
            >, " thrust ");
    static_assert( std::is_same_v< dg::get_value_type<
            thrust::device_vector<MyOwnScalar> >, MyOwnScalar
            >, " thrust ");
    }

    { // Test is_scalar_v, is_vector_v, is_matrix_v, has_policy_v
    static_assert( dg::is_scalar_v< MyOwnScalar>, "My own scalar");
    static_assert( dg::is_vector_v< thrust::device_vector<double>,
        dg::SharedVectorTag>, "Thrust is_vector_v");
    static_assert( !dg::is_scalar_v< thrust::host_vector<int>>, "Not scalar");
    static_assert( dg::has_policy_v< thrust::device_vector<double>,
        execution_policy>, "Thrust has_policy_v");
    static_assert( !dg::has_policy_v< std::vector<std::complex<double>>,
        dg::CudaTag >, " std has not cuda policy");
    }
    // ... continue
    return 0;
}

