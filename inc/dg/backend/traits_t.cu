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
    // This is necessary only for std::vector
    // thrust::device_vector assumes a scalar by default
template<>
struct TensorTraits<MyOwnScalar>
{
    using value_type = MyOwnScalar;
    using tensor_category = dg::ScalarTag;
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

int main()
{
    // std::vector
    static_assert( std::is_same< dg::get_tensor_category<
            std::vector<std::complex<double>> >, dg::ThrustVectorTag
            >::value, " std ");
    static_assert( std::is_same< dg::get_tensor_category<
            std::vector<double> >, dg::ThrustVectorTag
            >::value, " std ");
    static_assert( std::is_same< dg::get_tensor_category<
            std::vector<std::vector<double>> >, dg::RecursiveVectorTag
            >::value, " std ");
    static_assert( std::is_same< dg::get_tensor_category<
            std::vector<MyOwnScalar> >, dg::ThrustVectorTag
            >::value, " std ");

    static_assert( std::is_same< dg::get_execution_policy<
            std::vector<std::complex<double>> >, dg::SerialTag
            >::value, " std ");
    static_assert( std::is_same< dg::get_execution_policy<
            std::vector<double>>, dg::SerialTag
            >::value, " std ");
    static_assert( std::is_same< dg::get_execution_policy<
            std::vector<std::vector<double>>>, dg::SerialTag
            >::value, " std ");
    static_assert( std::is_same< dg::get_execution_policy<
            std::vector<MyOwnScalar> >, dg::SerialTag
            >::value, " std ");

    static_assert( std::is_same< dg::get_value_type<
            std::vector<std::complex<double>> >, std::complex<double>
            >::value, " std ");
    static_assert( std::is_same< dg::get_value_type<
            std::vector<double> >, double
            >::value, " std ");
    static_assert( std::is_same< dg::get_value_type<
            std::vector<std::vector<double>> >, double
            >::value, " std ");
    static_assert( std::is_same< dg::get_value_type<
            std::vector<MyOwnScalar> >, MyOwnScalar
            >::value, " std ");

    // thrust::device_vector
    static_assert( std::is_same< dg::get_tensor_category<
            thrust::device_vector<std::complex<double>> >, dg::ThrustVectorTag
            >::value, " thrust ");
    static_assert( std::is_same< dg::get_tensor_category<
            thrust::device_vector<double> >, dg::ThrustVectorTag
            >::value, " thrust ");
    static_assert( std::is_same< dg::get_tensor_category<
            thrust::device_vector<MyOwnScalar> >, dg::ThrustVectorTag
            >::value, " thrust ");

    static_assert( std::is_same< dg::get_execution_policy<
            thrust::device_vector<std::complex<double>> >, execution_policy
            >::value, " thrust ");
    static_assert( std::is_same< dg::get_execution_policy<
            thrust::device_vector<double>>, execution_policy
            >::value, " thrust ");
    static_assert( std::is_same< dg::get_execution_policy<
            thrust::device_vector<MyOwnScalar>>, execution_policy
            >::value, " thrust ");

    static_assert( std::is_same< dg::get_value_type<
            thrust::device_vector<thrust::complex<double>> >, thrust::complex<double>
            >::value, " thrust ");
    static_assert( std::is_same< dg::get_value_type<
            thrust::device_vector<double> >, double
            >::value, " thrust ");
    static_assert( std::is_same< dg::get_value_type<
            thrust::device_vector<MyOwnScalar> >, MyOwnScalar
            >::value, " thrust ");

    // ... continue
    return 0;
}

