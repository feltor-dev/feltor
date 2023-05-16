#pragma once

namespace dg
{

///@addtogroup dispatch
///@{

/**
 * @brief Execution Policy base class
 *
 * The Execution Policy Tag indicates the type of hardware memory is physically
allocated on in a vector class and therefore indicates the
possible parallelization and optimization strategies.
 * @note actually "policy" is a misleading name since we do not inject a policy into a type (in the sense Alexandrescu might use) but rather treat the execution as a trait of a type.
 * It is therefore unfortunately not possible to easily change the execution policy of a type in a program other than brute force MACROS.
 */
struct AnyPolicyTag{};
/// Indicate that a type does not have an execution policy
struct NoPolicyTag{};
///@}
/**
 * @brief Indicate sequential execution
 * @note the currently only classes with the SerialTag are thrust::host_vector<T> and std::array<T,N> with T an arithmetic type and N the array size.
 */
struct SerialTag    : public AnyPolicyTag{};
struct CudaTag      : public AnyPolicyTag{};//!< CUDA implementation
struct OmpTag       : public AnyPolicyTag{};//!< OpenMP parallel execution


}//namespace dg
