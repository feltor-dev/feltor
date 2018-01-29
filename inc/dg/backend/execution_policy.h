#pragma once

namespace dg
{

///@addtogroup dispatch
///@{
struct AnyPolicyTag{}; //!< Execution Policy base class
///@}
struct SerialTag    : public AnyPolicyTag{};//!< sequential execution
struct CudaTag      : public AnyPolicyTag{};//!< CUDA implementation
struct OmpTag       : public AnyPolicyTag{};//!< OpenMP parallel execution


}//namespace dg
