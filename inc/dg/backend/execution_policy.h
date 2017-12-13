#pragma once

namespace dg
{

struct AnyPolicyTag{};
struct SerialTag    : public AnyPolicyTag{};//sequential
struct CudaTag      : public AnyPolicyTag{};//CUDA
struct OmpTag       : public AnyPolicyTag{};//OpenMP


}//namespace dg
