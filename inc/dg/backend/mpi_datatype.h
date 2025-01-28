#pragma once

#include <mpi.h>
#include <complex.h>
#include <thrust/complex.h>

namespace dg
{
///@cond
template<class value_type>
inline MPI_Datatype getMPIDataType(){ assert( false && "Type not supported!\n" ); return MPI_DATATYPE_NULL; }
template<> inline MPI_Datatype getMPIDataType<char>(){ return MPI_CHAR;}
template<> inline MPI_Datatype getMPIDataType<signed char>(){ return MPI_SIGNED_CHAR;}
template<> inline MPI_Datatype getMPIDataType<unsigned char>(){ return MPI_UNSIGNED_CHAR;}
template<> inline MPI_Datatype getMPIDataType<wchar_t>(){ return MPI_WCHAR;}
template<> inline MPI_Datatype getMPIDataType<signed short>(){ return MPI_SHORT;}
template<> inline MPI_Datatype getMPIDataType<unsigned short>(){ return MPI_UNSIGNED_SHORT;}
template<> inline MPI_Datatype getMPIDataType<int>(){ return MPI_INT;}
template<> inline MPI_Datatype getMPIDataType<unsigned int>(){ return MPI_UNSIGNED;}
template<> inline MPI_Datatype getMPIDataType<signed long int>(){ return MPI_LONG;}
template<> inline MPI_Datatype getMPIDataType<unsigned long int>(){ return MPI_UNSIGNED_LONG;}
template<> inline MPI_Datatype getMPIDataType<signed long long int>(){ return MPI_LONG_LONG;}
template<> inline MPI_Datatype getMPIDataType<unsigned long long int>(){ return MPI_UNSIGNED_LONG_LONG;}
template<> inline MPI_Datatype getMPIDataType<float>(){ return MPI_FLOAT;}
template<> inline MPI_Datatype getMPIDataType<double>(){ return MPI_DOUBLE;}
template<> inline MPI_Datatype getMPIDataType<long double>(){ return MPI_LONG_DOUBLE;}
//template<> inline MPI_Datatype getMPIDataType<std::int8_t>(){ return MPI_INT8_T;}
//template<> inline MPI_Datatype getMPIDataType<std::int16_t>(){ return MPI_INT16_T;}
//template<> inline MPI_Datatype getMPIDataType<std::int32_t>(){ return MPI_INT32_T;}
//template<> inline MPI_Datatype getMPIDataType<std::int64_t>(){ return MPI_INT64_T;}
//template<> inline MPI_Datatype getMPIDataType<std::uint8_t>(){ return MPI_UINT8_T;}
//template<> inline MPI_Datatype getMPIDataType<std::uint16_t>(){ return MPI_UINT16_T;}
//template<> inline MPI_Datatype getMPIDataType<std::uint32_t>(){ return MPI_UINT32_T;}
//template<> inline MPI_Datatype getMPIDataType<std::uint64_t>(){ return MPI_UINT64_T;}
template<> inline MPI_Datatype getMPIDataType<bool>(){ return MPI_C_BOOL;}
template<> inline MPI_Datatype getMPIDataType<std::complex<float>>(){ return MPI_C_COMPLEX;}
template<> inline MPI_Datatype getMPIDataType<std::complex<double>>(){ return MPI_C_DOUBLE_COMPLEX;}
template<> inline MPI_Datatype getMPIDataType<std::complex<long double>>(){ return MPI_C_LONG_DOUBLE_COMPLEX;}
template<> inline MPI_Datatype getMPIDataType<thrust::complex<float>>(){ return MPI_C_COMPLEX;}
template<> inline MPI_Datatype getMPIDataType<thrust::complex<double>>(){ return MPI_C_DOUBLE_COMPLEX;}
template<> inline MPI_Datatype getMPIDataType<thrust::complex<long double>>(){ return MPI_C_LONG_DOUBLE_COMPLEX;}
///@endcond
}//namespace dg
