#pragma once
#include <stdint.h>
namespace exblas
{
namespace gpu
{

//first define device function equivalent to mylibm.hpp
//returns the original value at address
__device__ int64_t atomicAdd( int64_t* address, int64_t val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old; //*address_as_ull might change during the time the CAS is reached
        old = atomicCAS(address_as_ull, assumed,
                          (unsigned long long int)(val + (int64_t)old));
    } while( old != assumed);//try as often as necessary
    //assume that bit patterns don't change when casting
    //return the original value stored at address
    return (int64_t)(old);
}
// signedcarry in {-1, 0, 1}
__device__ int64_t xadd( int64_t &sa, int64_t x, unsigned char &of) {
    // OF and SF  -> carry=1
    // OF and !SF -> carry=-1
    // !OF        -> carry=0
    //int64_t y = atom_add(sa, x);
    int64_t y = atomicAdd(&sa, x); 
    int64_t z = y + x; // since the value sa->superacc[i] can be changed by another work item

    // TODO: cover also underflow
    of = 0;
    if(x > 0 && y > 0 && z < 0)
        of = 1;
    if(x < 0 && y < 0 && z > 0)
        of = 1;

    return y;
}
// Assumptions: th>tl>=0, no overlap between th and tl
__device__
inline double OddRoundSumNonnegative(double th, double tl) {
    // Adapted from:
    // Sylvie Boldo, and Guillaume Melquiond. "Emulation of a FMA and correctly rounded sums: proved algorithms using rounding to odd." IEEE Transactions on Computers, 57, no. 4 (2008): 462-471.
    union {
        double d;
        long l;
    } thdb;

    thdb.d = th + tl;
    // - if the mantissa of th is odd, there is nothing to do
    // - otherwise, round up as both tl and th are positive
    // in both cases, this means setting the msb to 1 when tl>0
    thdb.l |= (tl != 0.0);
    return thdb.d;
}
}//namespace gpu
}//namespace exblas
