#pragma once

#include "mpi_matrix.h"
#include "projection.cuh"
#include "typedefs.cuh"

/*! @file
  @brief Useful MPI typedefs and overloads of interpolation and projection
  */

namespace dg
{
///@addtogroup typedefs
///@{
//interpolation matrices
typedef MPIDistMat< dg::IHMatrix, GeneralComm< dg::iHVec, dg::HVec > > MIHMatrix; //!< MPI distributed CSR host Matrix
typedef MPIDistMat< dg::IDMatrix, GeneralComm< dg::iDVec, dg::DVec > > MIDMatrix; //!< MPI distributed CSR device Matrix
///@}

namespace create
{

///@addtogroup interpolation
///@{

///@copydoc dg::create::interpolation(const Grid1d&,const Grid1d&)
dg::MIHMatrix interpolation( const aMPITopology2d& g_new, const aMPITopology2d& g_old)
{
    return MIHMatrix( interpolation( g_new.local(), g_old.local()), GeneralComm<iHVec, HVec>());
}
///@copydoc interpolation(const Grid1d&,const Grid1d&)
dg::MIHMatrix interpolation( const aMPITopology3d& g_new, const aMPITopology3d& g_old)
{
    return MIHMatrix( interpolation( g_new.local(), g_old.local()), GeneralComm<iHVec, HVec>());
}

///@copydoc interpolationT(const Grid1d&,const Grid1d&)
dg::MIHMatrix interpolationT( const aMPITopology2d& g_new, const aMPITopology2d& g_old)
{
    return MIHMatrix( interpolationT( g_new.local(), g_old.local()), GeneralComm<iHVec, HVec>());
}
///@copydoc interpolationT(const Grid1d&,const Grid1d&)
dg::MIHMatrix interpolationT( const aMPITopology3d& g_new, const aMPITopology3d& g_old)
{
    return MIHMatrix( interpolationT( g_new.local(), g_old.local()), GeneralComm<iHVec, HVec>());
}

///@copydoc projection(const Grid1d&,const Grid1d&)
dg::MIHMatrix projection( const aMPITopology2d& g_new, const aMPITopology2d& g_old)
{
    return MIHMatrix( projection( g_new.local(), g_old.local()), GeneralComm<iHVec, HVec>());
}
///@copydoc projection(const Grid1d&,const Grid1d&)
dg::MIHMatrix projection( const aMPITopology3d& g_new, const aMPITopology3d& g_old)
{
    return MIHMatrix( projection( g_new.local(), g_old.local()), GeneralComm<iHVec, HVec>());
}
///@}

}//namespace create
}//namespace dg
