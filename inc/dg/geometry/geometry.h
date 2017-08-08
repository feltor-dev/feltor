#pragma once

#include <cassert>
#include "thrust/host_vector.h"
#include "../backend/evaluation.cuh"
#include "../backend/weights.cuh"
#ifdef MPI_VERSION
#include "../backend/mpi_vector.h"
#include "../backend/mpi_evaluation.h"
#include "../backend/mpi_precon.h"
#endif//MPI_VERSION
#include "base.h"
#include "curvilinear.h"
#include "cartesianX.h"
#ifdef MPI_VERSION
#include "mpi_base.h"
#include "mpi_curvilinear.h"
#endif//MPI_VERSION
#include "tensor.h"
#include "transform.h"
#include "multiply.h"


/*!@file 
 *
 * geometry functions
 */

namespace dg{

namespace create{
///@addtogroup metric
///@{

/**
 * @brief Create the volume element on the grid (including weights!!)
 *
 * This is the same as the weights multiplied by the volume form \f$ \sqrt{g}\f$
 * @tparam Geometry any Geometry class
 * @param g Geometry object
 *
 * @return  The volume form
 */
template< class Geometry>
typename HostVec< typename TopologyTraits<Geometry>::memory_category>::host_vector volume( const Geometry& g)
{
    typedef typename HostVec< typename TopologyTraits<Geometry>::memory_category>::host_vector host_vector;
    SparseElement<host_vecotr> vol = dg::tensor::determinant(g.metric());
    host_vector temp = dg::create::weights( g);
    dg::tensor::pointwiseDot( vol, temp, temp);
    return temp;
}

/**
 * @brief Create the inverse volume element on the grid (including weights!!)
 *
 * This is the same as the inv_weights divided by the volume form \f$ \sqrt{g}\f$
 * @tparam Geometry any Geometry class
 * @param g Geometry object
 *
 * @return  The inverse volume form
 */
template< class Geometry>
typename HostVec< typename TopologyTraits<Geometry>::memory_category>::host_vector inv_volume( const Geometry& g)
{
    typedef typename HostVec< typename TopologyTraits<Geometry>::memory_category>::host_vector host_vector;
    host_vector temp = volume(g);
    dg::blas1::transform(temp,temp,dg::INVERT<double>());
    return temp;
}

///@}
}//namespace create
}//namespace dg
