#pragma once
// must be included after both derivatives.h, derivativesX.h and mpi_derivatives.h are included
// else there is a problem with ADL
// https://stackoverflow.com/questions/40875453/declare-function-after-template-defined
/*! @file
  @brief Convenience functions to create derivatives
  */
namespace dg{
namespace create{
///@addtogroup creation
///@{
/// Short for <tt> dg::create::derivative( 0, g, bc, dir); </tt>
template<class Topology>
auto dx( const Topology& g, dg::bc bc, dg::direction dir = centered)
{
    static_assert( Topology::ndim() > 0 );
    return derivative( 0, g, bc, dir);
}
/// Short for <tt> dg::create::derivative( 1, g, bc, dir); </tt>
template<class Topology>
auto dy( const Topology& g, dg::bc bc, dg::direction dir = centered)
{
    static_assert( Topology::ndim() > 1 );
    return derivative( 1, g, bc, dir);
}
/// Short for <tt> dg::create::derivative( 2, g, bc, dir); </tt>
template<class Topology>
auto dz( const Topology& g, dg::bc bc, dg::direction dir = centered)
{
    static_assert( Topology::ndim() > 2 );
    return derivative( 2, g, bc, dir);
}

/// Short for <tt> dg::create::jump( 0, g, bc); </tt>
template<class Topology>
auto jumpX( const Topology& g, bc bc)
{
    static_assert( Topology::ndim() > 0 );
    return jump( 0, g, bc);
}
/// Short for <tt> dg::create::jump( 1, g, bc); </tt>
template<class Topology>
auto jumpY( const Topology& g, bc bc)
{
    static_assert( Topology::ndim() > 1 );
    return jump( 1, g, bc);
}
/// Short for <tt> dg::create::jump( 2, g, bc); </tt>
template<class Topology>
auto jumpZ( const Topology& g, bc bc)
{
    static_assert( Topology::ndim() > 2 );
    return jump( 2, g, bc);
}
/// Short for <tt> dg::create::derivative( coord, g, g.bcx(), dir); </tt>
template<class Topology>
auto derivative( unsigned coord, const Topology& g, direction dir = centered)
{
    return derivative( coord, g, g.bcx(), dir);
}
/// Short for <tt> dg::create::jump( coord, g, g.bcx()); </tt>
template<class Topology>
auto jump( unsigned coord, const Topology& g)
{
    return jump( coord, g, g.bcx());
}
/// Short for <tt> dg::create::derivative( 0, g, g.bcx() dir); </tt>
template<class Topology>
auto dx( const Topology& g, dg::direction dir = centered)
{
    static_assert( Topology::ndim() > 0 );
    return derivative( 0, g, g.bcx(), dir);
}
/// Short for <tt> dg::create::derivative( 1, g, g.bcy(), dir); </tt>
template<class Topology>
auto dy( const Topology& g, dg::direction dir = centered)
{
    static_assert( Topology::ndim() > 1 );
    return derivative( 1, g, g.bcy(), dir);
}
/// Short for <tt> dg::create::derivative( 2, g, g.bcz(), dir); </tt>
template<class Topology>
auto dz( const Topology& g, dg::direction dir = centered)
{
    static_assert( Topology::ndim() > 2 );
    return derivative( 2, g, g.bcz(), dir);
}

/// Short for <tt> dg::create::jump( 0, g, g.bcx()); </tt>
template<class Topology>
auto jumpX( const Topology& g)
{
    static_assert( Topology::ndim() > 0 );
    return jump( 0, g, g.bcx());
}
/// Short for <tt> dg::create::jump( 1, g, g.bcy()); </tt>
template<class Topology>
auto jumpY( const Topology& g)
{
    static_assert( Topology::ndim() > 1 );
    return jump( 1, g, g.bcy());
}
/// Short for <tt> dg::create::jump( 2, g, g.bcz()); </tt>
template<class Topology>
auto jumpZ( const Topology& g)
{
    static_assert( Topology::ndim() > 2 );
    return jump( 2, g, g.bcz());
}
///@}
} //namespace create
}//namespace dg
