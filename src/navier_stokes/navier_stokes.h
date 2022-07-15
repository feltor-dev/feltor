#pragma once

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::operator()(
    double t,
    const std::array<std::array<Container,2>,2>& y,
    std::array<std::array<Container,2>,2>& yp)
{
    m_called++;
    m_upToDate = false;
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    /* y[0][0] := n
       y[0][1] := n
       y[1][0] := u^\dagger
       y[1][1] := u^\dagger
    */

    dg::Timer timer;
    double accu = 0.;//accumulated time
    timer.tic();

    std::string advection = m_js["advection"].get("type",
            "velocity-staggered").asString();

    if( "velocity-staggered" == advection ||
        "velocity-staggered-implicit" == advection)
    {
        // Compute transformations in two steps
        // compute fluxes on staggered grid, transform afterwards
        dg::blas1::copy( y[0][1], m_density[0]);
        dg::blas1::copy( y[0][1], m_density[1]);

        dg::blas1::copy( 0., m_velocity[0]);
        dg::blas1::copy( y[1][1], m_velocityST[0]);
        dg::blas1::copy( y[1][1], m_velocityST[1]);

        dg::blas1::copy( 0., yp);

        {
        // Compute dsN and staggered density
        m_faST( dg::geo::zeroMinus, m_density[1], m_minusSTN[1]);
        m_faST( dg::geo::einsPlus,  m_density[1], m_plusSTN[1]);
        update_parallel_bc_1st( m_minusSTN[1], m_plusSTN[1], m_p.bcxN, m_p.bcxN ==
                dg::DIR ? m_p.nbc : 0.);
        dg::blas1::axpby( 0.5, m_minusSTN[1], 0.5, m_plusSTN[1], m_densityST[1]);

        // Compute dsU and velocity
        m_faST( dg::geo::einsMinus, m_velocityST[1], m_minusSTU[1]);
        m_faST( dg::geo::zeroPlus,  m_velocityST[1], m_plusSTU[1]);
        update_parallel_bc_1st( m_minusSTU[1], m_plusSTU[1], m_p.bcxU, 0.);
        dg::blas1::axpby( 0.5, m_minusSTU[1], 0.5, m_plusSTU[1], m_velocity[1]);
        }

        // compute qhat
        compute_parallel_flux( m_velocityST[1], m_minusSTN[1], m_plusSTN[1],
                m_divNUb[1], m_p.slope_limiter);
        m_faST( dg::geo::zeroPlus, m_divNUb[1], m_plus);
        m_faST( dg::geo::einsMinus, m_divNUb[1], m_minus);
        // We always use NEU for the fluxes for now
        update_parallel_bc_1st( m_minus, m_plus, dg::NEU, 0.);
        // Now compute divNUb
        dg::geo::ds_divCentered( m_faST, 1., m_minus, m_plus, 0., m_divNUb[1]);
        dg::blas1::axpby( -1., m_divNUb[1], 1., yp[0][1]);

        // compute fhat
        compute_parallel_flux( m_velocity[1], m_minusSTU[1], m_plusSTU[1],
                m_temp0, m_p.slope_limiter);
        m_faST( dg::geo::einsPlus, m_temp0, m_plus);
        m_faST( dg::geo::zeroMinus, m_temp0, m_minus);
        update_parallel_bc_1st( m_minus, m_plus, dg::NEU, 0.);
        dg::geo::ds_centered( m_faST, -0.5, m_minus, m_plus, 1, yp[1][1]);

        // Add density gradient
        if( advection == "velocity-staggered")
        {
            double tau = m_p.tau[1], mu = m_p.mu[1], delta = m_fa.deltaPhi();
            dg::blas1::subroutine( [tau, mu, delta ]DG_DEVICE ( double& UDot,
                        double QN, double PN, double bphi)
                    {
                        UDot -= tau/mu*bphi*(PN-QN)/delta/2.*(1/PN + 1/QN);
                    },
                    yp[1][1], m_minusSTN[1], m_plusSTN[1], m_fa.bphi()
            );
        }
        // Add parallel viscosity
        if( m_p.nu_parallel_u[1] > 0)
        {
            m_fa( dg::geo::einsMinus, m_velocityST[1], m_minus);
            m_fa( dg::geo::einsPlus, m_velocityST[1], m_plus);
            update_parallel_bc_2nd( m_fa, m_minus, m_velocityST[1],
                    m_plus, m_p.bcxU, 0.);
            dg::geo::dssd_centered( m_fa, m_p.nu_parallel_u[1],
                    m_minus, m_velocityST[1], m_plus, 0., m_temp0);
            dg::blas1::pointwiseDivide( 1., m_temp0, m_densityST[1], 1., yp[1][1]);
        }
    }
    else if( "velocity-staggered-fieldaligned" == advection ||
        "velocity-staggered-fieldaligned-implicit" == advection)
    {
        // compute fluxes and derivatives on locally fieldaligned grid centered on plane k (for n) and k+1/2 (for u)
        // i.e. do all trafos first and compute fluxes in one go
        dg::blas1::copy( y[0][1], m_density[0]);
        dg::blas1::copy( y[0][1], m_density[1]);

        dg::blas1::copy( 0., m_velocity[0]);
        dg::blas1::copy( y[1][1], m_velocityST[0]);
        dg::blas1::copy( y[1][1], m_velocityST[1]);

        dg::blas1::copy( 0., yp);

        // Compute transformed fieldaligned grid
        m_faST( dg::geo::einsMinus, m_velocityST[1], m_minusSTU[1]);
        m_faST( dg::geo::zeroPlus,  m_velocityST[1], m_plusSTU[1]);
        update_parallel_bc_1st( m_minusSTU[1], m_plusSTU[1], m_p.bcxU, 0.);
        dg::blas1::axpby( 0.5, m_minusSTU[1], 0.5, m_plusSTU[1], m_velocity[1]);
        m_fa( dg::geo::einsMinus, m_density[1], m_minusN[1]);
        m_fa( dg::geo::einsPlus,  m_density[1], m_plusN[1]);
        update_parallel_bc_2nd( m_fa, m_minusN[1], m_density[1], m_plusN[1],
                m_p.bcxN, m_p.bcxN == dg::DIR ? m_p.nbc : 0.);
        // compute qhat
        compute_parallel_flux( m_minusSTU[1], m_plusSTU[1],
                m_minusN[1], m_density[1], m_plusN[1],
                m_fluxM, m_fluxP, m_p.slope_limiter);
        // Now compute divNUb
        dg::geo::ds_divCentered( m_faST, 1., m_fluxM, m_fluxP, 0.,
                m_divNUb[1]);
        dg::blas1::axpby( -1., m_divNUb[1], 1., yp[0][1]);


        // Compute transformed fieldaligned ADJOINT grid
        m_faST( dg::geo::zeroMinus, m_density[1], m_minusSTN[1]);
        m_faST( dg::geo::einsPlus,  m_density[1], m_plusSTN[1]);
        update_parallel_bc_1st( m_minusSTN[1], m_plusSTN[1],
                m_p.bcxN, m_p.bcxN == dg::DIR ? m_p.nbc : 0.);
        dg::blas1::axpby( 0.5, m_minusSTN[1], 0.5, m_plusSTN[1], m_densityST[1]);
        m_fa( dg::geo::einsMinus, m_velocityST[1], m_minusU[1]);
        m_fa( dg::geo::einsPlus,  m_velocityST[1], m_plusU[1]);
        update_parallel_bc_2nd( m_fa, m_minusU[1], m_velocityST[1],
                m_plusU[1], m_p.bcxU, 0.);

        // compute fhat
        dg::blas1::axpby( 0.25, m_minusU[1], 0.25, m_velocityST[1], m_minusSTU[1]);
        dg::blas1::axpby( 0.25, m_velocityST[1], 0.25, m_plusU[1], m_plusSTU[1]);
        compute_parallel_flux( m_minusSTU[1], m_plusSTU[1],
                m_minusU[1], m_velocityST[1], m_plusU[1],
                m_fluxM, m_fluxP,
                m_p.slope_limiter);
        dg::geo::ds_centered( m_faST, -1., m_fluxM, m_fluxP, 0., m_temp0 );
        // Add density gradient
        if( advection == "velocity-staggered-fieldaligned")
        {
            double tau = m_p.tau[1], mu = m_p.mu[1], delta = m_fa.deltaPhi();
            dg::blas1::subroutine( [tau, mu, delta ]DG_DEVICE ( double& UDot,
                        double QN, double PN, double bphi)
                    {
                        UDot -= tau/mu*bphi*(PN-QN)/delta/2.*(1/PN + 1/QN);
                    },
                    m_temp0, m_minusSTN[1], m_plusSTN[1], m_fa.bphi()
            );
        }

        // Add parallel viscosity
        if( m_p.nu_parallel_u[1] > 0)
        {
            m_fa( dg::geo::einsMinus, m_velocityST[1], m_minus);
            m_fa( dg::geo::einsPlus, m_velocityST[1], m_plus);
            update_parallel_bc_2nd( m_fa, m_minus, m_velocityST[1],
                    m_plus, m_p.bcxU, 0.);
            dg::geo::dssd_centered( m_fa, m_p.nu_parallel_u[1],
                    m_minus, m_velocityST[1], m_plus, 0., m_temp1);
            dg::blas1::pointwiseDivide( 1., m_temp1, m_densityST[1], 1., m_temp0);
        }
        dg::blas1::axpby( 1., m_temp0, 1., yp[1][1]);
    }
    else if( "centered" == advection || "centered-forward" == advection)
    {
        dg::blas1::copy( y[0][1], m_density[0]);
        dg::blas1::copy( y[0][1], m_density[1]);
        dg::blas1::copy( y[0][1], m_densityST[0]),
        dg::blas1::copy( y[0][1], m_densityST[1]),
        dg::blas1::copy( y[1][1], m_velocity[0]);
        dg::blas1::copy( y[1][1], m_velocity[1]);

        dg::blas1::copy( y[1][1], m_velocityST[0]);
        dg::blas1::copy( y[1][1], m_velocityST[1]);

        dg::blas1::copy( 0., yp);
        // Now compute divNUb and grad U^2/2
        m_fa( dg::geo::einsPlus, m_density[1], m_plus);
        m_fa( dg::geo::einsMinus, m_density[1], m_minus);
        update_parallel_bc_2nd( m_fa, m_minus, m_density[1], m_plus, m_p.bcxN,
                m_p.bcxN == dg::DIR ? m_p.nbc : 0.);
        dg::geo::ds_centered( m_fa, 1., m_minus, m_plus, 0., m_dsN[1]);
        if( "centered-forward" == advection)
            dg::geo::ds_forward( m_fa, 1., m_density[1], m_plus, 0., m_temp0);

        m_fa( dg::geo::einsPlus, m_velocity[1], m_plus);
        m_fa( dg::geo::einsMinus, m_velocity[1], m_minus);
        update_parallel_bc_2nd( m_fa, m_minus, m_velocity[1], m_plus, m_p.bcxU,
                0.);
        dg::geo::ds_centered( m_fa, 1., m_minus, m_plus, 0.,
                m_dsU[1]);
        if( "centered-forward" == advection)
            dg::geo::ds_backward( m_fa, 1., m_minus, m_velocity[1], 0., m_temp1);

        dg::blas1::pointwiseDot( 1., m_velocity[1], m_density[1], m_divb, 0.,
                m_divNUb[1]);
        dg::blas1::pointwiseDot( 1., m_velocity[1], m_dsN[1], 1., m_divNUb[1]);
        if( "centered-forward" == advection)
            dg::blas1::pointwiseDot( 1., m_density[1], m_temp1, 1., m_divNUb[1]);
        else
            dg::blas1::pointwiseDot( 1., m_density[1], m_dsU[1], 1., m_divNUb[1]);
        dg::blas1::axpby( -1., m_divNUb[1], 1., yp[0][1]);

        // U dsU
        dg::blas1::pointwiseDot( 2., m_velocity[1], m_dsU[1], 0., m_dsU[1]);
        dg::blas1::axpby( -1./2., m_dsU[1], 1., yp[1][1]);
        if( "centered-forward" == advection)
            dg::blas1::pointwiseDivide( -m_p.tau[1]/m_p.mu[1], m_temp0,
                m_density[1], 1., yp[1][1]);
        else
            dg::blas1::pointwiseDivide( -m_p.tau[1]/m_p.mu[1], m_dsN[1],
                m_density[1], 1., yp[1][1]);
        // Add parallel viscosity
        if( m_p.nu_parallel_u[1] > 0)
        {
            m_fa( dg::geo::einsMinus, m_velocity[1], m_minus);
            m_fa( dg::geo::einsPlus, m_velocity[1], m_plus);
            update_parallel_bc_2nd( m_fa, m_minus, m_velocity[1],
                    m_plus, m_p.bcxU, 0.);
            dg::geo::dssd_centered( m_fa, m_p.nu_parallel_u[1],
                    m_minus, m_velocity[1], m_plus, 0., m_temp0);
            dg::blas1::pointwiseDivide( 1., m_temp0, m_density[1], 1., yp[1][1]);
        }

    }
    else if( "diffusion" == advection)
    {
        // solve diffusion equation in density
        dg::blas1::copy( y[0], m_density);
        dg::blas1::copy( y[0], m_densityST);
        dg::blas1::copy( 0., yp);
        // Add parallel viscosity
        if( m_p.nu_parallel_u[1] > 0)
        {
            // here we can try out difference between linear and cubic
            m_fa( dg::geo::einsMinus, y[0][1], m_minus);
            m_fa( dg::geo::einsPlus, y[0][1], m_plus);
            update_parallel_bc_2nd( m_fa, m_minus, m_density[1], m_plus,
                    m_p.bcxN, m_p.bcxN == dg::DIR ? m_p.nbc : 0.);
            dg::geo::dssd_centered( m_fa, m_p.nu_parallel_u[1],
                    m_minus, m_density[1], m_plus, 1., yp[0][1]);
        }
    }
    else if( "log-staggered" == advection)
    {
        // density is in log(n)
        dg::blas1::transform( y[0][1], m_density[0], dg::EXP<double>());
        dg::blas1::transform( y[0][1], m_density[1], dg::EXP<double>());

        dg::blas1::copy( 0., m_velocity[0]);
        dg::blas1::copy( y[1][1], m_velocityST[0]);
        dg::blas1::copy( y[1][1], m_velocityST[1]);

        dg::blas1::copy( 0., yp);

        {
        // Compute dsN and staggered density
        m_faST( dg::geo::zeroMinus, y[0][1], m_minusSTN[1]);
        m_faST( dg::geo::einsPlus,  y[0][1], m_plusSTN[1]);
        update_parallel_bc_1st( m_minusSTN[1], m_plusSTN[1], m_p.bcxN, m_p.bcxN ==
                dg::DIR ? m_p.nbc : 0.);

        // Compute dsU and velocity
        m_faST( dg::geo::einsMinus, m_velocityST[1], m_minusSTU[1]);
        m_faST( dg::geo::zeroPlus,  m_velocityST[1], m_plusSTU[1]);
        update_parallel_bc_1st( m_minusSTU[1], m_plusSTU[1], m_p.bcxU, 0.);
        dg::blas1::axpby( 0.5, m_minusSTU[1], 0.5, m_plusSTU[1], m_velocity[1]);
        }

        // compute qhat with ln N
        compute_parallel_advection( m_velocityST[1], m_minusSTN[1], m_plusSTN[1],
                m_divNUb[1], m_p.slope_limiter);
        m_faST( dg::geo::zeroPlus, m_divNUb[1], m_plus);
        m_faST( dg::geo::einsMinus, m_divNUb[1], m_minus);
        // We always use NEU for the fluxes for now
        update_parallel_bc_1st( m_minus, m_plus, dg::NEU, 0.);
        dg::geo::ds_centered( m_faST, 1., m_minus, m_plus, 0, m_divNUb[1]);
        dg::blas1::pointwiseDot( m_velocity[1], m_divNUb[1], m_divNUb[1]);
        dg::geo::ds_divCentered( m_faST, 1., m_minusSTU[1], m_plusSTU[1], 1., m_divNUb[1]);
        dg::blas1::axpby( -1., m_divNUb[1], 1., yp[0][1]);
        dg::blas1::pointwiseDot( m_density[1], m_divNUb[1], m_divNUb[1]);

        // compute fhat
        compute_parallel_flux( m_velocity[1], m_minusSTU[1], m_plusSTU[1],
                m_temp0, m_p.slope_limiter);
        m_faST( dg::geo::einsPlus, m_temp0, m_plus);
        m_faST( dg::geo::zeroMinus, m_temp0, m_minus);
        update_parallel_bc_1st( m_minus, m_plus, dg::NEU, 0.);
        dg::geo::ds_centered( m_faST, -0.5, m_minus, m_plus, 1, yp[1][1]);

        // Add density gradient
        dg::geo::ds_centered( m_faST, -m_p.tau[1] /m_p.mu[1], m_minusSTN[1], m_plusSTN[1], 1., yp[1][1]);
        // Add parallel viscosity
        if( m_p.nu_parallel_u[1] > 0)
        {
            m_faST( dg::geo::zeroMinus, m_density[1], m_minusSTN[1]);
            m_faST( dg::geo::einsPlus,  m_density[1], m_plusSTN[1]);
            update_parallel_bc_1st( m_minusSTN[1], m_plusSTN[1], m_p.bcxN, m_p.bcxN ==
                    dg::DIR ? m_p.nbc : 0.);
            dg::blas1::axpby( 0.5, m_minusSTN[1], 0.5, m_plusSTN[1], m_densityST[1]);
            m_fa( dg::geo::einsMinus, m_velocityST[1], m_minus);
            m_fa( dg::geo::einsPlus, m_velocityST[1], m_plus);
            update_parallel_bc_2nd( m_fa, m_minus, m_velocityST[1],
                    m_plus, m_p.bcxU, 0.);
            dg::geo::dssd_centered( m_fa, m_p.nu_parallel_u[1],
                    m_minus, m_velocityST[1], m_plus, 0., m_temp0);
            dg::blas1::pointwiseDivide( 1., m_temp0, m_densityST[1], 1., yp[1][1]);
        }
    }
    else if( "staggered" == advection || "staggered-implicit" == advection)
    {
        // Compute transformations in two steps
        // compute fluxes on staggered grid
        dg::blas1::copy( y[0][1], m_density[0]);
        dg::blas1::copy( y[0][1], m_density[1]);


        dg::blas1::copy( 0., m_velocity[0]);
        dg::blas1::copy( 0., yp);

        { // Compute dsN and staggered density
        m_faST( dg::geo::zeroMinus, m_density[1], m_minusSTN[1]);
        m_faST( dg::geo::einsPlus,  m_density[1], m_plusSTN[1]);
        update_parallel_bc_1st( m_minusSTN[1], m_plusSTN[1], m_p.bcxN, m_p.bcxN ==
                dg::DIR ? m_p.nbc : 0.);
        dg::blas1::axpby( 0.5, m_minusSTN[1], 0.5, m_plusSTN[1], m_densityST[1]);
        }
        dg::blas1::pointwiseDivide( y[1][1], m_densityST[1], m_velocityST[0]); //NU^st
        dg::blas1::pointwiseDivide( y[1][1], m_densityST[1], m_velocityST[1]); //NU^st

        { // Compute dsU and velocity
        m_faST( dg::geo::einsMinus, m_velocityST[1], m_minusSTU[1]);
        m_faST( dg::geo::zeroPlus,  m_velocityST[1], m_plusSTU[1]);
        update_parallel_bc_1st( m_minusSTU[1], m_plusSTU[1], m_p.bcxU, 0.);
        dg::blas1::axpby( 0.5, m_minusSTU[1], 0.5, m_plusSTU[1], m_velocity[1]);
        }

        // compute qhat
        compute_parallel_flux( m_velocityST[1], m_minusSTN[1], m_plusSTN[1],
                m_divNUb[1], m_p.slope_limiter);
        m_faST( dg::geo::zeroPlus, m_divNUb[1], m_plus);
        m_faST( dg::geo::einsMinus, m_divNUb[1], m_minus);
        // We always use NEU for the fluxes for now
        update_parallel_bc_1st( m_minus, m_plus, dg::NEU, 0.);
        // Now compute divNUb
        dg::geo::ds_divCentered( m_faST, 1., m_minus, m_plus, 0, m_divNUb[1]);
        dg::blas1::axpby( 0.5, m_minus, 0.5, m_plus, m_fluxM); // qhat
        dg::blas1::axpby( -1., m_divNUb[1], 1., yp[0][1]);

        // compute fhat
        compute_parallel_flux( m_fluxM, m_minusSTU[1], m_plusSTU[1],
                m_fluxP, m_p.slope_limiter);
        m_faST( dg::geo::einsPlus, m_fluxP, m_plus);
        m_faST( dg::geo::zeroMinus, m_fluxP, m_minus);
        update_parallel_bc_1st( m_minus, m_plus, dg::NEU, 0.);
        dg::geo::ds_centered( m_faST, -1., m_minus, m_plus, 1, yp[1][1]);

        // Add density gradient
        if( advection == "staggered")
            dg::geo::ds_centered(m_faST, -m_p.tau[1] /m_p.mu[1], m_minusSTN[1], m_plusSTN[1], 1., yp[1][1]);
        // Add parallel viscosity
        if( m_p.nu_parallel_u[1] > 0)
        {
            m_fa( dg::geo::einsMinus, m_velocityST[1], m_minus);
            m_fa( dg::geo::einsPlus, m_velocityST[1], m_plus);
            update_parallel_bc_2nd( m_fa, m_minus, m_velocityST[1],
                    m_plus, m_p.bcxU, 0.);
            dg::geo::dssd_centered( m_fa, m_p.nu_parallel_u[1],
                    m_minus, m_velocityST[1], m_plus, 1., yp[1][1]);
        }
    }
    else if( "staggered-fieldaligned" == advection || "staggered-fieldaligned-implicit" == advection)
    {
        // compute fluxes and derivatives on locally fieldaligned grid centered on plane k (for n) and k+1/2 (for u)
        // i.e. do all trafos first and compute fluxes in one go
        dg::blas1::copy( y[0][1], m_density[0]);
        dg::blas1::copy( y[0][1], m_density[1]);


        dg::blas1::copy( 0., m_velocity[0]);
        dg::blas1::copy( 0., yp);

        {
        // Compute transformed fieldaligned ADJOINT grid
        m_faST( dg::geo::zeroMinus, m_density[1], m_minusSTN[1]);
        m_faST( dg::geo::einsPlus,  m_density[1], m_plusSTN[1]);
        update_parallel_bc_1st( m_minusSTN[1], m_plusSTN[1],
                m_p.bcxN, m_p.bcxN == dg::DIR ? m_p.nbc : 0.);
        dg::blas1::axpby( 0.5, m_minusSTN[1], 0.5, m_plusSTN[1], m_densityST[1]);
        }
        dg::blas1::pointwiseDivide( y[1][1], m_densityST[1], m_velocityST[0]); //NU^st
        dg::blas1::pointwiseDivide( y[1][1], m_densityST[1], m_velocityST[1]); //NU^st

        // Compute transformed fieldaligned grid
        {
        m_faST( dg::geo::einsMinus, m_velocityST[1], m_minusSTU[1]);
        m_faST( dg::geo::zeroPlus,  m_velocityST[1], m_plusSTU[1]);
        update_parallel_bc_1st( m_minusSTU[1], m_plusSTU[1], m_p.bcxU, 0.);
        dg::blas1::axpby( 0.5, m_minusSTU[1], 0.5, m_plusSTU[1], m_velocity[1]);
        }
        m_fa( dg::geo::einsMinus, m_density[1], m_minusN[1]);
        m_fa( dg::geo::einsPlus,  m_density[1], m_plusN[1]);
        update_parallel_bc_2nd( m_fa, m_minusN[1], m_density[1], m_plusN[1],
                m_p.bcxN, m_p.bcxN == dg::DIR ? m_p.nbc : 0.);

        // compute qhat
        compute_parallel_flux( m_minusSTU[1], m_plusSTU[1],
                m_minusN[1], m_density[1], m_plusN[1],
                m_fluxM, m_fluxP, m_p.slope_limiter);
        // Now compute divNUb
        dg::geo::ds_divCentered( m_faST, 1., m_fluxM, m_fluxP, 0.,
                m_divNUb[1]);
        dg::blas1::axpby( -1., m_divNUb[1], 1., yp[0][1]);

        // Compute transformed fieldaligned ADJOINT grid
        dg::blas1::axpby( 0.5, m_fluxM, 0.5, m_fluxP, m_temp0); // qhat
        m_faST( dg::geo::zeroMinus, m_temp0, m_fluxM);
        m_faST( dg::geo::einsPlus,  m_temp0, m_fluxP);
        update_parallel_bc_1st( m_fluxM, m_fluxP,
                m_p.bcxN, m_p.bcxN == dg::DIR ? m_p.nbc : 0.);

        m_fa( dg::geo::einsMinus, m_velocityST[1], m_minusU[1]);
        m_fa( dg::geo::einsPlus,  m_velocityST[1], m_plusU[1]);
        update_parallel_bc_2nd( m_fa, m_minusU[1], m_velocityST[1],
                m_plusU[1], m_p.bcxU, 0.);

        // compute fhat
        compute_parallel_flux( m_fluxM, m_fluxP,
                m_minusU[1], m_velocityST[1], m_plusU[1],
                m_minus, m_plus,
                m_p.slope_limiter);
        dg::geo::ds_divCentered( m_faST, -1., m_minus, m_plus, 1, yp[1][1]);

        // Add density gradient
        if( advection == "staggered-fieldaligned")
            dg::geo::ds_centered(m_faST, -m_p.tau[1] /m_p.mu[1], m_minusSTN[1], m_plusSTN[1], 1., yp[1][1]);
        // Add parallel viscosity
        if( m_p.nu_parallel_u[1] > 0)
        {
            m_fa( dg::geo::einsMinus, m_velocityST[1], m_minus);
            m_fa( dg::geo::einsPlus, m_velocityST[1], m_plus);
            update_parallel_bc_2nd( m_fa, m_minus, m_velocityST[1],
                    m_plus, m_p.bcxU, 0.);
            dg::geo::dssd_centered( m_fa, m_p.nu_parallel_u[1],
                    m_minus, m_velocityST[1], m_plus, 1., yp[1][1]);
        }
    }
    else if( "velocity-unstaggered" == advection)
    {
        // quite unstable for some reason (timer just stops)
        dg::blas1::copy( y[0][1], m_density[0]);
        dg::blas1::copy( y[0][1], m_density[1]);

        dg::blas1::copy( y[1][1], m_velocity[0]);
        dg::blas1::copy( y[1][1], m_velocity[1]);

        dg::blas1::copy( 0., yp);

        {
        // Compute dsN and staggered density
        m_faST( dg::geo::zeroMinus, m_density[1], m_minusSTN[1]);
        m_faST( dg::geo::einsPlus,  m_density[1], m_plusSTN[1]);
        update_parallel_bc_1st( m_minusSTN[1], m_plusSTN[1], m_p.bcxN, m_p.bcxN ==
                dg::DIR ? m_p.nbc : 0.);
        dg::blas1::axpby( 0.5, m_minusSTN[1], 0.5, m_plusSTN[1], m_densityST[1]);

        // Compute dsU and velocity
        m_faST( dg::geo::zeroMinus, m_velocity[1], m_minusSTU[1]);
        m_faST( dg::geo::einsPlus,  m_velocity[1], m_plusSTU[1]);
        update_parallel_bc_1st( m_minusSTU[1], m_plusSTU[1], m_p.bcxU, 0.);
        dg::blas1::axpby( 0.5, m_minusSTU[1], 0.5, m_plusSTU[1], m_velocityST[1]);
        }

        // compute qhat
        compute_parallel_flux( m_velocityST[1], m_minusSTN[1], m_plusSTN[1],
                m_divNUb[1], m_p.slope_limiter);
        m_faST( dg::geo::zeroPlus, m_divNUb[1], m_plus);
        m_faST( dg::geo::einsMinus, m_divNUb[1], m_minus);
        // We always use NEU for the fluxes for now
        update_parallel_bc_1st( m_minus, m_plus, dg::NEU, 0.);
        // Now compute divNUb and grad U^2/2
        dg::geo::ds_divCentered( m_faST, 1., m_minus, m_plus, 0, m_divNUb[1]);
        dg::blas1::axpby( -1., m_divNUb[1], 1., yp[0][1]);

        // compute fhat
        compute_parallel_flux( m_velocityST[1], m_minusSTU[1], m_plusSTU[1],
                m_fluxM, m_p.slope_limiter);
        m_faST( dg::geo::zeroPlus, m_fluxM, m_plus);
        m_faST( dg::geo::einsMinus, m_fluxM, m_minus);
        update_parallel_bc_1st( m_minus, m_plus, dg::NEU, 0.);
        dg::geo::ds_centered( m_faST, -0.5, m_minus, m_plus, 1, yp[1][1]);

        // density m_dsN
        m_fa( dg::geo::einsMinus, m_density[1], m_minus);
        m_fa( dg::geo::einsPlus,  m_density[1], m_plus);
        update_parallel_bc_2nd( m_fa, m_minus, m_density[1], m_plus,
                m_p.bcxN, m_p.bcxN == dg::DIR ? m_p.nbc : 0.);
        dg::geo::ds_centered( m_fa, 1., m_minus, m_plus, 0., m_dsN[1]);
        // parallel pressure gradient
        dg::blas1::pointwiseDivide( -m_p.tau[1]/m_p.mu[1], m_dsN[1],
                m_density[1], 1., yp[1][1]);
        // Add parallel viscosity
        if( m_p.nu_parallel_u[1] > 0)
        {
            m_fa( dg::geo::einsMinus, m_velocity[1], m_minus);
            m_fa( dg::geo::einsPlus, m_velocity[1], m_plus);
            update_parallel_bc_2nd( m_fa, m_minus, m_velocity[1],
                    m_plus, m_p.bcxU, 0.);
            dg::geo::dssd_centered( m_fa, m_p.nu_parallel_u[1],
                    m_minus, m_velocity[1], m_plus, 0., m_temp0);
            dg::blas1::pointwiseDivide( 1., m_temp0, m_density[1], 1., yp[1][1]);
        }
        // to get the diffusion right
        dg::blas1::copy( m_velocity[1], m_velocityST[1]);
    }
    //-------------Add regularization----------------------------//
    if( "log-staggered" == advection)
    {
        compute_perp_diffusiveN( 1., m_density[1], m_temp0, m_temp1, 0.,
            m_temp1);
        dg::blas1::pointwiseDivide( 1., m_temp1, m_density[1], 1., yp[0][1]);
    }
    else
        compute_perp_diffusiveN( 1., m_density[1], m_temp0, m_temp1, 1.,
            yp[0][1]);
    compute_perp_diffusiveU( 1., m_velocityST[1], m_densityST[1], m_temp0, m_temp1, 1.,
            yp[1][1]);

    // partitioned means imex timestepper
    for( unsigned i=0; i<2; i++)
    {
        for( unsigned j=0; j<2; j++)
            multiply_rhs_penalization( yp[i][j]); // F*(1-chi_w-chi_s)
    }

    //Add source terms
    // set m_s
    add_source_terms( yp);

    add_wall_and_sheath_terms( yp);

    dg::blas1::copy( 0., yp[0][0]);
    dg::blas1::copy( 0., yp[1][0]);

    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Add parallel dynamics and sources took "
                       << timer.diff()<<"s\t A: "<<accu<<"\n";
}

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::implicit(
        double t,
        const std::array<std::array<Container,2>,2> & y,
        std::array<std::array<Container,2>,2>& yp)
{
    dg::blas1::copy( 0., yp);
    std::string advection = m_js["advection"].get("type",
        "velocity-staggered").asString();
    if( advection == "staggered-implicit" ||
        advection == "staggered-fieldaligned-implicit")
    {
        dg::blas1::copy( y[0], m_density),
        // Compute dsN and staggered density
        m_faST( dg::geo::zeroMinus, m_density[1], m_minusSTN[1]);
        m_faST( dg::geo::einsPlus,  m_density[1], m_plusSTN[1]);
        update_parallel_bc_1st( m_minusSTN[1], m_plusSTN[1], m_p.bcxN, m_p.bcxN ==
                dg::DIR ? m_p.nbc : 0.);
        dg::geo::ds_slope( m_faST, 1., m_minusSTN[1], m_plusSTN[1], 0., m_dsN[1]);
        dg::blas1::axpby( 0.5, m_minusSTN[1], 0.5, m_plusSTN[1], m_densityST[1]);
        // Add density gradient
        dg::blas1::axpby( -m_p.tau[1] /m_p.mu[1], m_dsN[1], 1., yp[1][1]);
    }
    else if( advection == "velocity-staggered-implicit" ||
         advection == "velocity-staggered-fieldaligned-implicit")
    {
        dg::blas1::copy( y[0], m_density),

        // Compute dsN and staggered density
        m_faST( dg::geo::zeroMinus, m_density[1], m_minusSTN[1]);
        m_faST( dg::geo::einsPlus,  m_density[1], m_plusSTN[1]);
        update_parallel_bc_1st( m_minusSTN[1], m_plusSTN[1], m_p.bcxN, m_p.bcxN ==
                dg::DIR ? m_p.nbc : 0.);

        double tau = m_p.tau[1], mu = m_p.mu[1], delta = m_fa.deltaPhi();
        dg::blas1::subroutine( [tau, mu, delta ]DG_DEVICE ( double& UDot,
                double QN, double PN, double bphi)
            {
                UDot -= tau/mu*bphi*(PN-QN)/delta/2.*(1/PN + 1/QN);
            },
            yp[1][1], m_minusSTN[1], m_plusSTN[1], m_fa.bphi()
        );
    }
}

template< class Geometry, class IMatrix, class Matrix, class Container >
struct ImplicitSolver
{
    ImplicitSolver() {}
    ImplicitSolver( Explicit<Geometry,IMatrix,Matrix,Container>& ex, double eps_time): m_ex(&ex)    {
        dg::assign( dg::evaluate( dg::zero, ex.grid()), m_tmp[0][0] );
        m_tmp[0][1] = m_tmp[0][0];
        m_tmp[1] = m_tmp[0];
    }
    const std::array<std::array<Container,2>,2>& copyable() const{
        return m_tmp;
    }
    // solve (y + alpha I(t,y) = rhs
    void operator()( double alpha,
            double t,
            std::array<std::array<Container,2>,2>& y,
            const std::array<std::array<Container,2>,2>& rhs)
    {
        dg::blas1::copy( rhs[0], y[0]);// I_n = 0
        m_ex->implicit( t, y, m_tmp); //ignores y[1], solves Poisson at time t for y[0] and
        // writes 0 in m_tmp[0] and updates m_tmp[1]
        dg::blas1::axpby( 1., rhs[1], +alpha, m_tmp[1], y[1]); // u = rhs_u + alpha I_u
    }
    private:
    std::array<std::array<Container,2>,2> m_tmp;
    Explicit<Geometry,IMatrix,Matrix,Container>* m_ex; // does not own anything
};
// The following is still for the NavierStokes project
// (turns out doing the parallel force implicitly does not help with anything)
template< class Geometry, class IMatrix, class Matrix, class Container >
struct Implicit
{
    Implicit() {}
    Implicit( Explicit<Geometry,IMatrix,Matrix,Container>& ex)
        : m_ex(&ex){}
    void operator() ( double t,
            const std::array<std::array<Container,2>,2> & y,
            std::array<std::array<Container,2>,2>& yp)
    {
        m_ex->implicit( t, y, yp);
    }
    private:
    Explicit<Geometry,IMatrix,Matrix,Container>* m_ex; // does not own anything
};
