#pragma once

template<class Geometry, class IMatrix, class Matrix, class Container>
void Explicit<Geometry, IMatrix, Matrix, Container>::operator()(
    double t,
    const std::array<std::array<Container,2>,2>& y,
    std::array<std::array<Container,2>,2>& yp)
{
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

    if( "velocity-staggered" == advection)
    {
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
        dg::geo::ds_slope( m_faST, 1., m_minusSTN[1], m_plusSTN[1], 0., m_dsN[1]);
        dg::geo::ds_average( m_faST, 1., m_minusSTN[1], m_plusSTN[1], 0., m_densityST[1]);

        // Compute dsU and velocity
        m_faST( dg::geo::einsMinus, m_velocityST[1], m_minusSTU[1]);
        m_faST( dg::geo::zeroPlus,  m_velocityST[1], m_plusSTU[1]);
        update_parallel_bc_1st( m_minusSTU[1], m_plusSTU[1], m_p.bcxU, 0.);
        dg::geo::ds_slope( m_faST, 1., m_minusSTU[1], m_plusSTU[1], 0., m_dsU[1]);
        dg::geo::ds_average( m_faST, 1., m_minusSTU[1], m_plusSTU[1], 0.,
                m_velocity[1]);
        }

        // compute qhat
        compute_parallel_flux( m_velocityST[1], m_minusSTN[1], m_plusSTN[1],
                m_dsN[1], m_divNUb[1], m_p.slope_limiter);
        m_faST( dg::geo::zeroPlus, m_divNUb[1], m_plus);
        m_faST( dg::geo::einsMinus, m_divNUb[1], m_minus);
        // We always use NEU for the fluxes for now
        update_parallel_bc_1st( m_minus, m_plus, dg::NEU, 0.);
        // Now compute divNUb and grad U^2/2
        dg::geo::ds_slope( m_faST, 1., m_minus, m_plus, 0, m_divNUb[1]);
        dg::geo::ds_average( m_faST, 1., m_minus, m_plus, 0, m_temp0);
        dg::blas1::pointwiseDot( 1., m_divb, m_temp0, 1., m_divNUb[1]);
        dg::blas1::axpby( -1., m_divNUb[1], 1., yp[0][1]);

        // compute fhat
        compute_parallel_flux( m_velocity[1], m_minusSTU[1], m_plusSTU[1],
                m_dsU[1], m_temp0, m_p.slope_limiter);
        m_faST( dg::geo::einsPlus, m_temp0, m_plus);
        m_faST( dg::geo::zeroMinus, m_temp0, m_minus);
        update_parallel_bc_1st( m_minus, m_plus, dg::NEU, 0.);
        dg::geo::ds_slope( m_faST, -0.5, m_minus, m_plus, 0, yp[1][1]);

        // Add density gradient
        double tau = m_p.tau[1], mu = m_p.mu[1];
        dg::blas1::subroutine( [tau, mu ]DG_DEVICE ( double& gradN,
                    double dsN, double QN, double PN, double hm, double hp)
                {
                    gradN -= tau/mu*dsN/(hm+hp)*(hm/PN + hp/QN);
                },
                yp[1][1], m_dsN[1], m_minusSTN[1], m_plusSTN[1], m_faST.hm(),
                m_faST.hp()
        );
        // Add parallel viscosity
        if( m_p.nu_parallel_u[1] > 0)
        {
            m_fa_diff( dg::geo::einsMinus, m_velocityST[1], m_minus);
            m_fa_diff( dg::geo::einsPlus, m_velocityST[1], m_plus);
            update_parallel_bc_2nd( m_fa_diff, m_minus, m_velocityST[1],
                    m_plus, m_p.bcxU, 0.);
            dg::geo::dssd_centered( m_divb, m_fa_diff, m_p.nu_parallel_u[1],
                    m_minus, m_velocityST[1], m_plus, 0., m_temp0);
            dg::blas1::pointwiseDivide( 1., m_temp0, m_densityST[1], 1., yp[1][1]);
        }
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
        dg::geo::ds_centered( m_fa, 1., m_minus, m_density[1], m_plus, 0., m_dsN[1]);
        if( "centered-forward" == advection)
            dg::geo::ds_forward( m_fa, 1., m_density[1], m_plus, 0., m_temp0);

        m_fa( dg::geo::einsPlus, m_velocity[1], m_plus);
        m_fa( dg::geo::einsMinus, m_velocity[1], m_minus);
        update_parallel_bc_2nd( m_fa, m_minus, m_velocity[1], m_plus, m_p.bcxU,
                0.);
        dg::geo::ds_centered( m_fa, 1., m_minus, m_velocity[1], m_plus, 0.,
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
            m_fa_diff( dg::geo::einsMinus, m_velocity[1], m_minus);
            m_fa_diff( dg::geo::einsPlus, m_velocity[1], m_plus);
            update_parallel_bc_2nd( m_fa_diff, m_minus, m_velocity[1],
                    m_plus, m_p.bcxU, 0.);
            dg::geo::dssd_centered( m_divb, m_fa_diff, m_p.nu_parallel_u[1],
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
            dg::geo::dssd_centered( m_divb, m_fa, m_p.nu_parallel_u[1],
                    m_minus, m_density[1], m_plus, 1., yp[0][1]);
        }
    }
    //-------------Add regularization----------------------------//
    compute_perp_diffusiveN( 1., m_density[1], m_temp0, m_temp1, 1.,
            yp[0][1]);
    compute_perp_diffusiveU( 1., m_velocityST[1], m_temp0, m_temp1, 1.,
            yp[1][1]);

    //Add source terms
    // set m_s
    add_source_terms( yp);

    add_rhs_penalization( yp);

    add_wall_and_sheath_terms( yp);

    dg::blas1::copy( 0., yp[0][0]);
    dg::blas1::copy( 0., yp[1][0]);

    timer.toc();
    accu += timer.diff();
    DG_RANK0 std::cout << "## Add parallel dynamics and sources took "
                       << timer.diff()<<"s\t A: "<<accu<<"\n";
}
