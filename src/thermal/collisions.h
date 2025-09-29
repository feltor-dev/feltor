#pragma once

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "parameters.h"

namespace thermal
{
template< class Geometry, class IMatrix, class Matrix, class Container >
struct Collisions
{
    Collisions( const Geometry& g, thermal::Parameters p,
        dg::geo::TokamakMagneticField mag, dg::file::WrappedJsonValue js):m_p(p), m_js(js)
    {
        dg::assign( dg::evaluate( dg::zero, g), m_temp0 );
        m_temp1 = m_temp0;
        dg::assign(  dg::pullback(dg::geo::Bmodule(mag), g), m_B2);
        dg::blas1::pointwiseDot( m_B2, m_B2, m_B2);
        m_lapperpM.construct ( g, p.bcx, p.bcy,  p.diff_dir);
        m_R0 = mag.R0();
    }

    void add_coulomb_collisions(
        unsigned s,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp);
    void add_lorentz_collisions(
        unsigned s,
        const std::map<std::string, std::vector<Container>>& q,
        const std::array<std::vector<Container>,6>& y,
        std::array<std::vector<Container>,6>& yp);
    private:
    const thermal::Parameters m_p;
    const dg::file::WrappedJsonValue m_js;
    dg::Elliptic2d< Geometry, Matrix, Container> m_lapperpM;
    Container m_temp0, m_temp1, m_B2;
    double m_R0;
};
template<class Grid, class IMatrix, class Matrix, class Container>
void Collisions<Grid, IMatrix, Matrix, Container>::add_coulomb_collisions(
        unsigned s,
        const std::map<std::string, std::vector<Container>>& q,
        std::array<std::vector<Container>,6>& yp)
{
    double nu_ref = m_p.nu_ref;
    double zs = m_p.z[s], mus = m_p.mu[s];
    // CPperp
    dg::blas1::copy( 0., m_temp0);
    for( unsigned k=0; k<m_p.num_species; k++)
        if( s != k)
        {
            double zk = m_p.z[k];
            double muk = m_p.mu[k];
            dg::blas1::subroutine( [nu_ref, zs, zk, mus, muk] DG_DEVICE(
                double& spperp, double ns, double nk, double ts, double tk)
            {
                spperp += -2*nu_ref*ns*nk*zs*zs*zk*zk/mus/muk/(
                    ts/mus + tk/muk)/sqrt( ts/mus + tk/muk)*(ts-tk);
            }, m_temp0, q.at("N")[s], q.at("N")[k], q.at("Tperp")[s], q.at("Tperp")[k]);
        }
    dg::blas1::axpby( 1., m_temp0, 1., yp[1][s]);


    // CN
    dg::blas1::pointwiseDivide( m_p.mu[s]/2./m_p.z[s]/m_p.z[s], m_temp0, m_B2, 0., m_temp0);
    dg::blas2::symv( m_lapperpM, m_temp0, m_temp1);
    dg::blas1::axpby( 1., m_temp1, 1., yp[0][s]);

    // CPpara
    dg::blas1::pointwiseDot( m_p.mu[s],  q.at("U")[s], q.at("U")[s], m_temp1, 0., m_temp0); // mu U^2 S_N
    for( unsigned k=0; k<m_p.num_species; k++)
        if( s != k)
        {
            double zs = m_p.z[s], zk = m_p.z[k];
            double mus = m_p.mu[s], muk = m_p.mu[k];
            dg::blas1::subroutine( [nu_ref, zs, zk, mus, muk] DG_DEVICE(
                double& sppara, double ns, double nk, double ts, double tk, double us, double uk)
            {
                sppara += -2*nu_ref*ns*nk*zs*zs*zk*zk/mus/muk/(
                    ts/mus + tk/muk)/sqrt( ts/mus + tk/muk)*((ts-tk) - 0.51*muk*(us-uk)*(us-uk));
            }, m_temp0, q.at("N")[s], q.at("N")[k], q.at("Tpara")[s], q.at("Tpara")[k], q.at("U")[s], q.at("U")[k]);
        }
    dg::blas1::axpby( 1., m_temp0, 1., yp[2][s]);


    // CPperp ST
    dg::blas1::copy( 0., m_temp0);
    for( unsigned k=0; k<m_p.num_species; k++)
        if( s != k)
        {
            double zk = m_p.z[k];
            double muk = m_p.mu[k];
            dg::blas1::subroutine( [nu_ref, zs, zk, mus, muk] DG_DEVICE(
                double& spperp, double ns, double nk, double ts, double tk)
            {
                spperp += -2*nu_ref*ns*nk*zs*zs*zk*zk/mus/muk/(
                    ts/mus + tk/muk)/sqrt( ts/mus + tk/muk)*(ts-tk);
            }, m_temp0, q.at("ST N")[s], q.at("ST N")[k], q.at("ST Tperp")[s], q.at("ST Tperp")[k]);
        }
    // CN ST
    dg::blas1::pointwiseDivide( m_p.mu[s]/2./m_p.z[s]/m_p.z[s], m_temp0, m_B2, 0., m_temp0);
    dg::blas2::symv( m_lapperpM, m_temp0, m_temp1);
    // CU ST
    dg::blas1::pointwiseDivide( q.at("ST U")[s], q.at("ST N")[s], m_temp0); // U/N
    dg::blas1::pointwiseDot( -1., m_temp0, m_temp1, 0., m_temp0); // -U/N CN ST
    for( unsigned k=0; k<m_p.num_species; k++)
        if( s != k)
        {
            double zs = m_p.z[s], zk = m_p.z[k];
            double mus = m_p.mu[s], muk = m_p.mu[k];
            dg::blas1::subroutine( [nu_ref, zs, zk, mus, muk] DG_DEVICE(
                double& su, double nk, double ts, double tk, double us, double uk)
            {
                su += -0.51*nu_ref*nk*zs*zs*zk*zk*(mus+muk)/mus/mus/muk/(
                    ts/mus + tk/muk)/sqrt( ts/mus + tk/muk)*(us-uk);
            }, m_temp0, q.at("ST N")[k], q.at("ST Tpara")[s], q.at("ST Tpara")[k], q.at("ST U")[s], q.at("ST U")[k]);
        }
    dg::blas1::axpby( 1., m_temp0, 1., yp[3][s]);
}

template<class Grid, class IMatrix, class Matrix, class Container>
void Collisions<Grid, IMatrix, Matrix, Container>::add_lorentz_collisions(
        unsigned s,
        const std::map<std::string, std::vector<Container>>& q,
        const std::array<std::vector<Container>,6>& y,
        std::array<std::vector<Container>,6>& yp)
{
    double nu_ref = m_p.nu_ref;
    double zs = m_p.z[s], mus = m_p.mu[s];
    // spperp Lorentz
    double pis = m_p.pi[s];
    dg::blas1::subroutine( [nu_ref, zs, mus, pis] DG_DEVICE(
        double & spperp, double ns, double pperps, double pparas)
    {
        double ts = pperps/ns;
        double nu_ss = nu_ref*ns*zs*zs*zs*zs/sqrt(2*mus*ts)/ts;
        spperp += nu_ss/pis/3*(pparas-pperps);
    }, yp[1][s], y[0][s], y[1][s], y[2][s]);

    // sppara Lorentz
    dg::blas1::subroutine( [nu_ref, zs, mus,pis] DG_DEVICE(
        double & sppara, double ns, double pperps, double pparas)
    {
        double ts = pparas/ns;
        double nu_ss = nu_ref*ns*zs*zs*zs*zs/sqrt(2*mus*ts)/ts;
        sppara += -2.*nu_ss/pis/3*(pparas-pperps);
    }, yp[2][s], y[0][s], y[1][s], y[2][s]);

    // SQperp Lorentz and Landau
    double kappas = m_p.kappa[s];
    double qlandau = m_p.qlandau;
    double R0 = m_R0;
    dg::blas1::subroutine( [nu_ref, zs, mus,kappas, qlandau, R0] DG_DEVICE(
        double & sqperp, double ns, double ts, double qperp, double qpara)
    {
        double nu_ss = nu_ref*ns*zs*zs*zs*zs/sqrt(2*mus*ts)/ts;
        sqperp += - 5./2./kappas*nu_ss*( qperp - 1.28*(0.5*qpara - 1.5*qperp))
                  - 1./qlandau/R0*sqrt( ts/mus)*qperp;
    }, yp[4][s], q.at("ST N")[s], q.at("ST Tperp")[s], y[4][s], y[5][s]);

    // SQpara Lorentz and Landau
    dg::blas1::subroutine( [nu_ref, zs, mus,kappas, qlandau, R0] DG_DEVICE(
        double & sqpara, double ns, double ts, double qperp, double qpara)
    {
        double nu_ss = nu_ref*ns*zs*zs*zs*zs/sqrt(2*mus*ts)/ts;
        sqpara += - 5./kappas*nu_ss*( 0.5*qpara + 1.28*(0.5*qpara - 1.5*qperp))
                  - 1./qlandau/R0*sqrt( ts/mus)*qpara;
    }, yp[5][s], q.at("ST N")[s], q.at("ST Tpara")[s], y[4][s], y[5][s]);

}

}
