// =========================================================================
//                            TOEFL-2D "T3L"
//                   "TOkamak Edge (gyro)-FLuid"
//               3-species local version (AK, Sep. 2012)
//
// ulimit -s unlimited
// g++ t3l.C -O2 -lfftw3 -lm -fopenmp -lpthread -lfftw3_threads -o t3l
// =========================================================================
#ifdef _OPENMP 
#include <omp.h>
#endif

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <fftw3.h>

const static int nxmax=512+1, nymax=1024+1;  // nx +1 = 2^n + 1  for fft
const static double ZwPi = 2.*M_PI, VPQI = 1./(4.*M_PI*M_PI);
static int nx, ny, n2x, n4x, nxh, nyh, nr, nx1, ny1, nxsep;
static int itstp, itmax, npar, ibx;
static double hy, hy2, hy2inv, hinvsq, yyy, yyymhw, xyi, xy0, dr;
static double ddd, diff, n00, ly, dt, dtt, mhw, incon, mcv, sinth, costh;
static double khyn, khyz, aaz, aai, mui, muz, taui, tauz, nuimp, nd0;
static double soldiss, perpdis, solgn;
static double gge, ggi, ggz;
static double cpoti[nxmax][nymax], cpotz[nxmax][nymax], cvor[nxmax][nymax];
static double cpoti2[2*nxmax][nymax], cpotz2[2*nxmax][nymax], cvor2[2*nxmax][nymax];
static bool snaps, newpuff, imp;

void initpar(void);
void arakawa2(double (*uuu)[nymax], double (*vvv)[nymax], double (*www)[nymax]);
void poisson(double (*pe)[nymax], double (*cp)[nymax], double (*pz)[nymax]);
void poisson2(double (*pe)[nymax], double (*cp)[nymax], double (*pz)[nymax]);
void laplace(double (*ww)[nymax], double (*pp)[nymax]);
void fourier(double (*pp)[nymax], double (*ww)[nymax]);
void diagnose(double ttt, double t00, int it, int itmax,
	      double (*nn)[nymax], double (*pp)[nymax], 
	      double (*ww)[nymax], double (*dd)[nymax], double (*ni)[nymax]);

// --------------------------- Hauptprogramm ---------------------------------

int main(void)
{ 
  int i, j, k, it, is, itn, itend, i0, j0;
  int im,ip,jm,jp,icc;

  // electrostatic potential pe (gyroaveraged pi, pz), vorticity:
  double pe[nxmax][nymax], pi[nxmax][nymax], pz[nxmax][nymax], ww[nxmax][nymax]; 

  // gyrocenter densities
  double ne[nxmax][nymax], ne0[nxmax][nymax], ne1[nxmax][nymax], ne2[nxmax][nymax]; 
  double ni[nxmax][nymax], ni0[nxmax][nymax], ni1[nxmax][nymax], ni2[nxmax][nymax]; 
  double nz[nxmax][nymax], nz0[nxmax][nymax], nz1[nxmax][nymax], nz2[nxmax][nymax]; 

  // gyroscreened densities
  double gnz[nxmax][nymax], gni[nxmax][nymax], gsum[nxmax][nymax];

  // r.h.s. of continuity equations
  double fe0[nxmax][nymax], fe1[nxmax][nymax], fe2[nxmax][nymax];
  double fi0[nxmax][nymax], fi1[nxmax][nymax], fi2[nxmax][nymax];
  double fz0[nxmax][nymax], fz1[nxmax][nymax], fz2[nxmax][nymax];

  // Arakawas and Laplacians
  double ae[nxmax][nymax], vise[nxmax][nymax], hyve[nxmax][nymax];
  double ai[nxmax][nymax], visi[nxmax][nymax], hyvi[nxmax][nymax];
  double az[nxmax][nymax], visz[nxmax][nymax], hyvz[nxmax][nymax];

  // gradient, curvature and dissipative coupling terms
  double dre[nxmax][nymax],dri[nxmax][nymax], drz[nxmax][nymax],  cpl[nxmax][nymax];
  double ce[nxmax][nymax], ci[nxmax][nymax], cz[nxmax][nymax];

  // boundaries:
  double bndys[nxmax];

  double ppzavg, nnzavg, ttt, t00, phase, zuf,rhom;
  double nue, nui, nuz;
  double c0 = 18./11., c1 = 9./11., c2 = 2./11., cf = 6./11.;
  double arbconst, xxx;
  double dx0, bndy;

  fftw_plan hinp, herp;
  char s[80]; 
  char ach[80];
  char const *add, *addc, *inpmdat;
  char const *fn, *fw;
  char str[80]; 

  FILE *f, *g, *h, *g1, *g2, *g3, *g4, *g5;

  initpar();
  t00 =0.;

  fftw_plan_with_nthreads(npar);
  fftw_complex wwk[nx][ny], ppk[nx][ny], wwx[nx][ny], ppx[nx][ny];
  hinp = fftw_plan_dft_2d(nx,ny,&wwx[0][0],&wwk[0][0],FFTW_FORWARD, FFTW_ESTIMATE);
  herp = fftw_plan_dft_2d(nx,ny,&ppk[0][0],&ppx[0][0],FFTW_BACKWARD,FFTW_ESTIMATE);


 // Initialisierung der Felder ------------------------------------------

  if (incon==0.)  // ion density blob  
    {
      printf("| ion blob initial condition...\n");
      dr = double(nr)/16.;
      dr *= dr;
     for (i=0; i<=nx1; ++i)
	for (j=0; j<=ny1; ++j)
	  {
	    ne[i][j] = n00*exp(-(i-nxh)*(i-nxh)/dr) *exp(-(j-nyh)*(j-nyh)/dr);
	    ni[i][j] = ne[i][j]; 
	    ww[i][j] = 0.; // = ne[i][j] - ni[i][j]; 
	  }
      poisson(ww,cvor,pe);
   }


   if (incon==1.)  // turbulent bath
    {
      printf("| bath initial condition...\n");
      double ppk;
      int nk = int(ly)/2;
      double phs[nk+1][nk+1];
      // srand(time(NULL) + getpid());

      nk = 32;

#ifdef _OPENMP
#pragma omp parallel private(zuf) shared(nk,phs)
      {
#pragma omp for
#endif
      for (int ik=1; ik<=nk; ik++) 
	for (int jk=1; jk<=nk; jk++)
	  {
	    zuf = 0.; // 0.01*(rand() % 100);
	    phs[ik][jk] = cos(ZwPi*zuf)
	      *n00/sqrt(1.+pow((3.125*(ik*ik+jk*jk)/(nk*nk)),4));
	    phs[ik][jk] = n00/sqrt(1.+pow((3.125*(ik*ik+jk*jk)/(nk*nk)),4));

	  }
#ifdef _OPENMP
      }
#pragma omp barrier
#endif

      int mmm = 1;
      for (int my=0; my<mmm; my++)
	{
#ifdef _OPENMP
#pragma omp parallel private(i,j) shared(my,mmm,nx1,ny1,nk,nx,ny,ni,phs,pe)
      {
#pragma omp for
#endif
     for (i=0; i<=nx1; i++)
       // for (j=0; j<=ny1; j++)
       for (j=(my*ny1/mmm); j<=((1+my)*(ny1/mmm)); j++)
	  {
	    // printf("%d  %d\n",i,j);
	    zuf = 0.01*(rand() % 100);
	    ni[i][j] = 0.;
	    for (int ik=1; ik<=nk; ik++) 
	      for (int jk=1; jk<=nk; jk++)
		{
		  zuf = 0.01*(rand() % 100);
		  ne[i][j] += phs[ik][jk]*cos(ZwPi*zuf)
		    *cos(ZwPi*ik*(i/double(nx)))*cos(ZwPi*jk*(mmm*j/double(ny)));
		}
	  }

     for (i=0; i<=nx1+1; ++i)
       for (j=0; j<=ny1+1; ++j)
	 {
	   nz[i][j] = nd0;
	   ni[i][j] = (ne[i][j]-aaz*nz[i][j])/(aai+1.e-9); 
	   ww[i][j] = 0.; // ne[i][j]-ni[i][j];
	 }
     
#ifdef _OPENMP
      }
#pragma omp barrier
#endif
	}
      poisson(ww,cvor,pe);
    }
   
   // set artificial previous time values
   for (i=0; i<=nx1; ++i)
     for (j=0; j<=ny1; ++j)
       {
	 ne0[i][j] = ne[i][j]; ne1[i][j] = ne[i][j]; ne2[i][j] = ne[i][j];
	 ni0[i][j] = ni[i][j]; ni1[i][j] = ni[i][j]; ni2[i][j] = ni[i][j];
	 nz0[i][j] = nz[i][j]; nz1[i][j] = nz[i][j]; nz2[i][j] = nz[i][j];
	 fe1[i][j] = 0.; fe2[i][j] = 0.; 
	 fi1[i][j] = 0.; fi2[i][j] = 0.;
	 fz1[i][j] = 0.; fz2[i][j] = 0.;
	 ce[i][j] = 0.; ci[i][j] = 0.; cz[i][j] = 0.; cpl[i][j] = 0.;
       }
   

   if (incon==2.) // restart from saved
     {
      printf("| continued data set... \n");
      g = fopen( "restart.dat", "r" );
      fscanf(g,"%s ",str);  t00  = atof(str); 
      while (!feof(g)) 
      { 
	  fscanf(g,"%s ",str);  i = atoi(str); 
	  fscanf(g,"%s ",str);  j = atoi(str); 
	  fscanf(g,"%s ",str);  pe[i][j]  = atof(str); 
	  fscanf(g,"%s ",str);  ne[i][j]  = atof(str); 
	  fscanf(g,"%s ",str);  ne1[i][j] = atof(str); 
	  fscanf(g,"%s ",str);  ne2[i][j] = atof(str); 
	  fscanf(g,"%s ",str);  ni[i][j]  = atof(str); 
	  fscanf(g,"%s ",str);  ni1[i][j] = atof(str); 
	  fscanf(g,"%s ",str);  ni2[i][j] = atof(str); 
	  fscanf(g,"%s ",str);  nz[i][j]  = atof(str); 
	  fscanf(g,"%s ",str);  nz1[i][j] = atof(str); 
	  fscanf(g,"%s ",str);  nz2[i][j] = atof(str); 
	  fscanf(g,"%s ",str);  fe1[i][j] = atof(str); 
	  fscanf(g,"%s ",str);  fe2[i][j] = atof(str); 
	  fscanf(g,"%s ",str);  fi1[i][j] = atof(str); 
	  fscanf(g,"%s ",str);  fi2[i][j] = atof(str); 
	  fscanf(g,"%s ",str);  fz1[i][j] = atof(str); 
	  fscanf(g,"%s ",str);  fz2[i][j] = atof(str); 
      }
      for (i=0; i<=nr; i++) for (j=0; j<=ny; j++) ne0[i][j] = ne[i][j];
      for (i=0; i<=nr; i++) for (j=0; j<=ny; j++) ni0[i][j] = ni[i][j];
      for (i=0; i<=nr; i++) for (j=0; j<=ny; j++) nz0[i][j] = nz[i][j];
      fclose( g ); 
    }


  // new impurity puff
  if (newpuff)
    {
      for (i=0; i<=nx1; ++i)
	for (j=0; j<=ny1; ++j)
	  { 
	    nz[i][j] = nd0;
	    // nz[i][j] = 1.+nd0*exp(-double((j-nyh)*(j-nyh))/xi); 
	    nz0[i][j] = nz[i][j]; nz1[i][j] = nz[i][j]; nz2[i][j] = nz[i][j];
	    fz1[i][j] = 0.; fz2[i][j] = 0.;
	  }
    }

  // poloidally variable curvature
  double sss[nymax], ccc[nymax], asym, theta;
  for (j=0; j<=ny1; ++j)
    {
      theta = ZwPi*double(j-nyh)/double(ny);
      ccc[j] = -mcv*.5*hy*cos(theta);
      sss[j] = -mcv*.5*hy*sin(theta);
    }
  for (i=0; i<=nx1; ++i) 
    for (j=0; j<=ny1; ++j)
      {	ce[i][j]=0.; ci[i][j]=0.; cz[i][j]=0.; }

  // x Dirichlet boundary
  dx0 = double(ibx*ibx);
  for (i=0; i<=nx1; i++) 
    bndys[i] = 1.-exp(-double(i*i)/dx0)-exp(-double((nx1-i)*(nx1-i))/dx0);

  // inertial scaling
  rhom = aai*mui+aaz*muz;


  // Initialisierung Zeitschritt ----------------------------------------

  itn = 0;
  itend = itmax / itstp;
  dtt = dt*cf;
  double ddtt = double(itstp)*dt;

  // Zeitschritt --------------------------------------------------------
  // 3rd order Karniadakis time scheme / Arakawa brackets

  for (it=itn/itstp; it<itend; ++it) // total time loop
    {
      for (is=0; is<itstp; ++is)     // inner time loop
	{
	  ttt = t00 + (it+1)*ddtt;

	  // gyro-screened potentials	
	  poisson(pe,cpoti,pi);
	  if (imp) poisson(pe,cpotz,pz);

	  for (i=0; i<=nx1; ++i) 
	    {
	      // dissipative coupling term for electrons (Hasegawa-Wakatani)
	      if (ddd>0.)
		{
		  ppzavg=0.; nnzavg=0.;
		  if (mhw==1.) // MHW: subtract zonal average
		    {
		      for (j=0; j<=ny1; ++j) 
			{ ppzavg += pe[i][j]; nnzavg += ne[i][j]; }
		      ppzavg*= yyymhw; nnzavg*= yyymhw;
		    }
		  for (j=0; j<=ny1; ++j) 
		    {
		      cpl[i][j] = ddd*((pe[i][j]-ppzavg) - (ne[i][j]-nnzavg));
		      if (i>nxsep) cpl[i][j] = 0.;
		    }
		}

	      // gradient (drift wave) drive term
	      for (j=0; j<=ny1; ++j)
		{
		  jm = (j==0)   ? ny1 : j-1;
		  jp = (j==ny1) ? 0   : j+1;
		  dre[i][j] = khyn*(pe[i][jp]-pe[i][jm]);
		  dri[i][j] = khyn*ggi*(pi[i][jp]-pi[i][jm]);
		  if (i>nxsep) {dre[i][j]*=solgn; dri[i][j]*=solgn; }
		  if (imp) drz[i][j] = khyz*(pz[i][jp]-pz[i][jm]);
		}
	    }

	  // curvature terms
	  if (mcv > 0.)
	    {
	      for (i=0; i<=nx1; ++i) 
		{
		  im = (i==0)   ? nx1 : i-1;
		  ip = (i==nx1) ? 0   : i+1;
		  for (j=0; j<=ny1; ++j)
		    {
		      jm = (j==0)   ? ny1 : j-1;
		      jp = (j==ny1) ? 0   : j+1;

		      // poloidal version:
		      // sinth = sss[j];
		      // costh = ccc[j];

		      // local version:
		      sinth = sss[nyh];
		      costh = ccc[nyh];

		      ce[i][j] = + sinth*(pe[ip][j]-pe[im][j]);
		      ce[i][j]+= + costh*(pe[i][jp]-pe[i][jm]);
		      ce[i][j]+= - sinth*(ne[ip][j]-ne[im][j]);
		      ce[i][j]+= - costh*(ne[i][jp]-ne[i][jm]);

		      ci[i][j] = + sinth*(pi[ip][j]-pi[im][j]);
		      ci[i][j]+= + costh*(pi[i][jp]-pi[i][jm]);
		      ci[i][j]+= + sinth*taui*(ni[ip][j]-ni[im][j]);
		      ci[i][j]+= + costh*taui*(ni[i][jp]-ni[i][jm]);

		      if (imp) 
			{
			  cz[i][j] = + nz[i][j]*sinth*(pz[ip][j]-pz[im][j]);
			  cz[i][j]+= + nz[i][j]*costh*(pz[i][jp]-pz[i][jm]);
			  cz[i][j]+= + sinth*tauz*(nz[ip][j]-nz[im][j]);
			  cz[i][j]+= + costh*tauz*(nz[i][jp]-nz[i][jm]);
			}
		    }
		}
	    }

	  arakawa2(ne,pe,ae); 
	  arakawa2(ni,pi,ai);
	  if (imp) arakawa2(nz,pz,az);

	  laplace(vise,ne);
	  laplace(visi,ni);
	  if (imp) laplace(visz,nz);
	  laplace(hyve,vise);
	  laplace(hyvi,visi);
	  if (imp) laplace(hyvz,visz);

	  for (i=0; i<=nx1; i++)
	    for (j=0; j<=ny1; j++)
	      {
		// r.h.s.
		fe0[i][j] =  - ae[i][j] - dre[i][j] + ce[i][j] + cpl[i][j];
		fi0[i][j] =  - ai[i][j] - dri[i][j] + ci[i][j];
		if (imp) fz0[i][j] =  - az[i][j] - drz[i][j] + cz[i][j];

		// dissipation and hyper-viscosity
		nue = diff*(   0.0*vise[i][j] - hyve[i][j] );
		nui = diff*(   0.0*visi[i][j] - hyvi[i][j] );
		if (imp) nuz = diff*( nuimp*visz[i][j] - hyvz[i][j] );

		if (i>nxsep) { nue-=soldiss*ne[i][j]; nui-=soldiss*ni[i][j];}
		if (i>nxsep) { nue+=perpdis*vise[i][j]; nui+=perpdis*visi[i][j];}

		// time step update

		ne[i][j] = c0*ne0[i][j] - c1*ne1[i][j] + c2*ne2[i][j] 
		  + dtt*(3.*(fe0[i][j] - fe1[i][j]) + fe2[i][j] + nue);

		ni[i][j] = c0*ni0[i][j] - c1*ni1[i][j] + c2*ni2[i][j] 
		  + dtt*(3.*(fi0[i][j] - fi1[i][j]) + fi2[i][j] + nui);
		if (imp) 
		  nz[i][j] = c0*nz0[i][j] - c1*nz1[i][j] + c2*nz2[i][j] 
		    + dtt*(3.*(fz0[i][j] - fz1[i][j]) + fz2[i][j] + nuz);
	      }

	  if (ibx == 0) // periodic b.c.: normal FFT Poisson solver
	    {
	      // gyro-averaged density
	      poisson(ni,cpoti,gni);
	      if (imp) poisson(nz,cpotz,gnz);

	      // gyrofluid vorticity
	      for (i=0; i<=nx1; i++) 
		for (j=0; j<=ny1; j++)
		  {
		    rhom = aai*mui + aaz*muz; // local/linear inertial
		    ww[i][j] = rhom*(ne[i][j] - aai*gni[i][j] - aaz*gnz[i][j]);
		  }
	      
	      // local/linear gyrofluid polarisation equation:
	      poisson(ww,cvor,pe);
	    }

	  else // Dirichlet b.c.: half-wave FFT Poisson solver
	    {
	      // gyro-averaged density
	      poisson2(ni,cpoti2,gni);
	      if (imp) poisson2(nz,cpotz2,gnz);

	      // gyrofluid vorticity
	      for (i=0; i<=nx1; i++) 
		for (j=0; j<=ny1; j++)
		  {
		    rhom = aai*muz + aaz*muz; // local/linear inertia
		    ww[i][j] = rhom*(ne[i][j] - aai*gni[i][j] - aaz*gnz[i][j]);
		  }
	      
	      // local/linear gyrofluid polarisation equation:
	      poisson2(ww,cvor2,pe);

	      for (i=0; i<=nx1; i++) 
		for (j=0; j<=ny1; ++j)
		  { 
		    pe[i][j] *= bndys[i]; pi[i][j] *= bndys[i]; pz[i][j] *= bndys[i]; 
		    ne[i][j] *= bndys[i]; ni[i][j] *= bndys[i]; nz[i][j] *= bndys[i]; 
		  }
	    }


	  // remember values two steps backwards
	  for (i=0; i<=nx1; i++)
	    for (j=0; j<=ny1; j++)
	      {
		ne2[i][j] = ne1[i][j]; ne1[i][j] = ne0[i][j]; ne0[i][j] = ne[i][j];
		ni2[i][j] = ni1[i][j]; ni1[i][j] = ni0[i][j]; ni0[i][j] = ni[i][j];
		if (imp) nz2[i][j] = nz1[i][j]; nz1[i][j] = nz0[i][j]; nz0[i][j] = nz[i][j];
		fe2[i][j] = fe1[i][j]; fe1[i][j] = fe0[i][j];
		fi2[i][j] = fi1[i][j]; fi1[i][j] = fi0[i][j];
		if (imp) fz2[i][j] = fz1[i][j]; fz1[i][j] = fz0[i][j];
	      }
	}
      // low-res diagnostics: energies and snapshots
      diagnose(ttt,t00,it,itmax,ne,pe,ww,nz,ni);
    }

  fftw_destroy_plan(hinp);
  fftw_destroy_plan(herp);

  // Ende Zeitschritt ------------------------------------------------

  g = fopen( "restart.dat", "w" );
  fprintf(g,"%5e  \n", ttt);
  for (i=0; i<=nx1; i++)
    for (j=0; j<=ny1; j++)
      fprintf(g,"%d  %d  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e\n",
	      i,j,pe[i][j],ne[i][j],ne1[i][j],ne2[i][j],ni[i][j],ni1[i][j],
	      ni2[i][j],nz[i][j],nz1[i][j],nz2[i][j],fe1[i][j],fe2[i][j],
	      fi1[i][j],fi2[i][j],fz1[i][j],fz2[i][j]);
  fclose( g );

  g = fopen( "cutx.dat", "w" );
  for (i=0; i<=nx1; i++)
    for (j=nyh; j<=nyh; j++)
      fprintf(g,"%d  %.6e   %.6e   %.6e\n",i,ne[i][j],pe[i][j],ww[i][j]);
  fclose( g );

  g = fopen( "cuty.dat", "w" );
  for (i=nxh; i<=nxh; i++)
    for (j=0; j<=ny1; j++)
      fprintf(g,"%d  %.6e   %.6e   %.6e\n", j,ne[i][j],pe[i][j],ww[i][j]);
  fclose( g );

  // Fourier ky spectra
  double py[ny];
  fftw_complex ky[ny/2], sumky[ny/2];
#ifdef _OPENMP 
  fftw_plan_with_nthreads(npar);
#endif
  fftw_plan hindfty;

  hindfty = fftw_plan_dft_r2c_1d(ny, &py[0], &ky[0], FFTW_ESTIMATE);

  for (j=0; j<=ny/2; j++) sumky[j][0] = 0.;

  for (i=0; i<=nx1; i++)
    {
      for (j=0; j<=ny1; j++) py[j] = pe[i][j];
      fftw_execute(hindfty);
      for (j=0; j<=ny/2; j++) sumky[j][0] += ky[j][0]*ky[j][0]; 
    }
  for (j=0; j<=ny/2; j++) sumky[j][0] /= (nx*ny); 

  fftw_destroy_plan(hindfty);

  g = fopen( "pky.dat", "w" );
  for (j=1; j<=ny/2; j++) 
    fprintf(g,"%.6e  %.6e\n",hy*double(j)/ly,sumky[j][0]);
  fclose( g );

  // Fourier kx spectra
  double px[nx];
  fftw_complex kx[nx/2], sumkx[nx/2];
  fftw_plan hindftx;

  hindftx = fftw_plan_dft_r2c_1d(nx, &px[0], &kx[0], FFTW_ESTIMATE);

  for (i=0; i<=nx/2; i++) sumkx[i][0] = 0.;
  for (j=0; j<=ny1; j++)
    {
      for (i=0; i<=nx1; i++) px[i] = pe[i][j];
      fftw_execute(hindftx);
      for (i=0; i<=nx/2; i++) sumkx[i][0] += kx[i][0]*kx[i][0]; 
    }
  for (i=0; i<=nx/2; i++) sumkx[i][0] /= (nx*ny); 
  
  fftw_destroy_plan(hindftx);

  g = fopen( "pkx.dat", "w" );
  for (i=1; i<=nx/2; i++) 
    fprintf(g,"%.6e  %.6e\n",hy*double(i)/ly,sumkx[i][0]);
  fclose( g );

#ifdef _OPENMP 
  fftw_cleanup_threads();
#endif

  printf("|TOEFL beendet.\n\n");
}
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Einlesen der Inputdaten
void initpar(void)
{ 
  int i,j; char s[80]; FILE *g, *f;
  double para[30];

  for (int i=0;i<=40;i++) printf("_");
  printf("\n| TOEFL Start...\n");

  g = fopen( "thw.inp", "r" );
  i = 0;
  while (!feof(g)) { fscanf(g, "%s ", s); if (strstr(s,"=")) i++; }; 
  rewind(g);
  for (j=1; j<=i; j++)
    { 
      while (!strstr(s,"=")) fscanf(g,"%s ",s); 
      fscanf(g,"%s ",s); para[j]=atof(s); 
    }; 
  fclose( g );

  ly = para[12];
  nx = int(para[13]);
  ny = int(para[14]);
  hy = double(ny)/ly; 

  ddd   = para[1];
  gge = para[2];
  ggz = para[3];

  khyn  = -.5*hy*gge;
  khyz  = -.5*hy*ggz;

  taui  = para[4];
  tauz  = para[22];
  muz  = para[23];
  aaz  = para[24];
  mcv   = para[6];
  mhw   = para[7];
  diff  = para[8];
  nuimp = para[9];
  n00   = para[10];
  nd0   = para[11];
  imp = (para[11]>0.) ? true : false;
  dt    = para[15];
  itstp = int(para[16]);
  itmax = int(para[17]);
  npar  = int(para[18]);
  incon = para[19];
  snaps = (para[20]==1.) ? true : false;
  ibx   = int(para[21]);
  newpuff = (para[5]==1.) ? true : false;
  nxsep = int((para[25]*para[13]));
  soldiss = para[26];
  perpdis   = para[27];
  solgn = para[28];

  mui = 1.;
  aai = 1.-aaz;
  ggi = (1.-aaz*ggz)/(aai+1.e-9);

  // Initialisierung von Open_MP ...................
#ifdef _OPENMP 
  fftw_init_threads();
  omp_set_num_threads(npar);
  int np, kp;
#endif

#ifdef _OPENMP
  printf("| parallel threads ");
#pragma omp parallel private(kp)
  { 
      np = omp_get_num_threads();
      kp = omp_get_thread_num();
      printf("%d/",kp);
  } 
  printf(" of %d active.\n",np);
  double dnp=double(np);
#else
  printf("| single core processing\n");
  double dnp=1.;
#pragma omp barrier
#endif

  // Initialiserung der Parameter ..................
  nxh   = nx/2;
  n2x   = 2*nx;
  n4x   = 4*nx;
  nyh   = ny/2; 

  nr  = nx-1;
  nx1 = nx -1;
  ny1 = ny -1;

  hy2 = hy*hy;
  hy2inv = 1./hy2;
  hinvsq = - hy2/12.;
 
  xyi = 1./double(nx*ny);
  xy0 = 1./double(nx*ny);
  yyy = 1./double(ny+1);
  yyymhw = yyy*mhw;

  if (incon!=2.)
    {
      f = fopen("imp.dat","w");  fclose(f);
      f = fopen("time.dat","w");  fclose(f);
      f = fopen("zfx.dat", "w"); fclose(f);
    }

  // coeff pre-calculation for FFTW Poisson solver

  int ik, jk;
  double kx, ky, dxy = 1./(hy*hy);
  double axy = double(ny)/double(nx);
  double cnx = double(nx);
  double cny = double(ny);
  double cxy = 1./(cnx*cny);
  double kkqq;

  for (int i=0; i<=nx1; ++i) 
    for (int j=0; j<=ny1; ++j) 
       {
	 ik = (i>=nx/2) ? i - nx : i;
	 jk = (j>=ny/2) ? j - ny : j;

	 kkqq = (ZwPi*hy*ik/cnx)*(ZwPi*hy*ik/cnx) + (ZwPi*hy*jk/cny)*(ZwPi*hy*jk/cny);

	 cpoti[i][j] =  cxy/(1.+.5*taui*aai*mui*kkqq);
	 cpotz[i][j] =  cxy/(1.+.5*tauz*aaz*muz*kkqq);

	 cvor[i][j] = - cxy*(1.+taui*aai*mui*kkqq)/kkqq;
       }
  cvor[0][0] = 0.;

  for (int i=0; i<=n2x-1; ++i) 
    for (int j=0; j<=ny1; ++j) 
       {
	 ik = (i>=n2x/2) ? i - n2x: i;
	 jk = (j>=nyh)   ? j - ny : j;

	 kkqq = (.5*ZwPi*hy*ik/cnx)*(.5*ZwPi*hy*ik/cnx) 
	   + (ZwPi*hy*jk/cny)*(ZwPi*hy*jk/cny);

	 cpoti2[i][j] =  .5*cxy/(1.+.5*taui*aai*mui*kkqq);
	 cpotz2[i][j] =  .5*cxy/(1.+.5*tauz*aaz*muz*kkqq);

	 cvor2[i][j] = - .5*cxy*(1.+taui*aai*mui*kkqq)/kkqq;
       }
  cvor2[0][0] = 0.;

}

// ---------------------------------------------------------------------------
void diagnose(double ttt, double t00, int it, int itmax,
	      double (*ne)[nymax], double (*pp)[nymax], double (*ww)[nymax], double (*dd)[nymax], double (*ni)[nymax])
{
  int i,j, im,ip,jm,jp;
  double enn, enp, enw, eeb, fne, zfx;
  FILE *f, *g, *h, *g1, *g2, *g3, *g4, *g5, *g6;

  // energetics
  enn = 0.; enp = 0.; enw = 0.; eeb = 0.; fne = 0.;
  for (i=0; i<=nx1; i++)
    {
      for (j=0; j<=ny1; j++)
	{
	  jm = (j==0)   ? ny1 : j-1;
	  jp = (j==ny1) ? 0   : j+1;
	  // total thermal free energy:
	  enn += ne[i][j]*ne[i][j]+aai*taui*ni[i][j]*ni[i][j]
		     +aaz*tauz*dd[i][j]*dd[i][j];
	  // (mostly zonal) flow energy:
	  enp += pp[i][j]*pp[i][j];
	  // turbulent energy:
	  enw += ww[i][j]*ww[i][j];
	  // ExB energy:
	  eeb += pp[i][j]*(ne[i][j]+aai*ni[i][j]+aaz*dd[i][j]);
	  // electron particle transport:
	  fne += ne[i][j]*.5*hy*(pp[i][jp]-pp[i][jm]);
	}
    }
  enn *= .5*xyi; enp *= .5*xyi; enw *= .5*xyi; eeb *= .5*xyi; fne *= .5*xyi;

  // sample correlation coefficient:
  /*
  double neavg = 0., niavg = 0., ndavg = 0., ppavg = 0., wwavg = 0., dfavg = 0.;
  for (i=1; i<=nx1; i++) 
    for (j=1; j<=ny1; j++) 
      {
	neavg += ne[i][j];
	niavg += ni[i][j];
	ndavg += dd[i][j];
	ppavg += pp[i][j];
	wwavg += ww[i][j];
	dfavg += dd[i][j] + muz*ww[i][j];
      }
  neavg/=(nx1*ny1); niavg/=(nx1*ny1); ndavg/=(nx1*ny1); 
  ppavg/=(nx1*ny1); wwavg/=(nx1*ny1); dfavg/=(nx1*ny1); 
  double samp_pp_nd = 0., samp_ww_nd = 0.;
  double samp_ne_nd = 0., samp_ni_nd = 0.;
  double samp_ww_ne = 0., samp_ww_ni = 0.;

  double samp_ne_df = 0., samp_ww_df = 0.;

  double msq_pp =0., msq_nd = 0., msq_ww = 0.;
  double msq_ne =0., msq_ni = 0., msq_df = 0.; 
  double ddel, wdel;
  double dflu;
  */

  double cor1;

  // xy plots
  g1 = fopen( "n2d0.dat", "w" ); g2 = fopen( "w2d0.dat", "w" ); 
  g3 = fopen( "p2d0.dat", "w" ); g4 = fopen( "d2d0.dat", "w" );
  g5 = fopen( "i2d0.dat", "w" ); g6 = fopen( "r2d0.dat", "w" );
  for (i=0; i<=nx1; i++) 
    {
      for (j=0; j<=ny1; j++) 
	{
	  /*
	  ddel = dd[i][j]-ndavg;
	  wdel = ww[i][j]-wwavg;

	  dflu = dd[i][j] + muz*ww[i][j] - dfavg;

	  samp_pp_nd += (pp[i][j]-ppavg)*ddel;
	  samp_ww_nd += wdel*ddel;
	  samp_ww_ne += wdel*(ne[i][j]-neavg);
	  samp_ww_ni += wdel*(ni[i][j]-niavg);
	  samp_ne_nd += (ne[i][j]-neavg)*ddel;
	  samp_ni_nd += (ni[i][j]-niavg)*ddel;

	  samp_ne_df += (ne[i][j]-neavg)*dflu;
	  samp_ww_df += wdel*dflu;

	  msq_pp += (pp[i][j]-ppavg)*(pp[i][j]-ppavg);
	  msq_nd += ddel*ddel;
	  msq_ww += wdel*wdel;
	  msq_ne += (ne[i][j]-neavg)*(ne[i][j]-neavg);
	  msq_ni += (ni[i][j]-niavg)*(ni[i][j]-niavg);

	  msq_df += dflu*dflu;
	  */

	  cor1 = dd[i][j] + muz*ww[i][j];

	  fprintf(g1,"%d  %d  %.6e\n",i,j,ne[i][j]); 
	  fprintf(g2,"%d  %d  %.6e\n",i,j,ww[i][j]); 
	  fprintf(g3,"%d  %d  %.6e\n",i,j,pp[i][j]); 
	  fprintf(g4,"%d  %d  %.6e\n",i,j,dd[i][j]);
	  fprintf(g5,"%d  %d  %.6e\n",i,j,ni[i][j]); 
	  fprintf(g6,"%d  %d  %.6e\n",i,j,cor1); 
	}
      fprintf(g1,"\n"); fprintf(g2,"\n"); fprintf(g3,"\n"); 
      fprintf(g4,"\n"); fprintf(g5,"\n"); fprintf(g6,"\n"); 
    }
  fclose( g1 ); fclose( g2 ); fclose( g3 ); fclose( g4 ); fclose( g5 );
  rename("n2d0.dat", "n2d.dat"); rename("w2d0.dat", "w2d.dat"); 
  rename("p2d0.dat", "p2d.dat"); rename("d2d0.dat", "d2d.dat"); 
  rename("i2d0.dat", "i2d.dat"); rename("r2d0.dat", "r2d.dat"); 

  /*
  double epsi = 1.e-9;

  samp_pp_nd /= sqrt(msq_pp*msq_nd + epsi);
  samp_ww_nd /= sqrt(msq_ww*msq_nd + epsi);
  samp_ww_ne /= sqrt(msq_ww*msq_ne + epsi);
  samp_ww_ni /= sqrt(msq_ww*msq_ni + epsi);
  samp_ne_nd /= sqrt(msq_ne*msq_nd + epsi);
  samp_ni_nd /= sqrt(msq_ni*msq_nd + epsi);

  samp_ne_df /= sqrt(msq_ne*msq_df + epsi);
  samp_ww_df /= sqrt(msq_ww*msq_df + epsi);

  f = fopen("corr.dat","a");
  fprintf(f,"%.3f  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e\n", ttt, 
	  samp_ww_nd, samp_ww_ne, samp_ww_ni, samp_ne_nd, 
	  samp_ni_nd, samp_pp_nd, samp_ne_df, samp_ww_df);
  fclose(f);
  */

  // zonal xt plot
  h = fopen( "zfx.dat", "a" );
  for (i=0; i<=nx1; i++)
    {
      for (zfx=0., j=0; j<=ny1; j++) zfx += pp[i][j]/ny;
      fprintf(h,"%.2f  %d  %.5e \n", ttt,i, zfx);
    } 
  fprintf(h,"\n");
  fclose(h);

  // time series
  ttt = t00 + double((it+1)*itstp)*dt;
  f = fopen("imp.dat","a");
  fprintf(f,"%.3f  %.5e  %.5e  %.5e  %.5e  %.5e\n", 
	  ttt, enn, enp, enw, eeb, fne);
  fclose(f);

  printf("|Zeit: %.2f von %.2f:  enn = %.3e\n", ttt,(t00+itmax*dt),enn*xyi);
}

// ---------------------------------------------------------------------------
// Arakawa scheme for brackets - compact version (slightly faster):
void arakawa2(double (*uuu)[nymax], double (*vvv)[nymax], double (*www)[nymax])
{ 
  int i0,j0,ip,jp,im,jm;
  double xxx;

  for (i0=0; i0<=nx1; ++i0)
    {
      ip = (i0==nx1) ? 0   : i0+1;  
      im = (i0==0)   ? nx1 : i0-1;

      for (j0=0; j0<=ny1; ++j0)
	{
	  jp = (j0==ny1) ? 0   : j0+1; 
	  jm = (j0==0)   ? ny1 : j0-1;
	  
	  xxx = vvv[i0][jm]*( uuu[ip][j0] -uuu[im][j0] -uuu[im][jm] +uuu[ip][jm] );
	  xxx+= vvv[i0][jp]*(-uuu[ip][j0] +uuu[im][j0] -uuu[ip][jp] +uuu[im][jp] );
	  xxx+= vvv[ip][j0]*( uuu[i0][jp] -uuu[i0][jm] +uuu[ip][jp] -uuu[ip][jm] );
	  xxx+= vvv[im][j0]*(-uuu[i0][jp] +uuu[i0][jm] +uuu[im][jm] -uuu[im][jp] );
	  xxx+= vvv[ip][jm]*( uuu[ip][j0] -uuu[i0][jm]);
	  xxx+= vvv[ip][jp]*( uuu[i0][jp] -uuu[ip][j0]);
	  xxx+= vvv[im][jm]*( uuu[i0][jm] -uuu[im][j0]);
	  xxx+= vvv[im][jp]*( uuu[im][j0] -uuu[i0][jp]);

	  www[i0][j0] = -hinvsq*xxx;
	};
    }
}

// ---------------------------------------------------------------------------
// 2D Laplace operator (5-point stencil)
void laplace(double (*fo)[nymax], double (*fi)[nymax])
{ 
  int i,j,im,ip,jm,jp; 
  for (i=0; i<=nx1; i++) 
    {
      im = (i==0)   ? nx1 : i-1;
      ip = (i==nx1) ? 0   : i+1;
      for (j=0; j<=ny1; j++)
	{
	  jm = (j==0)   ? ny1 : j-1;
	  jp = (j==ny1) ? 0   : j+1;

	  fo[i][j] = hy2*(fi[ip][j]+fi[im][j]+fi[i][jp]+fi[i][jm]-4.*fi[i][j]);
	}
    }
}

// ---------------------------------------------------------------------------
// FFT Poisson solver
void poisson(double (*pe)[nymax], double (*cp)[nymax], double (*pz)[nymax])
{
  int i,j; 
  fftw_complex wwk[nx][ny], ppk[nx][ny], wwx[nx][ny], ppx[nx][ny];
  fftw_plan hinpsub, herpsub;

  hinpsub=fftw_plan_dft_2d(nx,ny,&wwx[0][0],&wwk[0][0],FFTW_FORWARD, FFTW_ESTIMATE);
  herpsub=fftw_plan_dft_2d(nx,ny,&ppk[0][0],&ppx[0][0],FFTW_BACKWARD,FFTW_ESTIMATE);

  for (i=0; i<=nx1; ++i) for (j=0; j<=ny1; ++j) wwx[i][j][0] = pe[i][j];
  fftw_execute(hinpsub);

  for (i=0; i<=nx1; ++i) 
    for (j=0; j<=ny1; ++j)  
      {
	ppk[i][j][0] = wwk[i][j][0]*cp[i][j];
	ppk[i][j][1] = wwk[i][j][1]*cp[i][j]; 
      }

  fftw_execute(herpsub);
  for (i=0; i<=nx1; ++i) for (j=0; j<=ny1; ++j) pz[i][j] = ppx[i][j][0]; 
   
  fftw_destroy_plan(hinpsub);
  fftw_destroy_plan(herpsub);
}

// ---------------------------------------------------------------------------
// FFT half-wave Poisson solver for Dirichlet b.c.
void poisson2(double (*pe)[nymax], double (*cp)[nymax], double (*pz)[nymax])
{
  int i,j; 
  fftw_complex wwk[n2x][ny], ppk[n2x][ny], wwx[n2x][ny], ppx[n2x][ny];
  fftw_plan hinpsub, herpsub;

  hinpsub=fftw_plan_dft_2d(n2x,ny,&wwx[0][0],&wwk[0][0],FFTW_FORWARD, FFTW_ESTIMATE);
  herpsub=fftw_plan_dft_2d(n2x,ny,&ppk[0][0],&ppx[0][0],FFTW_BACKWARD,FFTW_ESTIMATE);

  for (i=0; i<=nx1; ++i) 
    for (j=0; j<=ny1; ++j) 
      {
	wwx[i][j][0] = pe[i][j];
	wwx[n2x-1-i][j][0] = - pe[i][j];
      }
  fftw_execute(hinpsub);

  for (i=0; i<=n2x-1; ++i) 
    for (j=0; j<=ny1; ++j)  
      {
	ppk[i][j][0] = wwk[i][j][0]*cp[i][j];
	ppk[i][j][1] = wwk[i][j][1]*cp[i][j]; 
      }

  fftw_execute(herpsub);
  for (i=0; i<=nx1; ++i) for (j=0; j<=ny1; ++j) pz[i][j] = ppx[i][j][0]; 

  fftw_destroy_plan(hinpsub);
  fftw_destroy_plan(herpsub);
}


// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
