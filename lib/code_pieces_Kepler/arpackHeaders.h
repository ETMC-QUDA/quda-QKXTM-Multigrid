#include <complex>

#define _AFT(s) s ## _

extern "C"{

//ARPACK initlog and finilog routines for printing the ARPACK log (same for serial and parallel version)
extern int _AFT(initlog) (int*, char*, int);
extern int _AFT(finilog) (int*);


//ARPACK driver routines for computing eigenvectors (serial version) 
extern int _AFT(znaupd) (int *ido, char *bmat, int *n, char *which, int *nev, double *tol,
                         std::complex<double> *resid, int *ncv, std::complex<double> *v, int *ldv, 
                         int *iparam, int *ipntr, std::complex<double> *workd, std::complex<double> *workl, 
                         int *lworkl, double *rwork, int *info, int bmat_size, int which_size );
			
extern int _AFT(zneupd) (int *comp_evecs, char *howmany, int *select, std::complex<double> *evals, 
			 std::complex<double> *v, int *ldv, std::complex<double> *sigma, std::complex<double> *workev, 
			 char *bmat, int *n, char *which, int *nev, double *tol, std::complex<double> *resid, 
                         int *ncv, std::complex<double> *v1, int *ldv1, int *iparam, int *ipntr, 
                         std::complex<double> *workd, std::complex<double> *workl, int *lworkl, double *rwork, int *info,
                         int howmany_size, int bmat_size, int which_size);

extern int _AFT(mcinitdebug)(int*,int*,int*,int*,int*,int*,int*,int*);

//PARPACK routines (parallel version)
#ifdef MPI_COMMS


extern int _AFT(pznaupd) (int *comm, int *ido, char *bmat, int *n, char *which, int *nev, double *tol,
                         std::complex<double> *resid, int *ncv, std::complex<double> *v, int *ldv, 
                         int *iparam, int *ipntr, std::complex<double> *workd, std::complex<double> *workl, 
                         int *lworkl, double *rwork, int *info, int bmat_size, int which_size );

extern int _AFT(pzneupd) (int *comm, int *comp_evecs, char *howmany, int *select, std::complex<double> *evals, 
                         std::complex<double> *v, int *ldv, std::complex<double> *sigma, std::complex<double> *workev, 
                         char *bmat, int *n, char *which, int *nev, double *tol, std::complex<double> *resid, 
                         int *ncv, std::complex<double> *v1, int *ldv1, int *iparam, int *ipntr, 
                         std::complex<double> *workd, std::complex<double> *workl, int *lworkl, double *rwork, int *info,
                         int howmany_size, int bmat_size, int which_size);

extern int _AFT(pmcinitdebug)(int*,int*,int*,int*,int*,int*,int*,int*);

#endif
}

