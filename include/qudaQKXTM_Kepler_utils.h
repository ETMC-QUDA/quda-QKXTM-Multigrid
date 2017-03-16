#include <comm_quda.h>
#include <quda_internal.h>
#include <quda.h>
#include <iostream>
#include <complex>
#include <cuda.h>
#include <color_spinor_field.h>
#include <enum_quda.h>
#include <typeinfo>
//DMH 
#include <dirac_quda.h>

#ifndef _QUDAQKXTM_KEPLER_UTILS_H
#define _QUDAQKXTM_KEPLER_UTILS_H


#define QUDAQKXTM_DIM 4
#define MAX_NSOURCES 1000
#define MAX_NMOMENTA 5000
#define MAX_TSINK 10
#define MAX_DEFLSTEPS 10
#define MAX_PROJS 5

#define LEXIC(it,iz,iy,ix,L) ( (it)*L[0]*L[1]*L[2] + (iz)*L[0]*L[1] + (iy)*L[0] + (ix) )
#define LEXIC_TZY(it,iz,iy,L) ( (it)*L[1]*L[2] + (iz)*L[1] + (iy) )
#define LEXIC_TZX(it,iz,ix,L) ( (it)*L[0]*L[2] + (iz)*L[0] + (ix) )
#define LEXIC_TYX(it,iy,ix,L) ( (it)*L[0]*L[1] + (iy)*L[0] + (ix) )
#define LEXIC_ZYX(iz,iy,ix,L) ( (iz)*L[0]*L[1] + (iy)*L[0] + (ix) )

//////////////////////////////////////
// Functions outside quda namespace //
//////////////////////////////////////

//Gauge utilities
void testPlaquette(void **gauge);
void testGaussSmearing(void **gauge);

namespace quda {
  
  enum SOURCE_T{UNITY,RANDOM};
  enum CORR_SPACE{POSITION_SPACE,MOMENTUM_SPACE};
  enum FILE_WRITE_FORMAT{ASCII_FORM,HDF5_FORM};
  
  typedef struct {
    int nsmearAPE;
    int nsmearGauss;
    double alphaAPE;
    double alphaGauss;
    int lL[QUDAQKXTM_DIM];
    int Nsources;
    int sourcePosition[MAX_NSOURCES][QUDAQKXTM_DIM];
    QudaPrecision Precision;
    int Q_sq;
    int Q_sq_loop;
    int Ntsink;
    int Nproj[MAX_TSINK];
    int traj;
    bool check_files;
    char *thrp_type[3];
    char *thrp_proj_type[5];
    char *baryon_type[10];
    char *meson_type[10];
    int tsinkSource[MAX_TSINK];
    int proj_list[MAX_TSINK][MAX_PROJS];
    int run3pt_src[MAX_NSOURCES];
    FILE_WRITE_FORMAT CorrFileFormat;
    SOURCE_T source_type;
    CORR_SPACE CorrSpace;
    bool HighMomForm;
    bool isEven;
    double kappa;
    double mu;
    double csw;
    double inv_tol;
  } qudaQKXTMinfo_Kepler;
  
  
#ifdef HAVE_ARPACK  
  enum WHICHSPECTRUM{SR,LR,SM,LM,SI,LI};
  
  typedef struct{
    int PolyDeg;     // degree of the Chebysev polynomial
    int nEv;         // Number of the eigenvectors we want
    int nKv;         // total size of Krylov space
    WHICHSPECTRUM spectrumPart; // which part of the spectrum to solve
    bool isACC;
    double tolArpack;
    int maxIterArpack;
    char arpack_logfile[512];
    double amin;
    double amax;
    bool isEven;
    bool isFullOp;
  }qudaQKXTM_arpackInfo;
#endif  

  typedef struct{
    int Nstoch;
    unsigned long int seed;
    int Ndump;
    char loop_fname[512];
#ifdef HAVE_ARPACK
    int nSteps_defl;
    int deflStep[MAX_DEFLSTEPS];
#endif
    int traj;
    int Nprint;
    int Nmoms;
    int Qsq;
    FILE_WRITE_FORMAT FileFormat;
    // = {"Scalar", "dOp", "Loops", "LoopsCv", "LpsDw", "LpsDwCv"}
    char *loop_type[6];
    // = { false  , false,  true   , true    ,  true  ,  true}
    bool loop_oneD[6];
    bool useTSM;
    int TSM_NHP;
    int TSM_NLP;
    int TSM_NdumpHP;
    int TSM_NdumpLP;
    int TSM_NprintHP;
    int TSM_NprintLP;
    long int TSM_maxiter;
    double TSM_tol;
  }qudaQKXTM_loopInfo;

  enum ALLOCATION_FLAG{NONE,HOST,DEVICE,BOTH,BOTH_EXTRA};
  enum CLASS_ENUM{FIELD,GAUGE,VECTOR,PROPAGATOR,PROPAGATOR3D,VECTOR3D};
  enum WHICHPARTICLE{PROTON,NEUTRON};
  enum WHICHPROJECTOR{G4,G5G123,G5G1,G5G2,G5G3};
  enum THRP_TYPE{THRP_LOCAL,THRP_NOETHER,THRP_ONED};
  
  enum APEDIM{D3,D4};
  
  ///////////////
  // Functions //
  ///////////////
  
  void init_qudaQKXTM_Kepler(qudaQKXTMinfo_Kepler *info);
  void printf_qudaQKXTM_Kepler();
  void run_calculatePlaq_kernel(cudaTextureObject_t gaugeTexPlaq, 
				int precision);
  void run_GaussianSmearing(void* out, cudaTextureObject_t vecTex, 
			    cudaTextureObject_t gaugeTex, int precision);
  void run_UploadToCuda(void* in, ColorSpinorField &qudaVec, int precision, 
			bool isEven);
  void run_DownloadFromCuda(void* out, ColorSpinorField &qudaVec, 
			    int precision, bool isEven);
  void run_ScaleVector(double a, void* inOut, int precision);
  void run_contractMesons (cudaTextureObject_t texProp1, 
			   cudaTextureObject_t texProp2, void* corr, int it, 
			   int isource, int precision, CORR_SPACE CorrSpace);
  void run_contractBaryons(cudaTextureObject_t texProp1, 
			   cudaTextureObject_t texProp2, void* corr, int it, 
			   int isource, int precision, CORR_SPACE CorrSpace);
  void run_rotateToPhysicalBase(void* inOut, int sign, int precision);
  void run_castDoubleToFloat(void *out, void *in);
  void run_castFloatToDouble(void *out, void *in);
  void run_conjugate_vector(void *inOut, int precision);
  void run_apply_gamma5_vector(void *inOut, int precision);
  void run_conjugate_propagator(void *inOut, int precision);
  void run_apply_gamma5_propagator(void *inOut, int precision);
  void run_seqSourceFixSinkPart1(void* out, int timeslice, 
				 cudaTextureObject_t tex1, 
				 cudaTextureObject_t tex2, int c_nu,int c_c2,
				 WHICHPROJECTOR PID, WHICHPARTICLE PARTICLE, 
				 int precision);
  void run_seqSourceFixSinkPart2(void* out, int timeslice, 
				 cudaTextureObject_t tex, int c_nu, int c_c2,
				 WHICHPROJECTOR PID, WHICHPARTICLE PARTICLE, 
				 int precision);
  void run_fixSinkContractions(void* corrThp_local, void* corrThp_noether, 
			       void* corrThp_oneD,cudaTextureObject_t fwdTex,
			       cudaTextureObject_t seqTex, 
			       cudaTextureObject_t gaugeTex,
			       WHICHPARTICLE PARTICLE, int partflag, int it, 
			       int isource, int precision, 
			       CORR_SPACE CorrSpace);

} //End namespace quda

#endif
