#include <qudaQKXTM_Kepler.h>
#include <errno.h>
#include <mpi.h>  
#include <limits>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>
#include <cuPrintf.cu>

#define THREADS_PER_BLOCK 64
#define PI 3.141592653589793
//#define TIMING_REPORT
using namespace quda;
extern Topology *default_topo;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                           
// $$ Section 2: Constant Refeneces $$                                                                                                                                                                 
/* block for device constants */
__constant__ bool c_dimBreak[4];
__constant__ int c_nColor;
__constant__ int c_nDim;
__constant__ int c_localL[4];
__constant__ int c_plusGhost[4];
__constant__ int c_minusGhost[4];
__constant__ int c_stride;
__constant__ int c_surface[4];
__constant__ int c_nSpin;
__constant__ double c_alphaAPE;
__constant__ double c_alphaGauss;
__constant__ int c_threads;
__constant__ int c_eps[6][3];
__constant__ int c_sgn_eps[6];
__constant__ int c_procPosition[4];
__constant__ int c_totalL[4];
__constant__ int c_Nmoms;
__constant__ short int c_moms[MAX_NMOMENTA][3];
__constant__ short int c_mesons_indices[10][16][4];
__constant__ short int c_NTN_indices[16][4];
__constant__ short int c_NTR_indices[64][6];
__constant__ short int c_RTN_indices[64][6];
__constant__ short int c_RTR_indices[256][8];
__constant__ short int c_Delta_indices[3][16][4];
__constant__ float c_mesons_values[10][16];
__constant__ float c_NTN_values[16];
__constant__ float c_NTR_values[64];
__constant__ float c_RTN_values[64];
__constant__ float c_RTR_values[256];
__constant__ float c_Delta_values[3][16];
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////                                                                                                                                                    
/* Block for global variables */
float GK_deviceMemory = 0.;
int GK_nColor;
int GK_nSpin;
int GK_nDim;
int GK_strideFull;
double GK_alphaAPE;
double GK_alphaGauss;
int GK_localVolume;
int GK_totalVolume;
int GK_nsmearAPE;
int GK_nsmearGauss;
bool GK_dimBreak[QUDAQKXTM_DIM];
int GK_localL[QUDAQKXTM_DIM];
int GK_totalL[QUDAQKXTM_DIM];
int GK_nProc[QUDAQKXTM_DIM];
int GK_plusGhost[QUDAQKXTM_DIM];
int GK_minusGhost[QUDAQKXTM_DIM];
int GK_surface3D[QUDAQKXTM_DIM];
bool GK_init_qudaQKXTM_Kepler_flag = false;
int GK_Nsources;
int GK_sourcePosition[MAX_NSOURCES][QUDAQKXTM_DIM];
int GK_Nmoms;
short int GK_moms[MAX_NMOMENTA][3];
short int GK_mesons_indices[10][16][4] = {0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1,0,3,0,3,0,3,1,2,0,3,2,1,0,3,3,0,1,2,0,3,1,2,1,2,1,2,2,1,1,2,3,0,2,1,0,3,2,1,1,2,2,1,2,1,2,1,3,0,3,0,0,3,3,0,1,2,3,0,2,1,3,0,3,0,0,3,0,3,0,3,1,2,0,3,2,1,0,3,3,0,1,2,0,3,1,2,1,2,1,2,2,1,1,2,3,0,2,1,0,3,2,1,1,2,2,1,2,1,2,1,3,0,3,0,0,3,3,0,1,2,3,0,2,1,3,0,3,0,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1};
float GK_mesons_values[10][16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1};
short int GK_NTN_indices[16][4] = {0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2};
float GK_NTN_values[16] = {-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1};
short int GK_NTR_indices[64][6] = {0,1,0,3,0,2,0,1,0,3,1,3,0,1,0,3,2,0,0,1,0,3,3,1,0,1,1,2,0,2,0,1,1,2,1,3,0,1,1,2,2,0,0,1,1,2,3,1,0,1,2,1,0,2,0,1,2,1,1,3,0,1,2,1,2,0,0,1,2,1,3,1,0,1,3,0,0,2,0,1,3,0,1,3,0,1,3,0,2,0,0,1,3,0,3,1,1,0,0,3,0,2,1,0,0,3,1,3,1,0,0,3,2,0,1,0,0,3,3,1,1,0,1,2,0,2,1,0,1,2,1,3,1,0,1,2,2,0,1,0,1,2,3,1,1,0,2,1,0,2,1,0,2,1,1,3,1,0,2,1,2,0,1,0,2,1,3,1,1,0,3,0,0,2,1,0,3,0,1,3,1,0,3,0,2,0,1,0,3,0,3,1,2,3,0,3,0,2,2,3,0,3,1,3,2,3,0,3,2,0,2,3,0,3,3,1,2,3,1,2,0,2,2,3,1,2,1,3,2,3,1,2,2,0,2,3,1,2,3,1,2,3,2,1,0,2,2,3,2,1,1,3,2,3,2,1,2,0,2,3,2,1,3,1,2,3,3,0,0,2,2,3,3,0,1,3,2,3,3,0,2,0,2,3,3,0,3,1,3,2,0,3,0,2,3,2,0,3,1,3,3,2,0,3,2,0,3,2,0,3,3,1,3,2,1,2,0,2,3,2,1,2,1,3,3,2,1,2,2,0,3,2,1,2,3,1,3,2,2,1,0,2,3,2,2,1,1,3,3,2,2,1,2,0,3,2,2,1,3,1,3,2,3,0,0,2,3,2,3,0,1,3,3,2,3,0,2,0,3,2,3,0,3,1};
float GK_NTR_values[64] = {1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1};
short int GK_RTN_indices[64][6] = {0,3,0,1,0,2,0,3,0,1,1,3,0,3,0,1,2,0,0,3,0,1,3,1,0,3,1,0,0,2,0,3,1,0,1,3,0,3,1,0,2,0,0,3,1,0,3,1,0,3,2,3,0,2,0,3,2,3,1,3,0,3,2,3,2,0,0,3,2,3,3,1,0,3,3,2,0,2,0,3,3,2,1,3,0,3,3,2,2,0,0,3,3,2,3,1,1,2,0,1,0,2,1,2,0,1,1,3,1,2,0,1,2,0,1,2,0,1,3,1,1,2,1,0,0,2,1,2,1,0,1,3,1,2,1,0,2,0,1,2,1,0,3,1,1,2,2,3,0,2,1,2,2,3,1,3,1,2,2,3,2,0,1,2,2,3,3,1,1,2,3,2,0,2,1,2,3,2,1,3,1,2,3,2,2,0,1,2,3,2,3,1,2,1,0,1,0,2,2,1,0,1,1,3,2,1,0,1,2,0,2,1,0,1,3,1,2,1,1,0,0,2,2,1,1,0,1,3,2,1,1,0,2,0,2,1,1,0,3,1,2,1,2,3,0,2,2,1,2,3,1,3,2,1,2,3,2,0,2,1,2,3,3,1,2,1,3,2,0,2,2,1,3,2,1,3,2,1,3,2,2,0,2,1,3,2,3,1,3,0,0,1,0,2,3,0,0,1,1,3,3,0,0,1,2,0,3,0,0,1,3,1,3,0,1,0,0,2,3,0,1,0,1,3,3,0,1,0,2,0,3,0,1,0,3,1,3,0,2,3,0,2,3,0,2,3,1,3,3,0,2,3,2,0,3,0,2,3,3,1,3,0,3,2,0,2,3,0,3,2,1,3,3,0,3,2,2,0,3,0,3,2,3,1};
float GK_RTN_values[64] = {-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1};
short int GK_RTR_indices[256][8] = {0,3,0,3,0,2,0,2,0,3,0,3,0,2,1,3,0,3,0,3,0,2,2,0,0,3,0,3,0,2,3,1,0,3,0,3,1,3,0,2,0,3,0,3,1,3,1,3,0,3,0,3,1,3,2,0,0,3,0,3,1,3,3,1,0,3,0,3,2,0,0,2,0,3,0,3,2,0,1,3,0,3,0,3,2,0,2,0,0,3,0,3,2,0,3,1,0,3,0,3,3,1,0,2,0,3,0,3,3,1,1,3,0,3,0,3,3,1,2,0,0,3,0,3,3,1,3,1,0,3,1,2,0,2,0,2,0,3,1,2,0,2,1,3,0,3,1,2,0,2,2,0,0,3,1,2,0,2,3,1,0,3,1,2,1,3,0,2,0,3,1,2,1,3,1,3,0,3,1,2,1,3,2,0,0,3,1,2,1,3,3,1,0,3,1,2,2,0,0,2,0,3,1,2,2,0,1,3,0,3,1,2,2,0,2,0,0,3,1,2,2,0,3,1,0,3,1,2,3,1,0,2,0,3,1,2,3,1,1,3,0,3,1,2,3,1,2,0,0,3,1,2,3,1,3,1,0,3,2,1,0,2,0,2,0,3,2,1,0,2,1,3,0,3,2,1,0,2,2,0,0,3,2,1,0,2,3,1,0,3,2,1,1,3,0,2,0,3,2,1,1,3,1,3,0,3,2,1,1,3,2,0,0,3,2,1,1,3,3,1,0,3,2,1,2,0,0,2,0,3,2,1,2,0,1,3,0,3,2,1,2,0,2,0,0,3,2,1,2,0,3,1,0,3,2,1,3,1,0,2,0,3,2,1,3,1,1,3,0,3,2,1,3,1,2,0,0,3,2,1,3,1,3,1,0,3,3,0,0,2,0,2,0,3,3,0,0,2,1,3,0,3,3,0,0,2,2,0,0,3,3,0,0,2,3,1,0,3,3,0,1,3,0,2,0,3,3,0,1,3,1,3,0,3,3,0,1,3,2,0,0,3,3,0,1,3,3,1,0,3,3,0,2,0,0,2,0,3,3,0,2,0,1,3,0,3,3,0,2,0,2,0,0,3,3,0,2,0,3,1,0,3,3,0,3,1,0,2,0,3,3,0,3,1,1,3,0,3,3,0,3,1,2,0,0,3,3,0,3,1,3,1,1,2,0,3,0,2,0,2,1,2,0,3,0,2,1,3,1,2,0,3,0,2,2,0,1,2,0,3,0,2,3,1,1,2,0,3,1,3,0,2,1,2,0,3,1,3,1,3,1,2,0,3,1,3,2,0,1,2,0,3,1,3,3,1,1,2,0,3,2,0,0,2,1,2,0,3,2,0,1,3,1,2,0,3,2,0,2,0,1,2,0,3,2,0,3,1,1,2,0,3,3,1,0,2,1,2,0,3,3,1,1,3,1,2,0,3,3,1,2,0,1,2,0,3,3,1,3,1,1,2,1,2,0,2,0,2,1,2,1,2,0,2,1,3,1,2,1,2,0,2,2,0,1,2,1,2,0,2,3,1,1,2,1,2,1,3,0,2,1,2,1,2,1,3,1,3,1,2,1,2,1,3,2,0,1,2,1,2,1,3,3,1,1,2,1,2,2,0,0,2,1,2,1,2,2,0,1,3,1,2,1,2,2,0,2,0,1,2,1,2,2,0,3,1,1,2,1,2,3,1,0,2,1,2,1,2,3,1,1,3,1,2,1,2,3,1,2,0,1,2,1,2,3,1,3,1,1,2,2,1,0,2,0,2,1,2,2,1,0,2,1,3,1,2,2,1,0,2,2,0,1,2,2,1,0,2,3,1,1,2,2,1,1,3,0,2,1,2,2,1,1,3,1,3,1,2,2,1,1,3,2,0,1,2,2,1,1,3,3,1,1,2,2,1,2,0,0,2,1,2,2,1,2,0,1,3,1,2,2,1,2,0,2,0,1,2,2,1,2,0,3,1,1,2,2,1,3,1,0,2,1,2,2,1,3,1,1,3,1,2,2,1,3,1,2,0,1,2,2,1,3,1,3,1,1,2,3,0,0,2,0,2,1,2,3,0,0,2,1,3,1,2,3,0,0,2,2,0,1,2,3,0,0,2,3,1,1,2,3,0,1,3,0,2,1,2,3,0,1,3,1,3,1,2,3,0,1,3,2,0,1,2,3,0,1,3,3,1,1,2,3,0,2,0,0,2,1,2,3,0,2,0,1,3,1,2,3,0,2,0,2,0,1,2,3,0,2,0,3,1,1,2,3,0,3,1,0,2,1,2,3,0,3,1,1,3,1,2,3,0,3,1,2,0,1,2,3,0,3,1,3,1,2,1,0,3,0,2,0,2,2,1,0,3,0,2,1,3,2,1,0,3,0,2,2,0,2,1,0,3,0,2,3,1,2,1,0,3,1,3,0,2,2,1,0,3,1,3,1,3,2,1,0,3,1,3,2,0,2,1,0,3,1,3,3,1,2,1,0,3,2,0,0,2,2,1,0,3,2,0,1,3,2,1,0,3,2,0,2,0,2,1,0,3,2,0,3,1,2,1,0,3,3,1,0,2,2,1,0,3,3,1,1,3,2,1,0,3,3,1,2,0,2,1,0,3,3,1,3,1,2,1,1,2,0,2,0,2,2,1,1,2,0,2,1,3,2,1,1,2,0,2,2,0,2,1,1,2,0,2,3,1,2,1,1,2,1,3,0,2,2,1,1,2,1,3,1,3,2,1,1,2,1,3,2,0,2,1,1,2,1,3,3,1,2,1,1,2,2,0,0,2,2,1,1,2,2,0,1,3,2,1,1,2,2,0,2,0,2,1,1,2,2,0,3,1,2,1,1,2,3,1,0,2,2,1,1,2,3,1,1,3,2,1,1,2,3,1,2,0,2,1,1,2,3,1,3,1,2,1,2,1,0,2,0,2,2,1,2,1,0,2,1,3,2,1,2,1,0,2,2,0,2,1,2,1,0,2,3,1,2,1,2,1,1,3,0,2,2,1,2,1,1,3,1,3,2,1,2,1,1,3,2,0,2,1,2,1,1,3,3,1,2,1,2,1,2,0,0,2,2,1,2,1,2,0,1,3,2,1,2,1,2,0,2,0,2,1,2,1,2,0,3,1,2,1,2,1,3,1,0,2,2,1,2,1,3,1,1,3,2,1,2,1,3,1,2,0,2,1,2,1,3,1,3,1,2,1,3,0,0,2,0,2,2,1,3,0,0,2,1,3,2,1,3,0,0,2,2,0,2,1,3,0,0,2,3,1,2,1,3,0,1,3,0,2,2,1,3,0,1,3,1,3,2,1,3,0,1,3,2,0,2,1,3,0,1,3,3,1,2,1,3,0,2,0,0,2,2,1,3,0,2,0,1,3,2,1,3,0,2,0,2,0,2,1,3,0,2,0,3,1,2,1,3,0,3,1,0,2,2,1,3,0,3,1,1,3,2,1,3,0,3,1,2,0,2,1,3,0,3,1,3,1,3,0,0,3,0,2,0,2,3,0,0,3,0,2,1,3,3,0,0,3,0,2,2,0,3,0,0,3,0,2,3,1,3,0,0,3,1,3,0,2,3,0,0,3,1,3,1,3,3,0,0,3,1,3,2,0,3,0,0,3,1,3,3,1,3,0,0,3,2,0,0,2,3,0,0,3,2,0,1,3,3,0,0,3,2,0,2,0,3,0,0,3,2,0,3,1,3,0,0,3,3,1,0,2,3,0,0,3,3,1,1,3,3,0,0,3,3,1,2,0,3,0,0,3,3,1,3,1,3,0,1,2,0,2,0,2,3,0,1,2,0,2,1,3,3,0,1,2,0,2,2,0,3,0,1,2,0,2,3,1,3,0,1,2,1,3,0,2,3,0,1,2,1,3,1,3,3,0,1,2,1,3,2,0,3,0,1,2,1,3,3,1,3,0,1,2,2,0,0,2,3,0,1,2,2,0,1,3,3,0,1,2,2,0,2,0,3,0,1,2,2,0,3,1,3,0,1,2,3,1,0,2,3,0,1,2,3,1,1,3,3,0,1,2,3,1,2,0,3,0,1,2,3,1,3,1,3,0,2,1,0,2,0,2,3,0,2,1,0,2,1,3,3,0,2,1,0,2,2,0,3,0,2,1,0,2,3,1,3,0,2,1,1,3,0,2,3,0,2,1,1,3,1,3,3,0,2,1,1,3,2,0,3,0,2,1,1,3,3,1,3,0,2,1,2,0,0,2,3,0,2,1,2,0,1,3,3,0,2,1,2,0,2,0,3,0,2,1,2,0,3,1,3,0,2,1,3,1,0,2,3,0,2,1,3,1,1,3,3,0,2,1,3,1,2,0,3,0,2,1,3,1,3,1,3,0,3,0,0,2,0,2,3,0,3,0,0,2,1,3,3,0,3,0,0,2,2,0,3,0,3,0,0,2,3,1,3,0,3,0,1,3,0,2,3,0,3,0,1,3,1,3,3,0,3,0,1,3,2,0,3,0,3,0,1,3,3,1,3,0,3,0,2,0,0,2,3,0,3,0,2,0,1,3,3,0,3,0,2,0,2,0,3,0,3,0,2,0,3,1,3,0,3,0,3,1,0,2,3,0,3,0,3,1,1,3,3,0,3,0,3,1,2,0,3,0,3,0,3,1,3,1};
float GK_RTR_values[256] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
short int GK_Delta_indices[3][16][4] = {0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2};
float GK_Delta_values[3][16] = {1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1};
// for mpi use global  variables
MPI_Group GK_fullGroup , GK_spaceGroup , GK_timeGroup;
MPI_Comm GK_spaceComm , GK_timeComm;
int GK_localRank;
int GK_localSize;
int GK_timeRank;
int GK_timeSize;

//////////////////////////////////////////////////  
static void createMomenta(int Q_sq){
  int counter=0;
  for(int iQ = 0 ; iQ <= Q_sq ; iQ++){
    for(int nx = iQ ; nx >= -iQ ; nx--)
      for(int ny = iQ ; ny >= -iQ ; ny--)
        for(int nz = iQ ; nz >= -iQ ; nz--){
          if( nx*nx + ny*ny + nz*nz == iQ ){
            GK_moms[counter][0] = nx;
            GK_moms[counter][1] = ny;
            GK_moms[counter][2] = nz;
            counter++;
          }
        }
  }
  if(counter > MAX_NMOMENTA)errorQuda("Error exceeded max number of momenta\n");
  GK_Nmoms=counter;
}

void quda::init_qudaQKXTM_Kepler(qudaQKXTMinfo_Kepler *info){

  if(GK_init_qudaQKXTM_Kepler_flag == false){
    GK_nColor = 3;
    GK_nSpin = 4;
    GK_nDim = QUDAQKXTM_DIM;
    GK_alphaAPE = info->alphaAPE;
    GK_alphaGauss = info->alphaGauss;
    GK_nsmearAPE = info->nsmearAPE;
    GK_nsmearGauss = info->nsmearGauss;
    createMomenta(info->Q_sq);
    // from now on depends on lattice and break format we choose

    for(int i = 0 ; i < GK_nDim ; i++)
      GK_nProc[i] = comm_dim(i);
    
        for(int i = 0 ; i < GK_nDim ; i++){   // take local and total lattice
      GK_localL[i] = info->lL[i];
      GK_totalL[i] = GK_nProc[i] * GK_localL[i];
    }
  
    GK_localVolume = 1;
    GK_totalVolume = 1;
    for(int i = 0 ; i < GK_nDim ; i++){
      GK_localVolume *= GK_localL[i];
      GK_totalVolume *= GK_totalL[i];
    }

    GK_strideFull = GK_localVolume;

    for (int i=0; i<GK_nDim; i++) {
      GK_surface3D[i] = 1;
      for (int j=0; j<GK_nDim; j++) {
	if (i==j) continue;
	GK_surface3D[i] *= GK_localL[j];
      }
    }

  for(int i = 0 ; i < GK_nDim ; i++)
    if( GK_localL[i] == GK_totalL[i] )
      GK_surface3D[i] = 0;
    
    for(int i = 0 ; i < GK_nDim ; i++){
      GK_plusGhost[i] =0;
      GK_minusGhost[i] = 0;
    }
    
#ifdef MULTI_GPU
    int lastIndex = GK_localVolume;
    for(int i = 0 ; i < GK_nDim ; i++)
      if( GK_localL[i] < GK_totalL[i] ){
	GK_plusGhost[i] = lastIndex ;
	GK_minusGhost[i] = lastIndex + GK_surface3D[i];
	lastIndex += 2*GK_surface3D[i];
      }
#endif
    

    for(int i = 0 ; i < GK_nDim ; i++){
      if( GK_localL[i] < GK_totalL[i])
	GK_dimBreak[i] = true;
      else
	GK_dimBreak[i] = false;
    }

    const int eps[6][3]=
      {
	{0,1,2},
	{2,0,1},
	{1,2,0},
	{2,1,0},
	{0,2,1},
	{1,0,2}
      };
    
    const int sgn_eps[6]=
      {
	+1,+1,+1,-1,-1,-1
      };

    int procPosition[4];
    
    for(int i= 0 ; i < 4 ; i++)
      procPosition[i] = comm_coords(default_topo)[i];


    // put it zero but change it later
    GK_Nsources = info->Nsources;
    if(GK_Nsources > MAX_NSOURCES) errorQuda("Error you exceeded maximum number of source position\n");

    for(int is = 0 ; is < GK_Nsources ; is++)
      for(int i = 0 ; i < 4 ; i++)
	GK_sourcePosition[is][i] = info->sourcePosition[is][i];

    
    // initialization consist also from define device constants
    cudaMemcpyToSymbol(c_nColor, &GK_nColor, sizeof(int) );
    cudaMemcpyToSymbol(c_nSpin, &GK_nSpin, sizeof(int) );
    cudaMemcpyToSymbol(c_nDim, &GK_nDim, sizeof(int) );
    cudaMemcpyToSymbol(c_stride, &GK_strideFull, sizeof(int) );
    cudaMemcpyToSymbol(c_alphaAPE, &GK_alphaAPE , sizeof(double) );
    cudaMemcpyToSymbol(c_alphaGauss, &GK_alphaGauss , sizeof(double) );
    cudaMemcpyToSymbol(c_threads , &GK_localVolume , sizeof(double) ); // may change

    cudaMemcpyToSymbol(c_dimBreak , GK_dimBreak , QUDAQKXTM_DIM*sizeof(bool) );
    cudaMemcpyToSymbol(c_localL , GK_localL , QUDAQKXTM_DIM*sizeof(int) );
    cudaMemcpyToSymbol(c_totalL , GK_totalL , QUDAQKXTM_DIM*sizeof(int) );
    cudaMemcpyToSymbol(c_plusGhost , GK_plusGhost , QUDAQKXTM_DIM*sizeof(int) );
    cudaMemcpyToSymbol(c_minusGhost , GK_minusGhost , QUDAQKXTM_DIM*sizeof(int) );
    cudaMemcpyToSymbol(c_surface , GK_surface3D , QUDAQKXTM_DIM*sizeof(int) );
    
    cudaMemcpyToSymbol(c_eps, &(eps[0][0]) , 6*3*sizeof(int) );
    cudaMemcpyToSymbol(c_sgn_eps, sgn_eps , 6*sizeof(int) );

    cudaMemcpyToSymbol(c_procPosition, procPosition, QUDAQKXTM_DIM*sizeof(int));

    cudaMemcpyToSymbol(c_Nmoms, &GK_Nmoms, sizeof(int));
    cudaMemcpyToSymbol(c_moms, GK_moms, MAX_NMOMENTA*3*sizeof(short int));
    cudaMemcpyToSymbol(c_mesons_indices,GK_mesons_indices,10*16*4*sizeof(short int));
    cudaMemcpyToSymbol(c_NTN_indices,GK_NTN_indices,16*4*sizeof(short int));
    cudaMemcpyToSymbol(c_NTR_indices,GK_NTR_indices,64*6*sizeof(short int));
    cudaMemcpyToSymbol(c_RTN_indices,GK_RTN_indices,64*6*sizeof(short int));
    cudaMemcpyToSymbol(c_RTR_indices,GK_RTR_indices,256*8*sizeof(short int));
    cudaMemcpyToSymbol(c_Delta_indices,GK_Delta_indices,3*16*4*sizeof(short int));
    
    cudaMemcpyToSymbol(c_mesons_values,GK_mesons_values,10*16*sizeof(float));
    cudaMemcpyToSymbol(c_NTN_values,GK_NTN_values,16*sizeof(float));
    cudaMemcpyToSymbol(c_NTR_values,GK_NTR_values,64*sizeof(float));
    cudaMemcpyToSymbol(c_RTN_values,GK_RTN_values,64*sizeof(float));
    cudaMemcpyToSymbol(c_RTR_values,GK_RTR_values,256*sizeof(float));
    cudaMemcpyToSymbol(c_Delta_values,GK_Delta_values,3*16*sizeof(float));

    checkCudaError();

    // create groups of process to use mpi reduce only on spatial points
    MPI_Comm_group(MPI_COMM_WORLD, &GK_fullGroup);
    int space3D_proc;
    space3D_proc = GK_nProc[0] * GK_nProc[1] * GK_nProc[2];
    int *ranks = (int*) malloc(space3D_proc*sizeof(int));

    for(int i= 0 ; i < space3D_proc ; i++)
      ranks[i] = comm_coords(default_topo)[3] + GK_nProc[3]*i;


    //    for(int i= 0 ; i < space3D_proc ; i++)
    //      printf("%d (%d,%d,%d,%d)\n",comm_rank(),comm_coords(default_topo)[0],comm_coords(default_topo)[1],comm_coords(default_topo)[2],comm_coords(default_topo)[3]);

    //  for(int i= 0 ; i < space3D_proc ; i++)
    //printf("%d %d\n",comm_rank(),ranks[i]);

    MPI_Group_incl(GK_fullGroup,space3D_proc,ranks,&GK_spaceGroup);
    MPI_Group_rank(GK_spaceGroup,&GK_localRank);
    MPI_Group_size(GK_spaceGroup,&GK_localSize);
    MPI_Comm_create(MPI_COMM_WORLD, GK_spaceGroup , &GK_spaceComm);

    //if(GK_spaceComm == MPI_COMM_NULL) printf("NULL %d\n",comm_rank());
    //exit(-1);
    // create group of process to use mpi gather
    int *ranksTime = (int*) malloc(GK_nProc[3]*sizeof(int));

    for(int i=0 ; i < GK_nProc[3] ; i++)
      ranksTime[i] = i;
    
    MPI_Group_incl(GK_fullGroup,GK_nProc[3], ranksTime, &GK_timeGroup);
    MPI_Group_rank(GK_timeGroup, &GK_timeRank);
    MPI_Group_size(GK_timeGroup, &GK_timeSize);
    MPI_Comm_create(MPI_COMM_WORLD, GK_timeGroup, &GK_timeComm);

    //////////////////////////////////////////////////////////////////////////////
    free(ranks);
    free(ranksTime);

    GK_init_qudaQKXTM_Kepler_flag = true;
    printfQuda("qudaQKXTM_Kepler has been initialized\n");
  }
  else
    return;

}

void quda::printf_qudaQKXTM_Kepler(){

  if(GK_init_qudaQKXTM_Kepler_flag == false) errorQuda("You must initialize init_qudaQKXTM_Kepler first");
  printfQuda("Number of colors is %d\n",GK_nColor);
  printfQuda("Number of spins is %d\n",GK_nSpin);
  printfQuda("Number of dimensions is %d\n",GK_nDim);
  printfQuda("Number of process in each direction is (x,y,z,t) %d x %d x %d x %d\n",GK_nProc[0],GK_nProc[1],GK_nProc[2],GK_nProc[3]);
  printfQuda("Total lattice is (x,y,z,t) %d x %d x %d x %d\n",GK_totalL[0],GK_totalL[1],GK_totalL[2],GK_totalL[3]);
  printfQuda("Local lattice is (x,y,z,t) %d x %d x %d x %d\n",GK_localL[0],GK_localL[1],GK_localL[2],GK_localL[3]);
  printfQuda("Total volume is %d\n",GK_totalVolume);
  printfQuda("Local volume is %d\n",GK_localVolume);
  printfQuda("Surface is (x,y,z,t) ( %d , %d , %d , %d)\n",GK_surface3D[0],GK_surface3D[1],GK_surface3D[2],GK_surface3D[3]);
  printfQuda("The plus Ghost points in directions (x,y,z,t) ( %d , %d , %d , %d )\n",GK_plusGhost[0],GK_plusGhost[1],GK_plusGhost[2],GK_plusGhost[3]);
  printfQuda("The Minus Ghost points in directixons (x,y,z,t) ( %d , %d , %d , %d )\n",GK_minusGhost[0],GK_minusGhost[1],GK_minusGhost[2],GK_minusGhost[3]);
  printfQuda("For APE smearing we use nsmear = %d , alpha = %lf\n",GK_nsmearAPE,GK_alphaAPE);
  printfQuda("For Gauss smearing we use nsmear = %d , alpha = %lf\n",GK_nsmearGauss,GK_alphaGauss);
  printfQuda("I got %d source positions to work on\n",GK_Nsources);
  printfQuda("I got %d number of momenta to work on\n",GK_Nmoms);
}



static __inline__ __device__ double2 fetch_double2(cudaTextureObject_t t, int i)
{
  int4 v =tex1Dfetch<int4>(t,i);
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}

static __inline__ __device__ float2 fetch_float2(cudaTextureObject_t t, int i)
{
  float2 v = tex1Dfetch<float2>(t,i);
  return v;
}

template<typename Float2>
__device__ inline Float2 operator*(const Float2 a, const Float2 b){
  Float2 res;
  res.x = a.x*b.x - a.y*b.y;
  res.y = a.x*b.y + a.y*b.x;
  return res;
}

/*
template<typename Float2, typename Float>
__device__ inline Float2 operator*(const Float a , const Float2 b){
  Float2 res;
  res.x = a*b.x;
  res.y = a*b.y;
  return res;
} 
*/

__device__ inline float2 operator*(const float a , const float2 b){
  float2 res;
  res.x = a*b.x;
  res.y = a*b.y;
  return res;
} 


__device__ inline double2 operator*(const double a , const double2 b){
  double2 res;
  res.x = a*b.x;
  res.y = a*b.y;
  return res;
} 


template<typename Float2>
__device__ inline Float2 operator*(const int a , const Float2 b){
  Float2 res;
  res.x = a*b.x;
  res.y = a*b.y;
  return res;
} 

template<typename Float2>
__device__ inline Float2 operator+(const Float2 a, const Float2 b){
  Float2 res;
  res.x = a.x + b.x;
  res.y = a.y + b.y;
  return res;
}

template<typename Float2>
__device__ inline Float2 operator-(const Float2 a, const Float2 b){
  Float2 res;
  res.x = a.x - b.x;
  res.y = a.y - b.y;
  return res;
}

template<typename Float2>
__device__ inline Float2 conj(const Float2 a){
  Float2 res;
  res.x = a.x;
  res.y = -a.y;
  return res;
}

__device__ inline float norm(const float2 a){
  float res;
  res = sqrt(a.x*a.x + a.y*a.y);
  return res;
}

__device__ inline double norm(const double2 a){
  double res;
  res = sqrt(a.x*a.x + a.y*a.y);
  return res;
}

template<typename Float2>
__device__ inline Float2 get_Projector(Float2 projector[4][4], 
				       WHICHPARTICLE PARTICLE, 
				       WHICHPROJECTOR PID){ 
  // important Projectors must be in twisted basis
#include <projectors_tm_base.h>
}

template<typename Float2>
__device__ inline Float2 get_Operator(Float2 gamma[4][4], int flag, 
				      WHICHPARTICLE TESTPARTICLE, 
				      int partFlag){
#include <gammas_tm_base.h>
}

#include <core_def_Kepler.h>

__global__ void calculatePlaq_kernel_double(cudaTextureObject_t gaugeTexPlaq,
					    double *partial_plaq){
#define FLOAT2 double2
#define FLOAT double
#define READGAUGE_FLOAT READGAUGE_double
#include <plaquette_core_Kepler.h>
#undef FLOAT2
#undef FLOAT
#undef READGAUGE_FLOAT
}

__global__ void calculatePlaq_kernel_float(cudaTextureObject_t gaugeTexPlaq,
					   float *partial_plaq){
#define FLOAT2 float2
#define FLOAT float
#define READGAUGE_FLOAT READGAUGE_float
#include <plaquette_core_Kepler.h>
#undef READGAUGE_FLOAT
#undef FLOAT2
#undef FLOAT
}

__global__ void gaussianSmearing_kernel_float(float2* out,
					      cudaTextureObject_t vecInTex,
					      cudaTextureObject_t gaugeTex ){
#define FLOAT2 float2
#define READGAUGE_FLOAT READGAUGE_float
#define READVECTOR_FLOAT READVECTOR_float
#include <Gauss_core_Kepler.h>
#undef READGAUGE_FLOAT
#undef READVECTOR_FLOAT
#undef FLOAT2
}

__global__ void gaussianSmearing_kernel_double(double2* out,
					       cudaTextureObject_t vecInTex,
					       cudaTextureObject_t gaugeTex ){
#define FLOAT2 double2
#define READGAUGE_FLOAT READGAUGE_double
#define READVECTOR_FLOAT READVECTOR_double
#include <Gauss_core_Kepler.h>
#undef READGAUGE_FLOAT
#undef READVECTOR_FLOAT
#undef FLOAT2
}

__global__ void contractMesons_kernel_float(float2* block, 
					    cudaTextureObject_t prop1Tex, 
					    cudaTextureObject_t prop2Tex,
					    int it, int x0, int y0, int z0){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2
#include <contractMesons_core_Kepler.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void contractMesons_kernel_PosSpace_float(float2* block, 
						     cudaTextureObject_t prop1Tex, 
						     cudaTextureObject_t prop2Tex,
						     int it, int x0, int y0, int z0){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2
#include <contractMesons_core_Kepler_PosSpace.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void contractMesons_kernel_double(double2* block, 
					     cudaTextureObject_t prop1Tex, 
					     cudaTextureObject_t prop2Tex,
					     int it, int x0, int y0, int z0){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2
#include <contractMesons_core_Kepler.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void contractBaryons_kernel_float(float2* block, 
					     cudaTextureObject_t prop1Tex, 
					     cudaTextureObject_t prop2Tex,
					     int it, int x0, int y0, int z0, int ip){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2
#include <contractBaryons_core_Kepler.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void contractBaryons_kernel_PosSpace_float(float2* block, 
						      cudaTextureObject_t prop1Tex, 
						      cudaTextureObject_t prop2Tex,
						      int it, int x0, int y0, int z0, int ip){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2
#include <contractBaryons_core_Kepler_PosSpace.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}


/*
__global__ void contractBaryons_kernel_double(double2* block, cudaTextureObject_t prop1Tex, cudaTextureObject_t prop2Tex,int it, int x0, int y0, int z0, int ip){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2
#include <contractBaryons_core_Kepler.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}
*/

__global__ void seqSourceFixSinkPart1_kernel_float(float2* out, int timeslice,
						   cudaTextureObject_t tex1, 
						   cudaTextureObject_t tex2, 
						   int c_nu, int c_c2, 
						   WHICHPROJECTOR PID, 
						   WHICHPARTICLE PARTICLE ){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2
#include <seqSourceFixSinkPart1_core_Kepler.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void seqSourceFixSinkPart2_kernel_float(float2* out, 
						   int timeslice, 
						   cudaTextureObject_t tex, 
						   int c_nu, int c_c2, 
						   WHICHPROJECTOR PID, 
						   WHICHPARTICLE PARTICLE ){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2
#include <seqSourceFixSinkPart2_core_Kepler.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void seqSourceFixSinkPart1_kernel_double(double2* out, 
						    int timeslice, 
						    cudaTextureObject_t tex1, 
						    cudaTextureObject_t tex2, 
						    int c_nu, int c_c2, 
						    WHICHPROJECTOR PID, 
						    WHICHPARTICLE PARTICLE ){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2
#include <seqSourceFixSinkPart1_core_Kepler.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void seqSourceFixSinkPart2_kernel_double(double2* out, 
						    int timeslice, 
						    cudaTextureObject_t tex, 
						    int c_nu, int c_c2,
						    WHICHPROJECTOR PID, 
						    WHICHPARTICLE PARTICLE ){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2
#include <seqSourceFixSinkPart2_core_Kepler.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

//- Fix Sink kernels, ultra-local
__global__ void fixSinkContractions_local_kernel_float(float2* block,  
						       cudaTextureObject_t fwdTex, 
						       cudaTextureObject_t seqTex, 
						       WHICHPARTICLE TESTPARTICLE, 
						       int partflag, int it, 
						       int x0, int y0, int z0){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2
#include <fixSinkContractions_local_core_Kepler.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void fixSinkContractions_local_kernel_PosSpace_float(float2* block,  
								cudaTextureObject_t fwdTex, 
								cudaTextureObject_t seqTex, 
								WHICHPARTICLE TESTPARTICLE, 
								int partflag, int it, 
								int x0, int y0, int z0){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2
#include <fixSinkContractions_local_core_Kepler_PosSpace.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void fixSinkContractions_local_kernel_double(double2* block,  
							cudaTextureObject_t fwdTex, 
							cudaTextureObject_t seqTex, 
							WHICHPARTICLE TESTPARTICLE, 
							int partflag, int it, 
							int x0, int y0, int z0){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2
#include <fixSinkContractions_local_core_Kepler.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void fixSinkContractions_local_kernel_PosSpace_double(double2* block,  
								 cudaTextureObject_t fwdTex, 
								 cudaTextureObject_t seqTex, 
								 WHICHPARTICLE TESTPARTICLE, 
								 int partflag, int it, 
								 int x0, int y0, int z0){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2
#include <fixSinkContractions_local_core_Kepler_PosSpace.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}
//-----------------------------------------------

//- Fix Sink kernels, noether
__global__ void fixSinkContractions_noether_kernel_float(float2* block,  
							 cudaTextureObject_t fwdTex, 
							 cudaTextureObject_t seqTex, 
							 cudaTextureObject_t gaugeTex, 
							 WHICHPARTICLE TESTPARTICLE, 
							 int partflag, int it, 
							 int x0, int y0, int z0){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2
#include <fixSinkContractions_noether_core_Kepler.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void fixSinkContractions_noether_kernel_PosSpace_float(float2* block,  
								  cudaTextureObject_t fwdTex, 
								  cudaTextureObject_t seqTex, 
								  cudaTextureObject_t gaugeTex, 
								  WHICHPARTICLE TESTPARTICLE, 
								  int partflag, int it, 
								  int x0, int y0, int z0){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2
#include <fixSinkContractions_noether_core_Kepler_PosSpace.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void fixSinkContractions_noether_kernel_double(double2* block,  
							  cudaTextureObject_t fwdTex, 
							  cudaTextureObject_t seqTex, 
							  cudaTextureObject_t gaugeTex, 
							  WHICHPARTICLE TESTPARTICLE, 
							  int partflag, int it, 
							  int x0, int y0, int z0){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2
#include <fixSinkContractions_noether_core_Kepler.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void fixSinkContractions_noether_kernel_PosSpace_double(double2* block,  
								   cudaTextureObject_t fwdTex, 
								   cudaTextureObject_t seqTex, 
								   cudaTextureObject_t gaugeTex, 
								   WHICHPARTICLE TESTPARTICLE, 
								   int partflag, int it, 
								   int x0, int y0, int z0){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2
#include <fixSinkContractions_noether_core_Kepler_PosSpace.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}
//-----------------------------------------------

//- Fix Sink kernels, one-derivative
__global__ void fixSinkContractions_oneD_kernel_float(float2* block,  
						      cudaTextureObject_t fwdTex, 
						      cudaTextureObject_t seqTex, 
						      cudaTextureObject_t gaugeTex, 
						      WHICHPARTICLE TESTPARTICLE, 
						      int partflag, int it, 
						      int dir, int x0, 
						      int y0, int z0){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2
#include <fixSinkContractions_oneD_core_Kepler.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void fixSinkContractions_oneD_kernel_PosSpace_float(float2* block,  
							       cudaTextureObject_t fwdTex, 
							       cudaTextureObject_t seqTex, 
							       cudaTextureObject_t gaugeTex, 
							       WHICHPARTICLE TESTPARTICLE, 
							       int partflag, int it, 
							       int dir, int x0, 
							       int y0, int z0){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2
#include <fixSinkContractions_oneD_core_Kepler_PosSpace.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void fixSinkContractions_oneD_kernel_double(double2* block,  
						       cudaTextureObject_t fwdTex, 
						       cudaTextureObject_t seqTex, 
						       cudaTextureObject_t gaugeTex, 
						       WHICHPARTICLE TESTPARTICLE, 
						       int partflag, int it, 
						       int dir, int x0, 
						       int y0, int z0){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2
#include <fixSinkContractions_oneD_core_Kepler.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}

__global__ void fixSinkContractions_oneD_kernel_PosSpace_double(double2* block, 
								cudaTextureObject_t fwdTex, 
								cudaTextureObject_t seqTex, 
								cudaTextureObject_t gaugeTex, 
								WHICHPARTICLE TESTPARTICLE, 
								int partflag, int it, 
								int dir, int x0, 
								int y0, int z0){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2
#include <fixSinkContractions_oneD_core_Kepler_PosSpace.h>
#undef FETCH_FLOAT2
#undef FLOAT2
#undef FLOAT
}
//-----------------------------------------------


template<typename Float, typename Float2>
__global__ void scaleVector_kernel(Float a, Float2* inOut){
#include <scaleVector_core_Kepler.h>
}

template<typename Float2> 
__global__ void uploadToCuda_kernel(Float2 *in, double2 *outEven, double2 *outOdd){
#include <uploadToCuda_core_Kepler.h>
}

template<typename Float2> 
__global__ void downloadFromCuda_kernel(Float2 *out, double2 *inEven, double2 *inOdd){
#include <downloadFromCuda_core_Kepler.h>
}

template<typename Float2>
__global__ void rotateToPhysicalBase_kernel(Float2 *inOut, int sign){
#include <rotateToPhysicalBase_core_Kepler.h>
}

__global__ void castDoubleToFloat_kernel(float2 *out, double2 *in){
#include <castDoubleToFloat_core_Kepler.h>
}

__global__ void castFloatToDouble_kernel(double2 *out, float2 *in){
#include <castFloatToDouble_core_Kepler.h>
}

template<typename Float2>
__global__ void conjugate_vector_kernel(Float2 *inOut){
#include <conjugate_vector_core_Kepler.h>
}

template<typename Float2>
__global__ void apply_gamma5_vector_kernel(Float2 *inOut){
#include <apply_gamma5_vector_core_Kepler.h>
}

template<typename Float2>
__global__ void conjugate_propagator_kernel(Float2 *inOut){
#include <conjugate_propagator_core_Kepler.h>
}

template<typename Float2>
__global__ void apply_gamma5_propagator_kernel(Float2 *inOut){
#include <apply_gamma5_propagator_core_Kepler.h>
}

template<typename Float>
static Float calculatePlaq_kernel(cudaTextureObject_t gaugeTexPlaq){
  Float plaquette = 0.;
  Float globalPlaquette = 0.;

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  Float *h_partial_plaq = NULL;
  Float *d_partial_plaq = NULL;
  h_partial_plaq = (Float*) malloc(gridDim.x * sizeof(Float) );
  if(h_partial_plaq == NULL) errorQuda("Error allocate memory for host partial plaq");
  cudaMalloc((void**)&d_partial_plaq, gridDim.x * sizeof(Float));

#ifdef TIMING_REPORT
  cudaEvent_t start,stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
#endif
  if( typeid(Float) == typeid(float) )
    calculatePlaq_kernel_float<<<gridDim,blockDim>>>(gaugeTexPlaq,(float*) d_partial_plaq);
  else if(typeid(Float) == typeid(double))
    calculatePlaq_kernel_double<<<gridDim,blockDim>>>(gaugeTexPlaq,(double*) d_partial_plaq);
#ifdef TIMING_REPORT
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printfQuda("Elapsed time for plaquette kernel is %f ms\n",elapsedTime);
#endif
  cudaMemcpy(h_partial_plaq, d_partial_plaq , gridDim.x * sizeof(Float) , cudaMemcpyDeviceToHost);

  for(int i = 0 ; i < gridDim.x ; i++)
    plaquette += h_partial_plaq[i];

  free(h_partial_plaq);
  cudaFree(d_partial_plaq);
  checkCudaError();

  int rc;
  if(typeid(Float) == typeid(double))
    rc = MPI_Allreduce(&plaquette , &globalPlaquette , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);
  else if( typeid(Float) == typeid(float) )
    rc = MPI_Allreduce(&plaquette , &globalPlaquette , 1 , MPI_FLOAT , MPI_SUM , MPI_COMM_WORLD);

  if( rc != MPI_SUCCESS ) errorQuda("Error in MPI reduction for plaquette");
  return globalPlaquette/(GK_totalVolume*GK_nColor*6);
}

void quda::run_calculatePlaq_kernel(cudaTextureObject_t gaugeTexPlaq, 
				    int precision){
  if(precision == 4){
    float plaq = calculatePlaq_kernel<float>(gaugeTexPlaq);
    printfQuda("Calculated plaquette in single precision is %f\n",plaq);
  }
  else if(precision == 8){
    double plaq = calculatePlaq_kernel<double>(gaugeTexPlaq);
    printfQuda("Calculated plaquette in double precision is %lf\n",plaq);
  }
  else{
    errorQuda("Precision not supported\n");
  }
    
}

template<typename Float>
static void gaussianSmearing_kernel(void* out,
				    cudaTextureObject_t vecInTex, 
				    cudaTextureObject_t gaugeTex){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

#ifdef TIMING_REPORT
  cudaEvent_t start,stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
#endif

  if( typeid(Float) == typeid(float) )
    gaussianSmearing_kernel_float<<<gridDim,blockDim>>>((float2*) out, vecInTex, gaugeTex);
  else if(typeid(Float) == typeid(double))
    gaussianSmearing_kernel_double<<<gridDim,blockDim>>>((double2*) out, vecInTex, gaugeTex);
  
  checkCudaError();

#ifdef TIMING_REPORT
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printfQuda("Elapsed time for 1 step in gaussian smearing is %f ms\n",elapsedTime);
#endif

}

void quda::run_GaussianSmearing(void* out, 
				cudaTextureObject_t vecInTex, 
				cudaTextureObject_t gaugeTex, 
				int precision){
  if(precision == 4){
    gaussianSmearing_kernel<float>(out,vecInTex,gaugeTex);
  }
  else if(precision == 8){
    gaussianSmearing_kernel<double>(out,vecInTex,gaugeTex);
  }
  else{
    errorQuda("Precision not supported\n");
  }    
}

void quda::run_UploadToCuda(void* in,ColorSpinorField &qudaVec, int precision, bool isEven){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  if( qudaVec.SiteSubset() == QUDA_PARITY_SITE_SUBSET ){
    if( isEven ){
      if(precision == 4){
	uploadToCuda_kernel<float2><<<gridDim,blockDim>>>((float2*) in,(double2*) qudaVec.V(), NULL );
      }
      else if(precision == 8){
	uploadToCuda_kernel<double2><<<gridDim,blockDim>>>((double2*) in,(double2*) qudaVec.V(), NULL );
      }
      else{
	errorQuda("Precision not supported\n");
      }       
    }
    else{
      if(precision == 4){
	uploadToCuda_kernel<float2><<<gridDim,blockDim>>>((float2*) in, NULL,(double2*) qudaVec.V() );
      }
      else if(precision == 8){
	uploadToCuda_kernel<double2><<<gridDim,blockDim>>>((double2*) in, NULL,(double2*) qudaVec.V());
      }
      else{
	errorQuda("Precision not supported\n");
      }       
    }
  }
  else{
    //    printfQuda("### Uploading to QUDA both even and odd sites\n");
    if(precision == 4){
      uploadToCuda_kernel<float2><<<gridDim,blockDim>>>((float2*) in,(double2*) qudaVec.Even().V(), (double2*) qudaVec.Odd().V() );
    }
    else if(precision == 8){
      uploadToCuda_kernel<double2><<<gridDim,blockDim>>>((double2*) in,(double2*) qudaVec.Even().V(), (double2*) qudaVec.Odd().V() );
    }
    else{
      errorQuda("Precision not supported\n");
    } 
  } 
  checkCudaError();
}

void quda::run_DownloadFromCuda(void* out,ColorSpinorField &qudaVec, int precision, bool isEven){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  if( qudaVec.SiteSubset() == QUDA_PARITY_SITE_SUBSET ){
    if( isEven ){
      if(precision == 4){
	downloadFromCuda_kernel<float2><<<gridDim,blockDim>>>((float2*) out,(double2*) qudaVec.V(), NULL );
      }
      else if(precision == 8){
	downloadFromCuda_kernel<double2><<<gridDim,blockDim>>>((double2*) out,(double2*) qudaVec.V(), NULL );
      }
      else{
	errorQuda("Precision not supported\n");
      }       
    }
    else{
      if(precision == 4){
	downloadFromCuda_kernel<float2><<<gridDim,blockDim>>>((float2*) out, NULL,(double2*) qudaVec.V() );
      }
      else if(precision == 8){
	downloadFromCuda_kernel<double2><<<gridDim,blockDim>>>((double2*) out, NULL,(double2*) qudaVec.V());
      }
      else{
	errorQuda("Precision not supported\n");
      }       
    }
  }
  else{
    //    printfQuda("### Downloading from QUDA both even and odd sites\n");
    if(precision == 4){
      downloadFromCuda_kernel<float2><<<gridDim,blockDim>>>((float2*) out,(double2*) qudaVec.Even().V(), (double2*) qudaVec.Odd().V() );
    }
    else if(precision == 8){
      downloadFromCuda_kernel<double2><<<gridDim,blockDim>>>((double2*) out,(double2*) qudaVec.Even().V(), (double2*) qudaVec.Odd().V() );
    }
    else{
      errorQuda("Precision not supported\n");
    } 
  } 
  checkCudaError();
}

void quda::run_ScaleVector(double a, void* inOut, int precision){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  if(precision == 4){
    scaleVector_kernel<float,float2><<<gridDim,blockDim>>>((float) a, (float2*) inOut);
  }
  else if(precision == 8){
    scaleVector_kernel<double,double2><<<gridDim,blockDim>>>((double) a, (double2*) inOut);
  }
  else{
    errorQuda("Precision not supported\n");
  }    
  checkCudaError();
}

template<typename Float2,typename Float>
static void contractMesons_kernel(cudaTextureObject_t texProp1,
				  cudaTextureObject_t texProp2,
				  Float (*corr)[2][10], 
				  int it, int isource, 
				  CORR_SPACE CorrSpace){

  if( typeid(Float2) != typeid(float2) ) errorQuda("Unsupported precision for Meson 2pt Contraction kernels!\n");

  int SpVol = GK_localVolume/GK_localL[3];

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (SpVol + blockDim.x -1)/blockDim.x , 1 , 1); // spawn threads only for the spatial volume

  Float *h_partial_block = NULL;
  Float *d_partial_block = NULL;

  if(CorrSpace==POSITION_SPACE){
    long int alloc_size = blockDim.x * gridDim.x; // That's basically local spatial volume
    h_partial_block = (Float*)malloc(alloc_size*2*10*2*sizeof(Float));
    if(h_partial_block == NULL) errorQuda("contractMesons_kernel: Cannot allocate host block.\n");

    cudaMalloc((void**)&d_partial_block, alloc_size*2*10*2*sizeof(Float));
    checkCudaError();

    contractMesons_kernel_PosSpace_float<<<gridDim,blockDim>>>((float2*) d_partial_block,  texProp1, texProp2, it,  GK_sourcePosition[isource][0] , GK_sourcePosition[isource][1], GK_sourcePosition[isource][2]);
    checkCudaError();

    cudaMemcpy(h_partial_block , d_partial_block , alloc_size*2*10*2*sizeof(Float) , cudaMemcpyDeviceToHost);
    checkCudaError();
    
    //-C.K. Copy host block into corr buffer
    for(int pt = 0; pt < 2 ; pt++){
      for( int mes = 0; mes < 10; mes++){
	for(int sv = 0; sv < SpVol ; sv++){
	  corr[ 0 + 2*sv + 2*SpVol*it ][pt][mes] = h_partial_block[ 0 + 2*sv + 2*SpVol*mes + 2*SpVol*10*pt ];
	  corr[ 1 + 2*sv + 2*SpVol*it ][pt][mes] = h_partial_block[ 1 + 2*sv + 2*SpVol*mes + 2*SpVol*10*pt ];
	}
      }
    }

    free(h_partial_block);
    cudaFree(d_partial_block);
    checkCudaError();
  }
  else if(CorrSpace==MOMENTUM_SPACE){
    h_partial_block = (Float*)malloc(GK_Nmoms*2*10*gridDim.x*2*sizeof(Float));
    if(h_partial_block == NULL) errorQuda("Error problem with allocation\n");

    cudaMalloc((void**)&d_partial_block, GK_Nmoms*2*10*gridDim.x*2 * sizeof(Float) );
    checkCudaError();

    Float *reduction =(Float*) calloc(GK_Nmoms*2*10*2,sizeof(Float));
  
    contractMesons_kernel_float<<<gridDim,blockDim>>>((float2*) d_partial_block,  texProp1, texProp2, it,  GK_sourcePosition[isource][0] , GK_sourcePosition[isource][1], GK_sourcePosition[isource][2]);
    checkCudaError();

    cudaMemcpy(h_partial_block , d_partial_block , GK_Nmoms*2*10*gridDim.x*2 * sizeof(Float) , cudaMemcpyDeviceToHost);
    checkCudaError();

    for(int imom = 0 ; imom < GK_Nmoms ; imom++)
      for(int iu = 0 ; iu < 2 ; iu++)
	for(int ip = 0 ; ip < 10 ; ip++)
	  for(int i =0 ; i < gridDim.x ; i++){
	    reduction[imom*2*10*2 + iu*10*2 + ip*2 + 0] += h_partial_block[imom*2*10*gridDim.x*2 + iu*10*gridDim.x*2 + ip*gridDim.x*2 + i*2 + 0];
	    reduction[imom*2*10*2 + iu*10*2 + ip*2 + 1] += h_partial_block[imom*2*10*gridDim.x*2 + iu*10*gridDim.x*2 + ip*gridDim.x*2 + i*2 + 1];
	  }
    
    for(int imom = 0 ; imom < GK_Nmoms ; imom++)
      for(int iu = 0 ; iu < 2 ; iu++)
	for(int ip = 0 ; ip < 10 ; ip++){
	  corr[it*GK_Nmoms*2 + imom*2 + 0][iu][ip] = reduction[imom*2*10*2 + iu*10*2 + ip*2 + 0];
	  corr[it*GK_Nmoms*2 + imom*2 + 1][iu][ip] = reduction[imom*2*10*2 + iu*10*2 + ip*2 + 1];
	}


    free(h_partial_block);
    cudaFree(d_partial_block);
    checkCudaError();
    free(reduction);
  }//-CorrSpace else
  else errorQuda("contractMesons_kernel: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

}

void quda::run_contractMesons(cudaTextureObject_t texProp1,
			      cudaTextureObject_t texProp2,
			      void* corr,  int it, int isource, 
			      int precision, CORR_SPACE CorrSpace){

  if (CorrSpace==POSITION_SPACE)     cudaFuncSetCacheConfig(contractMesons_kernel_PosSpace_float,cudaFuncCachePreferShared);
  else if(CorrSpace==MOMENTUM_SPACE) cudaFuncSetCacheConfig(contractMesons_kernel_float         ,cudaFuncCachePreferShared);
  else errorQuda("run_contractMesons: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");
  checkCudaError();

  if(precision == 4) contractMesons_kernel<float2,float>(texProp1,texProp2,(float(*)[2][10]) corr,it, isource, CorrSpace);
  else if(precision == 8) errorQuda("Double precision in Meson 2pt Contractions unsupported!!!\n");
  else errorQuda("run_contractMesons: Precision %d not supported\n",precision);

}

template<typename Float2,typename Float>
static void contractBaryons_kernel(cudaTextureObject_t texProp1,
				   cudaTextureObject_t texProp2,
				   Float (*corr)[2][10][4][4], 
				   int it, int isource, 
				   CORR_SPACE CorrSpace){
  
  if( typeid(Float2) != typeid(float2) ) errorQuda("Unsupported precision for Baryon 2pt Contraction kernels!\n");

  int SpVol = GK_localVolume/GK_localL[3];

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (SpVol + blockDim.x -1)/blockDim.x , 1 , 1); // spawn threads only for the spatial volume

  Float *h_partial_block = NULL;
  Float *d_partial_block = NULL;

  if(CorrSpace==POSITION_SPACE){
    long int alloc_size = blockDim.x * gridDim.x; // That's basically local spatial volume
    h_partial_block = (Float*)malloc(alloc_size*2*4*4*2*sizeof(Float));
    if(h_partial_block == NULL) errorQuda("contractBaryons_kernel: Cannot allocate host block.\n");

    cudaMalloc((void**)&d_partial_block, alloc_size*2*4*4*2*sizeof(Float));
    checkCudaError();

    for(int ip = 0 ; ip < 10 ; ip++){
      contractBaryons_kernel_PosSpace_float<<<gridDim,blockDim>>>((float2*) d_partial_block,  texProp1, texProp2, it,  GK_sourcePosition[isource][0] , GK_sourcePosition[isource][1], GK_sourcePosition[isource][2],ip);
      checkCudaError();

      cudaMemcpy(h_partial_block , d_partial_block , alloc_size*2*4*4*2*sizeof(Float) , cudaMemcpyDeviceToHost); //-C.K. Copy device block into host block
      checkCudaError();

      //-C.K. Copy host block into corr buffer
      for(int pt = 0; pt < 2 ; pt++){
	for(int ga = 0 ; ga < 4 ; ga++){
	  for(int gap = 0; gap < 4 ; gap++){
	    for(int sv = 0; sv < SpVol ; sv++){
	      corr[ 0 + 2*sv + 2*SpVol*it ][pt][ip][ga][gap] = h_partial_block[ 0 + 2*sv + 2*SpVol*gap + 2*SpVol*4*ga + 2*SpVol*4*4*pt ];
	      corr[ 1 + 2*sv + 2*SpVol*it ][pt][ip][ga][gap] = h_partial_block[ 1 + 2*sv + 2*SpVol*gap + 2*SpVol*4*ga + 2*SpVol*4*4*pt ];
	    }}}
      }

    }//-ip

    free(h_partial_block);
    cudaFree(d_partial_block);
    checkCudaError();
  }
  else if(CorrSpace==MOMENTUM_SPACE){
    h_partial_block = (Float*)malloc(GK_Nmoms*2*4*4*gridDim.x*2*sizeof(Float));
    if(h_partial_block == NULL) errorQuda("contractBaryons_kernel: Cannot allocate host block.\n");

    cudaMalloc((void**)&d_partial_block, GK_Nmoms*2*4*4*gridDim.x*2 * sizeof(Float) );
    checkCudaError();

    Float *reduction =(Float*) calloc(GK_Nmoms*2*4*4*2,sizeof(Float));
  
    for(int ip = 0 ; ip < 10 ; ip++){
      contractBaryons_kernel_float<<<gridDim,blockDim>>>((float2*) d_partial_block,  texProp1, texProp2, it,  GK_sourcePosition[isource][0] , GK_sourcePosition[isource][1], GK_sourcePosition[isource][2],ip);
      checkCudaError();

      cudaMemcpy(h_partial_block , d_partial_block , GK_Nmoms*2*4*4*gridDim.x*2 * sizeof(Float) , cudaMemcpyDeviceToHost);
      checkCudaError();

      memset(reduction,0,GK_Nmoms*2*4*4*2*sizeof(Float));      
      for(int imom = 0 ; imom < GK_Nmoms ; imom++)
	for(int iu = 0 ; iu < 2 ; iu++)
	  for(int gamma = 0 ; gamma < 4 ; gamma++)
	    for(int gammap = 0 ; gammap < 4 ; gammap++)
	      for(int i =0 ; i < gridDim.x ; i++){
		reduction[imom*2*4*4*2 + iu*4*4*2 + gamma*4*2 + gammap*2 + 0] += h_partial_block[imom*2*4*4*gridDim.x*2 + iu*4*4*gridDim.x*2 + gamma*4*gridDim.x*2 + gammap*gridDim.x*2 + i*2 + 0];
		reduction[imom*2*4*4*2 + iu*4*4*2 + gamma*4*2 + gammap*2 + 1] += h_partial_block[imom*2*4*4*gridDim.x*2 + iu*4*4*gridDim.x*2 + gamma*4*gridDim.x*2 + gammap*gridDim.x*2 + i*2 + 1];
	      }
      
      for(int imom = 0 ; imom < GK_Nmoms ; imom++)
	for(int iu = 0 ; iu < 2 ; iu++)
	  for(int gamma = 0 ; gamma < 4 ; gamma++)
	    for(int gammap = 0 ; gammap < 4 ; gammap++){
	      corr[it*GK_Nmoms*2 + imom*2 + 0][iu][ip][gamma][gammap] = reduction[imom*2*4*4*2 + iu*4*4*2 + gamma*4*2 + gammap*2 + 0];
	      corr[it*GK_Nmoms*2 + imom*2 + 1][iu][ip][gamma][gammap] = reduction[imom*2*4*4*2 + iu*4*4*2 + gamma*4*2 + gammap*2 + 1];
	    }
    }//-ip

    free(h_partial_block);
    cudaFree(d_partial_block);
    checkCudaError();
    free(reduction);
  }//-CorrSpace else
  else errorQuda("contractBaryons_kernel: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

}

void quda::run_contractBaryons(cudaTextureObject_t texProp1,
			       cudaTextureObject_t texProp2,
			       void* corr,  int it, 
			       int isource, int precision, 
			       CORR_SPACE CorrSpace){

  if (CorrSpace==POSITION_SPACE) cudaFuncSetCacheConfig(contractBaryons_kernel_PosSpace_float,cudaFuncCachePreferShared);
  else if(CorrSpace==MOMENTUM_SPACE)  cudaFuncSetCacheConfig(contractBaryons_kernel_float         ,cudaFuncCachePreferShared);
  else errorQuda("run_contractBaryons: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");
  checkCudaError();

  if(precision == 4) contractBaryons_kernel<float2,float>(texProp1,texProp2,(float(*)[2][10][4][4]) corr,it, isource, CorrSpace);
  else if(precision == 8) errorQuda("Double precision in Baryon 2pt Contractions unsupported!!!\n");
  else errorQuda("run_contractBaryons: Precision %d not supported\n",precision);

}

void quda::run_rotateToPhysicalBase(void* inOut, int sign, int precision){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  if(precision == 4){
    rotateToPhysicalBase_kernel<float2><<<gridDim,blockDim>>>((float2*) inOut,sign);
  }
  else if(precision == 8){
    rotateToPhysicalBase_kernel<double2><<<gridDim,blockDim>>>((double2*) inOut,sign);
  }
  else{
    errorQuda("Precision not supported\n");
  }    
  checkCudaError();
}

void quda::run_castDoubleToFloat(void *out, void *in){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  castDoubleToFloat_kernel<<<gridDim,blockDim>>>((float2*) out, (double2*) in);
  checkCudaError();
}

void quda::run_castFloatToDouble(void *out, void *in){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  castFloatToDouble_kernel<<<gridDim,blockDim>>>((double2*) out, (float2*) in);
  checkCudaError();
}

void quda::run_conjugate_vector(void *inOut, int precision){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  if(precision == 4){
    conjugate_vector_kernel<float2><<<gridDim,blockDim>>>((float2*) inOut);
  }
  else if(precision == 8){
    conjugate_vector_kernel<double2><<<gridDim,blockDim>>>((double2*) inOut);
  }
  else{
    errorQuda("Precision not supported\n");
  }    
  checkCudaError();
}

void quda::run_apply_gamma5_vector(void *inOut, int precision){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  if(precision == 4){
    apply_gamma5_vector_kernel<float2><<<gridDim,blockDim>>>((float2*) inOut);
  }
  else if(precision == 8){
    apply_gamma5_vector_kernel<double2><<<gridDim,blockDim>>>((double2*) inOut);
  }
  else{
    errorQuda("Precision not supported\n");
  }    
  checkCudaError();
}

void quda::run_conjugate_propagator(void *inOut, int precision){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  if(precision == 4){
    conjugate_propagator_kernel<float2><<<gridDim,blockDim>>>((float2*) inOut);
  }
  else if(precision == 8){
    conjugate_propagator_kernel<double2><<<gridDim,blockDim>>>((double2*) inOut);
  }
  else{
    errorQuda("Precision not supported\n");
  }    
  checkCudaError();
}

void quda::run_apply_gamma5_propagator(void *inOut, int precision){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  if(precision == 4){
    apply_gamma5_propagator_kernel<float2><<<gridDim,blockDim>>>((float2*) inOut);
  }
  else if(precision == 8){
    apply_gamma5_propagator_kernel<double2><<<gridDim,blockDim>>>((double2*) inOut);
  }
  else{
    errorQuda("Precision not supported\n");
  }    
  checkCudaError();
}

template<typename Float>
static void seqSourceFixSinkPart1_kernel(void* out, int timeslice, 
					 cudaTextureObject_t tex1, 
					 cudaTextureObject_t tex2, 
					 int c_nu, int c_c2, 
					 WHICHPROJECTOR PID, 
					 WHICHPARTICLE PARTICLE ){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume/GK_localL[3] + blockDim.x -1)/blockDim.x , 1 , 1);

  if( typeid(Float) == typeid(float) )
    seqSourceFixSinkPart1_kernel_float<<<gridDim,blockDim>>>((float2*) out,  timeslice,  tex1, tex2, c_nu,  c_c2,  PID, PARTICLE);
  else if(typeid(Float) == typeid(double))
    seqSourceFixSinkPart1_kernel_double<<<gridDim,blockDim>>>((double2*) out, timeslice, tex1, tex2, c_nu,  c_c2,  PID, PARTICLE);
  
  checkCudaError();
}

template<typename Float>
static void seqSourceFixSinkPart2_kernel(void* out, int timeslice, 
					 cudaTextureObject_t tex, 
					 int c_nu, int c_c2, 
					 WHICHPROJECTOR PID, 
					 WHICHPARTICLE PARTICLE ){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume/GK_localL[3] + blockDim.x -1)/blockDim.x , 1 , 1);

  if( typeid(Float) == typeid(float) )
    seqSourceFixSinkPart2_kernel_float<<<gridDim,blockDim>>>((float2*) out,  timeslice,  tex, c_nu,  c_c2,  PID, PARTICLE);
  else if(typeid(Float) == typeid(double))
    seqSourceFixSinkPart2_kernel_double<<<gridDim,blockDim>>>((double2*) out, timeslice, tex, c_nu,  c_c2,  PID, PARTICLE);
  
  checkCudaError();
}

void quda::run_seqSourceFixSinkPart1(void* out, int timeslice, 
				     cudaTextureObject_t tex1, 
				     cudaTextureObject_t tex2, 
				     int c_nu, int c_c2, 
				     WHICHPROJECTOR PID, 
				     WHICHPARTICLE PARTICLE, 
				     int precision){
  if(precision == 4){
    seqSourceFixSinkPart1_kernel<float>(out,  timeslice,  tex1, tex2, c_nu,  c_c2,  PID, PARTICLE);
  }
  else if(precision == 8){
    seqSourceFixSinkPart1_kernel<double>(out,  timeslice,  tex1, tex2, c_nu,  c_c2,  PID, PARTICLE);
  }
  else{
    errorQuda("Precision not supported\n");
  }
}

void quda::run_seqSourceFixSinkPart2(void* out, int timeslice, 
				     cudaTextureObject_t tex, 
				     int c_nu, int c_c2, 
				     WHICHPROJECTOR PID, 
				     WHICHPARTICLE PARTICLE, 
				     int precision){
  if(precision == 4){
    seqSourceFixSinkPart2_kernel<float>(out,  timeslice,  tex, c_nu,  c_c2,  PID, PARTICLE);
  }
  else if(precision == 8){
    seqSourceFixSinkPart2_kernel<double>(out,  timeslice, tex, c_nu,  c_c2,  PID, PARTICLE);
  }
  else{
    errorQuda("Precision not supported\n");
  }
}


template<typename Float2,typename Float>
static void fixSinkContractions_kernel(void* corrThp_local, 
				       void* corrThp_noether, 
				       void* corrThp_oneD, 
				       cudaTextureObject_t fwdTex, 
				       cudaTextureObject_t seqTex, 
				       cudaTextureObject_t gaugeTex,
				       WHICHPARTICLE PARTICLE, 
				       int partflag, int itime, 
				       int isource, CORR_SPACE CorrSpace){

  int SpVol = GK_localVolume/GK_localL[3];
  int lV = GK_localVolume;

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume/GK_localL[3] + blockDim.x -1)/blockDim.x , 1 , 1); // spawn threads only for the spatial volume

  Float *h_partial_block = NULL;
  Float *d_partial_block = NULL;

  if(CorrSpace==POSITION_SPACE){
    size_t alloc_buf;
    size_t copy_buf;

    //- Ultra-local operators
    alloc_buf = blockDim.x * gridDim.x * 16 * 2 * sizeof(Float);
    copy_buf  = SpVol * 16 * 2 * sizeof(Float);

    cudaMalloc((void**)&d_partial_block, alloc_buf);
    checkCudaError();
    cudaMemset(d_partial_block, 0, alloc_buf);
    checkCudaError();

    if( typeid(Float2) == typeid(float2) )
      fixSinkContractions_local_kernel_PosSpace_float<<<gridDim,blockDim>>> ((float2*) d_partial_block, fwdTex, seqTex, PARTICLE, partflag, itime,
									     GK_sourcePosition[isource][0], 
									     GK_sourcePosition[isource][1], 
									     GK_sourcePosition[isource][2]);
    else if( typeid(Float2) == typeid(double2) )
      fixSinkContractions_local_kernel_PosSpace_double<<<gridDim,blockDim>>>((double2*) d_partial_block, fwdTex, seqTex, PARTICLE, partflag, itime,
									     GK_sourcePosition[isource][0], 
									     GK_sourcePosition[isource][1], 
									     GK_sourcePosition[isource][2]);

    //-C.K. Copy device block into corrThp_local
    cudaMemcpy(&(((Float*)corrThp_local)[2*16*SpVol*itime]) , d_partial_block , copy_buf , cudaMemcpyDeviceToHost);
    checkCudaError();
    //----------------------------------------------------------------------

    //- One-derivative operators
    for(int dir = 0 ; dir < 4 ; dir++){
      cudaMemset(d_partial_block, 0, alloc_buf);
      checkCudaError();

      if( typeid(Float2) == typeid(float2) )
	fixSinkContractions_oneD_kernel_PosSpace_float<<<gridDim,blockDim>>> ((float2*) d_partial_block, fwdTex, seqTex, gaugeTex, 
									      PARTICLE, partflag, itime, dir,
									      GK_sourcePosition[isource][0], 
									      GK_sourcePosition[isource][1], 
									      GK_sourcePosition[isource][2]);
      else if( typeid(Float2) == typeid(double2) )
	fixSinkContractions_oneD_kernel_PosSpace_double<<<gridDim,blockDim>>>((double2*) d_partial_block, fwdTex, seqTex, gaugeTex, 
									      PARTICLE, partflag, itime, dir,
									      GK_sourcePosition[isource][0], 
									      GK_sourcePosition[isource][1], 
									      GK_sourcePosition[isource][2]);

      //-C.K. Copy device block into corrThp_oneD for each dir
      cudaMemcpy(&(((Float*)corrThp_oneD)[2*16*SpVol*itime + 2*16*lV*dir]), d_partial_block , copy_buf , cudaMemcpyDeviceToHost); 
      checkCudaError();
    }//-dir
    //----------------------------------------------------------------------

    //- Noether, conserved current
    //- it's better to reallocate the device block buffer here
    cudaFree(d_partial_block);
    checkCudaError();
    d_partial_block = NULL;
    alloc_buf = blockDim.x * gridDim.x * 4 * 2 * sizeof(Float);
    copy_buf  = SpVol * 4 * 2 * sizeof(Float);

    cudaMalloc((void**)&d_partial_block, alloc_buf);
    checkCudaError();
    cudaMemset(d_partial_block, 0, alloc_buf);
    checkCudaError();

    if( typeid(Float2) == typeid(float2) )
      fixSinkContractions_noether_kernel_PosSpace_float<<<gridDim,blockDim>>> ((float2*) d_partial_block, fwdTex, seqTex, gaugeTex, 
									       PARTICLE, partflag, itime,
									       GK_sourcePosition[isource][0], 
									       GK_sourcePosition[isource][1], 
									       GK_sourcePosition[isource][2]);
    else if( typeid(Float2) == typeid(double2) )
      fixSinkContractions_noether_kernel_PosSpace_double<<<gridDim,blockDim>>>((double2*) d_partial_block, fwdTex, seqTex, gaugeTex, 
									       PARTICLE, partflag, itime,
									       GK_sourcePosition[isource][0], 
									       GK_sourcePosition[isource][1], 
									       GK_sourcePosition[isource][2]);

    //-C.K. Copy device block to corrThp_noether
    cudaMemcpy(&(((Float*)corrThp_noether)[2*4*SpVol*itime]) , d_partial_block , copy_buf , cudaMemcpyDeviceToHost); 
    checkCudaError();

    cudaFree(d_partial_block);
    checkCudaError();
  }
  else if(CorrSpace==MOMENTUM_SPACE){
    h_partial_block = (Float*)malloc(GK_Nmoms*16*gridDim.x*2*sizeof(Float));
    if(h_partial_block == NULL) errorQuda("fixSinkContractions_kernel: Cannot allocate host block.\n");
    cudaMalloc((void**)&d_partial_block, GK_Nmoms*16*gridDim.x*2 * sizeof(Float) );
    checkCudaError();
    Float *reduction =(Float*) calloc(GK_Nmoms*16*2,sizeof(Float));
  
    //- Ultra-local operators
    if( typeid(Float2) == typeid(float2) )
      fixSinkContractions_local_kernel_float<<<gridDim,blockDim>>>((float2*) d_partial_block, fwdTex, seqTex, PARTICLE, 
								   partflag, itime, 
								   GK_sourcePosition[isource][0], 
								   GK_sourcePosition[isource][1], 
								   GK_sourcePosition[isource][2]);
    else if( typeid(Float2) == typeid(double2) )
      fixSinkContractions_local_kernel_double<<<gridDim,blockDim>>>((double2*) d_partial_block, fwdTex, seqTex, PARTICLE, 
								    partflag, itime, 
								    GK_sourcePosition[isource][0], 
								    GK_sourcePosition[isource][1], 
								    GK_sourcePosition[isource][2]);

    cudaMemcpy(h_partial_block , d_partial_block , GK_Nmoms*16*gridDim.x*2 * sizeof(Float) , cudaMemcpyDeviceToHost);
    checkCudaError();
    memset(reduction,0,GK_Nmoms*16*2*sizeof(Float));

    for(int imom = 0 ; imom < GK_Nmoms ; imom++)
      for(int iop = 0 ; iop < 16 ; iop++)
	for(int i =0 ; i < gridDim.x ; i++){
	  reduction[imom*16*2 + iop*2 + 0] += h_partial_block[imom*16*gridDim.x*2 + iop*gridDim.x*2 + i*2 + 0];
	  reduction[imom*16*2 + iop*2 + 1] += h_partial_block[imom*16*gridDim.x*2 + iop*gridDim.x*2 + i*2 + 1];
	}

    for(int imom = 0 ; imom < GK_Nmoms ; imom++)
      for(int iop = 0 ; iop < 16 ; iop++)
	for(int i =0 ; i < gridDim.x ; i++){
	  ((Float*) corrThp_local)[itime*GK_Nmoms*16*2 + imom*16*2 + iop*2 + 0] = reduction[imom*16*2 + iop*2 + 0];
	  ((Float*) corrThp_local)[itime*GK_Nmoms*16*2 + imom*16*2 + iop*2 + 1] = reduction[imom*16*2 + iop*2 + 1];
	}
    //---------------------------------------------------------------

    //- Noether, conserved current
    if( typeid(Float2) == typeid(float2) )
      fixSinkContractions_noether_kernel_float<<<gridDim,blockDim>>>((float2*) d_partial_block, fwdTex, seqTex, gaugeTex, 
								     PARTICLE, partflag, itime, 
								     GK_sourcePosition[isource][0], 
								     GK_sourcePosition[isource][1], 
								     GK_sourcePosition[isource][2]);
    else if( typeid(Float2) == typeid(double2) )
      fixSinkContractions_noether_kernel_double<<<gridDim,blockDim>>>((double2*) d_partial_block, fwdTex, seqTex, gaugeTex, 
								      PARTICLE, partflag, itime, 
								      GK_sourcePosition[isource][0], 
								      GK_sourcePosition[isource][1], 
								      GK_sourcePosition[isource][2]);

    cudaMemcpy(h_partial_block , d_partial_block , GK_Nmoms*4*gridDim.x*2 * sizeof(Float) , cudaMemcpyDeviceToHost);
    checkCudaError();
    memset(reduction,0,GK_Nmoms*4*2*sizeof(Float));

    for(int imom = 0 ; imom < GK_Nmoms ; imom++)
      for(int dir = 0 ; dir < 4 ; dir++)
	for(int i =0 ; i < gridDim.x ; i++){
	  reduction[imom*4*2 + dir*2 + 0] += h_partial_block[imom*4*gridDim.x*2 + dir*gridDim.x*2 + i*2 + 0];
	  reduction[imom*4*2 + dir*2 + 1] += h_partial_block[imom*4*gridDim.x*2 + dir*gridDim.x*2 + i*2 + 1];
	}
  
    for(int imom = 0 ; imom < GK_Nmoms ; imom++)
      for(int dir = 0 ; dir < 4 ; dir++)
	for(int i =0 ; i < gridDim.x ; i++){
	  ((Float*) corrThp_noether)[itime*GK_Nmoms*4*2 + imom*4*2 + dir*2 + 0] = reduction[imom*4*2 + dir*2 + 0];
	  ((Float*) corrThp_noether)[itime*GK_Nmoms*4*2 + imom*4*2 + dir*2 + 1] = reduction[imom*4*2 + dir*2 + 1];
	}
    //---------------------------------------------------------------

    //- One-derivative operators
    for(int dir = 0 ; dir < 4 ; dir++){
      if( typeid(Float2) == typeid(float2) )
	fixSinkContractions_oneD_kernel_float<<<gridDim,blockDim>>>((float2*) d_partial_block, fwdTex, seqTex, gaugeTex, 
								    PARTICLE, partflag, itime, dir, 
								    GK_sourcePosition[isource][0], 
								    GK_sourcePosition[isource][1], 
								    GK_sourcePosition[isource][2]);
      else if( typeid(Float2) == typeid(double2) )
	fixSinkContractions_oneD_kernel_double<<<gridDim,blockDim>>>((double2*) d_partial_block, fwdTex, seqTex, gaugeTex, 
								     PARTICLE, partflag, itime, dir, 
								     GK_sourcePosition[isource][0], 
								     GK_sourcePosition[isource][1], 
								     GK_sourcePosition[isource][2]);
      
      cudaMemcpy(h_partial_block , d_partial_block , GK_Nmoms*16*gridDim.x*2 * sizeof(Float) , cudaMemcpyDeviceToHost);
      checkCudaError();
      memset(reduction,0,GK_Nmoms*16*2*sizeof(Float));
    
      for(int imom = 0 ; imom < GK_Nmoms ; imom++)
	for(int iop = 0 ; iop < 16 ; iop++)
	  for(int i =0 ; i < gridDim.x ; i++){
	    reduction[imom*16*2 + iop*2 + 0] += h_partial_block[imom*16*gridDim.x*2 + iop*gridDim.x*2 + i*2 + 0];
	    reduction[imom*16*2 + iop*2 + 1] += h_partial_block[imom*16*gridDim.x*2 + iop*gridDim.x*2 + i*2 + 1];
	  }
    
      for(int imom = 0 ; imom < GK_Nmoms ; imom++)
	for(int iop = 0 ; iop < 16 ; iop++)
	  for(int i =0 ; i < gridDim.x ; i++){
	    ((Float*) corrThp_oneD)[itime*GK_Nmoms*4*16*2 + imom*4*16*2 + dir*16*2 + iop*2 + 0] = reduction[imom*16*2 + iop*2 + 0];
	    ((Float*) corrThp_oneD)[itime*GK_Nmoms*4*16*2 + imom*4*16*2 + dir*16*2 + iop*2 + 1] = reduction[imom*16*2 + iop*2 + 1];
	  }
    }//-dir
    //---------------------------------------------------------------

    free(h_partial_block);
    cudaFree(d_partial_block);
    checkCudaError();
    free(reduction);
  }
  else errorQuda("fixSinkContractions_kernel: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

}


void quda::run_fixSinkContractions(void* corrThp_local, void* corrThp_noether, 
				   void* corrThp_oneD, cudaTextureObject_t fwdTex, 
				   cudaTextureObject_t seqTex, cudaTextureObject_t gaugeTex,
				   WHICHPARTICLE PARTICLE, int partflag, int it, 
				   int isource, int precision, CORR_SPACE CorrSpace){
  
  if(precision == 4) 
    fixSinkContractions_kernel<float2,float>  (corrThp_local, corrThp_noether, 
					       corrThp_oneD, fwdTex, seqTex, 
					       gaugeTex, PARTICLE, partflag, 
					       it, isource, CorrSpace);
  else if(precision == 8) 
    fixSinkContractions_kernel<double2,double>(corrThp_local, corrThp_noether, 
					       corrThp_oneD, fwdTex, seqTex, 
					       gaugeTex, PARTICLE, partflag, 
					       it, isource, CorrSpace);
  else  errorQuda("run_fixSinkContractions: Precision %d not supported\n",precision);
  
}
