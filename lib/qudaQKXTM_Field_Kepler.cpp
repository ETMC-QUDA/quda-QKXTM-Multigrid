//-C.K. Interface for performing the loops and the correlation function 
//contractions, including the exact deflation using ARPACK
//#include <qudaQKXTM.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <qudaQKXTM_Kepler.h>
#include <qudaQKXTM_Kepler_utils.h>
#include <errno.h>
#include <mpi.h>
#include <limits>
//#include <mkl.h> //QXKTM: FIXME
#include <cblas.h>
#include <common.h>
#include <omp.h>
#include <hdf5.h>
 
#define PI 3.141592653589793
 
//using namespace quda;
extern Topology *default_topo;
 
/* Block for global variables */
extern float GK_deviceMemory;
extern int GK_nColor;
extern int GK_nSpin;
extern int GK_nDim;
extern int GK_strideFull;
extern double GK_alphaAPE;
extern double GK_alphaGauss;
extern int GK_localVolume;
extern int GK_totalVolume;
extern int GK_nsmearAPE;
extern int GK_nsmearGauss;
extern bool GK_dimBreak[QUDAQKXTM_DIM];
extern int GK_localL[QUDAQKXTM_DIM];
extern int GK_totalL[QUDAQKXTM_DIM];
extern int GK_nProc[QUDAQKXTM_DIM];
extern int GK_plusGhost[QUDAQKXTM_DIM];
extern int GK_minusGhost[QUDAQKXTM_DIM];
extern int GK_surface3D[QUDAQKXTM_DIM];
extern bool GK_init_qudaQKXTM_Kepler_flag;
extern int GK_Nsources;
extern int GK_sourcePosition[MAX_NSOURCES][QUDAQKXTM_DIM];
extern int GK_Nmoms;
extern short int GK_moms[MAX_NMOMENTA][3];
// for mpi use global variables
extern MPI_Group GK_fullGroup , GK_spaceGroup , GK_timeGroup;
extern MPI_Comm GK_spaceComm , GK_timeComm;
extern int GK_localRank;
extern int GK_localSize;
extern int GK_timeRank;
extern int GK_timeSize;

#define CC QKXTM_Field_Kepler<Float>
#define DEVICE_MEMORY_REPORT
#define CMPLX_FLOAT std::complex<Float>

//--------------------------//
// class QKXTM_Field_Kepler //
//--------------------------//

// This is is a class which allocates memory on either the
// the device, or host, or both for the structures:
// Field: one complex number per spacetime point.
// Gauge: one SU(3) link variable per spacetime point X spacetime dimension.
// Vector: 12 complex numbers (3colour x 4spin) per spacetime point.
// Vector3D: as above, but only defined on a single timeslice.
// Propagator: 12x12 complex matrix per sink position 
// (usually a whole spacetime volume.)
// Propagtor3D: as above, but with sinks only at one timeslice.

template<typename Float>
QKXTM_Field_Kepler<Float>::QKXTM_Field_Kepler(ALLOCATION_FLAG alloc_flag, 
					      CLASS_ENUM classT):
  h_elem(NULL) , d_elem(NULL) , h_ext_ghost(NULL) , h_elem_backup(NULL) , 
  isAllocHost(false) , isAllocDevice(false), isAllocHostBackup(false)
{
  if(GK_init_qudaQKXTM_Kepler_flag == false) 
    errorQuda("You must initialize init_qudaQKXTM_Kepler first");

  switch(classT){
  case FIELD:
    field_length = 1;
    total_length = GK_localVolume;
    break;
  case GAUGE:
    field_length = GK_nDim * GK_nColor * GK_nSpin;
    total_length = GK_localVolume;
    break;
  case VECTOR:
    field_length = GK_nSpin * GK_nColor;
    total_length = GK_localVolume;
    break;
  case PROPAGATOR:
    field_length = GK_nSpin * GK_nColor * GK_nSpin * GK_nColor;
    total_length = GK_localVolume;
    break;
  case PROPAGATOR3D:
    field_length = GK_nSpin * GK_nColor * GK_nSpin * GK_nColor;
    total_length = GK_localVolume/GK_localL[3];
    break;
  case VECTOR3D:
    field_length = GK_nSpin * GK_nColor;
    total_length = GK_localVolume/GK_localL[3];
    break;
  }

  ghost_length = 0;

  for(int i = 0 ; i < GK_nDim ; i++)
    ghost_length += 2*GK_surface3D[i];

  total_plus_ghost_length = total_length + ghost_length;

  bytes_total_length = total_length*field_length*2*sizeof(Float);
  bytes_ghost_length = ghost_length*field_length*2*sizeof(Float);
  bytes_total_plus_ghost_length = total_plus_ghost_length*field_length*2*sizeof(Float);

  if( alloc_flag == BOTH ){
    create_host();
    create_device();
  }
  else if (alloc_flag == HOST){
    create_host();
  }
  else if (alloc_flag == DEVICE){
    create_device();
  }
  else if (alloc_flag == BOTH_EXTRA){
    create_host();
    create_host_backup();
    create_device();    
  }

}

//Destructor
template<typename Float>
QKXTM_Field_Kepler<Float>::~QKXTM_Field_Kepler(){
  if(h_elem != NULL) destroy_host();
  if(h_elem_backup != NULL) destroy_host_backup();
  if(d_elem != NULL) destroy_device();
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::create_host(){
  h_elem = (Float*) malloc(bytes_total_plus_ghost_length);
  h_ext_ghost = (Float*) malloc(bytes_ghost_length);
  if(h_elem == NULL || 
     h_ext_ghost == NULL) errorQuda("Error with allocation host memory");
  isAllocHost = true;
  zero_host();
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::create_host_backup(){
  h_elem_backup = (Float*) malloc(bytes_total_plus_ghost_length);
  if(h_elem_backup == NULL) errorQuda("Error with allocation host memory");
  isAllocHostBackup = true;
  zero_host_backup();
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::create_device(){
  cudaMalloc((void**)&d_elem,bytes_total_plus_ghost_length);
  checkCudaError();
#ifdef DEVICE_MEMORY_REPORT
  // device memory in MB
  GK_deviceMemory += bytes_total_length/(1024.*1024.);          
  printfQuda("Device memory in use is %f MB A QKXTM \n",GK_deviceMemory);
#endif
  isAllocDevice = true;
  zero_device();
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::destroy_host(){
  free(h_elem);
  free(h_ext_ghost);
  h_elem=NULL;
  h_ext_ghost = NULL;
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::destroy_host_backup(){
  free(h_elem_backup);
  h_elem=NULL;
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::destroy_device(){
  cudaFree(d_elem);
  checkCudaError();
  d_elem = NULL;
#ifdef DEVICE_MEMORY_REPORT
  GK_deviceMemory -= bytes_total_length/(1024.*1024.);
  printfQuda("Device memory in used is %f MB D \n",GK_deviceMemory);
#endif
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::zero_host(){
  memset(h_elem,0,bytes_total_plus_ghost_length);
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::zero_host_backup(){
  memset(h_elem_backup,0,bytes_total_plus_ghost_length);
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::zero_device(){
  cudaMemset(d_elem,0,bytes_total_plus_ghost_length);
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::createTexObject(cudaTextureObject_t *tex){
  cudaChannelFormatDesc desc;
  memset(&desc, 0, sizeof(cudaChannelFormatDesc));
  int precision = CC::Precision();
  if(precision == 4) desc.f = cudaChannelFormatKindFloat;
  else desc.f = cudaChannelFormatKindSigned;

  if(precision == 4){
    desc.x = 8*precision;
    desc.y = 8*precision;
    desc.z = 0;
    desc.w = 0;
  }
  else if(precision == 8){
    desc.x = 8*precision/2;
    desc.y = 8*precision/2;
    desc.z = 8*precision/2;
    desc.w = 8*precision/2;
  }

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = d_elem;
  resDesc.res.linear.desc = desc;
  resDesc.res.linear.sizeInBytes = bytes_total_plus_ghost_length;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL);
  checkCudaError();
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::destroyTexObject(cudaTextureObject_t tex){
  cudaDestroyTextureObject(tex);
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::printInfo(){
  printfQuda("This object has precision %d\n",Precision());
  printfQuda("This object needs %f Mb\n",
	     bytes_total_plus_ghost_length/(1024.*1024.));
  printfQuda("The flag for the host allocation is %d\n",(int) isAllocHost);
  printfQuda("The flag for the device allocation is %d\n",(int) isAllocDevice);
}
