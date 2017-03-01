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


//--------------------------//
// class QKXTM_Gauge_Kepler //
//--------------------------//

template<typename Float>
QKXTM_Gauge_Kepler<Float>::QKXTM_Gauge_Kepler(ALLOCATION_FLAG alloc_flag, 
					      CLASS_ENUM classT): 
  QKXTM_Field_Kepler<Float>(alloc_flag, classT){ ; }

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::packGauge(void **gauge){

  double **p_gauge = (double**) gauge;
  
  for(int dir = 0 ; dir < GK_nDim ; dir++)
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++)
	  for(int part = 0 ; part < 2 ; part++){
	    CC::h_elem[dir*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       iv*2 + part] = 
	      (Float) p_gauge[dir][iv*GK_nColor*GK_nColor*2 + 
				   c1*GK_nColor*2 + c2*2 + part];
	  }
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::packGaugeToBackup(void **gauge){
  double **p_gauge = (double**) gauge;
  if(CC::h_elem_backup != NULL){
    for(int dir = 0 ; dir < GK_nDim ; dir++)
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++)
    for(int part = 0 ; part < 2 ; part++){
      CC::h_elem_backup[dir*GK_nColor*GK_nColor*GK_localVolume*2 + 
			c1*GK_nColor*GK_localVolume*2 + 
			c2*GK_localVolume*2 + 
			iv*2 + part] = 
	(Float) p_gauge[dir][iv*GK_nColor*GK_nColor*2 + 
			     c1*GK_nColor*2 + 
			     c2*2 + part];
    }
  }
  else{
    errorQuda("Error you can call this method only if you allocate memory for h_elem_backup");
  }

}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::justDownloadGauge(){
  cudaMemcpy(CC::h_elem,CC::d_elem,CC::bytes_total_length, 
	     cudaMemcpyDeviceToHost);
  checkCudaError();
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::loadGauge(){
  cudaMemcpy(CC::d_elem,CC::h_elem,CC::bytes_total_length, 
	     cudaMemcpyHostToDevice );
  checkCudaError();
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::loadGaugeFromBackup(){
  if(CC::h_elem_backup != NULL){
    cudaMemcpy(CC::d_elem,CC::h_elem_backup, CC::bytes_total_length, 
	       cudaMemcpyHostToDevice );
    checkCudaError();
  }
  else{
    errorQuda("Error you can call this method only if you allocate memory for h_elem_backup");
  }
}

// gpu collect ghost and send it to host
template<typename Float>
void QKXTM_Gauge_Kepler<Float>::ghostToHost(){   

  // direction x 
  if( GK_localL[0] < GK_totalL[0]){
    int position;
    // number of blocks that we need
    int height = GK_localL[1] * GK_localL[2] * GK_localL[3];
    size_t width = 2*sizeof(Float);
    size_t spitch = GK_localL[0]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;

    position = GK_localL[0]-1;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);
	  h_elem_offset = (CC::h_elem + 
			   GK_minusGhost[0]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[0]*2 + 
			   c1*GK_nColor*GK_surface3D[0]*2 + 
			   c2*GK_surface3D[0]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
    // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_plusGhost[0]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[0]*2 + 
			   c1*GK_nColor*GK_surface3D[0]*2 + 
			   c2*GK_surface3D[0]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  // direction y 
  if( GK_localL[1] < GK_totalL[1]){
    int position;
    // number of blocks that we need
    int height = GK_localL[2] * GK_localL[3];
    size_t width = GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[1]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*(GK_localL[1]-1);
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_minusGhost[1]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[1]*2 + 
			   c1*GK_nColor*GK_surface3D[1]*2 + 
			   c2*GK_surface3D[1]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
    // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_plusGhost[1]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[1]*2 + 
			   c1*GK_nColor*GK_surface3D[1]*2 + 
			   c2*GK_surface3D[1]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  
  // direction z 
  if( GK_localL[2] < GK_totalL[2]){

    int position;
    // number of blocks that we need
    int height = GK_localL[3]; 
    size_t width = GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[2]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*(GK_localL[2]-1);
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_minusGhost[2]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[2]*2 + 
			   c1*GK_nColor*GK_surface3D[2]*2 + 
			   c2*GK_surface3D[2]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
    // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_plusGhost[2]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[2]*2 + 
			   c1*GK_nColor*GK_surface3D[2]*2 + 
			   c2*GK_surface3D[2]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  // direction t 
  if( GK_localL[3] < GK_totalL[3]){
    int position;
    int height = GK_nDim*GK_nColor*GK_nColor;
    size_t width = GK_localL[2]*GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[3]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*GK_localL[2]*(GK_localL[3]-1);
    d_elem_offset=CC::d_elem+position*2;
    h_elem_offset=CC::h_elem+GK_minusGhost[3]*GK_nDim*GK_nColor*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,
		 width,height,cudaMemcpyDeviceToHost);
    // set minus points to plus area
    position = 0;
    d_elem_offset=CC::d_elem+position*2;
    h_elem_offset=CC::h_elem+GK_plusGhost[3]*GK_nDim*GK_nColor*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,
		 width,height,cudaMemcpyDeviceToHost);
  }
  checkCudaError();
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::cpuExchangeGhost(){
  if( comm_size() > 1 ){
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_from_back[4];
    MsgHandle *mh_from_fwd[4];
    MsgHandle *mh_send_back[4];

    Float *pointer_receive = NULL;
    Float *pointer_send = NULL;

    for(int idim = 0 ; idim < GK_nDim; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	size_t nbytes = 
	  GK_surface3D[idim]*GK_nColor*GK_nColor*GK_nDim*2*sizeof(Float);
	// send to plus
	pointer_receive = CC::h_ext_ghost + (GK_minusGhost[idim]-GK_localVolume)*GK_nColor*GK_nColor*GK_nDim*2;
	pointer_send = CC::h_elem + GK_minusGhost[idim]*GK_nColor*GK_nColor*GK_nDim*2;
	mh_from_back[idim] = comm_declare_receive_relative(pointer_receive,idim,-1,nbytes);
	mh_send_fwd[idim] = comm_declare_send_relative(pointer_send,idim,1,nbytes);
	comm_start(mh_from_back[idim]);
	comm_start(mh_send_fwd[idim]);
	comm_wait(mh_send_fwd[idim]);
	comm_wait(mh_from_back[idim]);
		
	// send to minus
	pointer_receive = CC::h_ext_ghost + (GK_plusGhost[idim]-GK_localVolume)*GK_nColor*GK_nColor*GK_nDim*2;
	pointer_send = CC::h_elem + GK_plusGhost[idim]*GK_nColor*GK_nColor*GK_nDim*2;
	mh_from_fwd[idim] = comm_declare_receive_relative(pointer_receive,idim,1,nbytes);
	mh_send_back[idim] = comm_declare_send_relative(pointer_send,idim,-1,nbytes);
	comm_start(mh_from_fwd[idim]);
	comm_start(mh_send_back[idim]);
	comm_wait(mh_send_back[idim]);
	comm_wait(mh_from_fwd[idim]);
		
	pointer_receive = NULL;
	pointer_send = NULL;

      }
    }

    for(int idim = 0 ; idim < GK_nDim ; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	comm_free(mh_send_fwd[idim]);
	comm_free(mh_from_fwd[idim]);
	comm_free(mh_send_back[idim]);
	comm_free(mh_from_back[idim]);
      }
    }
    
  }
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::ghostToDevice(){
  if(comm_size() > 1){
    Float *host = CC::h_ext_ghost;
    Float *device = CC::d_elem+GK_localVolume*GK_nColor*GK_nColor*GK_nDim*2;
    cudaMemcpy(device,host,CC::bytes_ghost_length,cudaMemcpyHostToDevice);
    checkCudaError();
  }
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::calculatePlaq(){
  cudaTextureObject_t tex;

  ghostToHost();
  cpuExchangeGhost();
  ghostToDevice();
  CC::createTexObject(&tex);
  run_calculatePlaq_kernel(tex, sizeof(Float));
  CC::destroyTexObject(tex);

}
