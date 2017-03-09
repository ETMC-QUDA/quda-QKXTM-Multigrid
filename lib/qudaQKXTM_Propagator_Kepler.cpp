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
//#include <mkl.h>
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

//-------------------------------//
// class QKXTM_Propagator_Kepler //
//-------------------------------//

template<typename Float>
QKXTM_Propagator_Kepler<Float>::
QKXTM_Propagator_Kepler(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT): 
  QKXTM_Field_Kepler<Float>(alloc_flag, classT){;}

template <typename Float>
void QKXTM_Propagator_Kepler<Float>::
absorbVectorToHost(QKXTM_Vector_Kepler<Float> &vec, int nu, int c2){
  Float *pointProp_host;
  Float *pointVec_dev;
  for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++){
      pointProp_host = (CC::h_elem + 
			mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
			nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
			c1*GK_nColor*GK_localVolume*2 + 
			c2*GK_localVolume*2);
      pointVec_dev = vec.D_elem() + mu*GK_nColor*GK_localVolume*2 + c1*GK_localVolume*2;
      cudaMemcpy(pointProp_host,pointVec_dev,GK_localVolume*2*sizeof(Float),cudaMemcpyDeviceToHost); 
    }
  checkCudaError();
}
 
template <typename Float>
void QKXTM_Propagator_Kepler<Float>::absorbVectorToDevice(QKXTM_Vector_Kepler<Float> &vec, int nu, int c2){
  Float *pointProp_dev;
  Float *pointVec_dev;
  for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++){
      pointProp_dev = (CC::d_elem + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2);
      pointVec_dev = vec.D_elem() + mu*GK_nColor*GK_localVolume*2 + c1*GK_localVolume*2;
      cudaMemcpy(pointProp_dev,pointVec_dev,GK_localVolume*2*sizeof(Float),
		 cudaMemcpyDeviceToDevice); 
    }
  checkCudaError();
}

template<typename Float>
void QKXTM_Propagator_Kepler<Float>::rotateToPhysicalBase_device(int sign){
  if( (sign != +1) && (sign != -1) ) errorQuda("The sign can be only +-1\n");
  run_rotateToPhysicalBase((void*) CC::d_elem, sign , sizeof(Float));
}

//QKXTM: DMH Rewrote some parts of this function to conform to new
// QUDA standards. Eg, assigning vaules to complex variable:
// var.real() = 1.0; is changed to var.real(1.0);
template <typename Float>
void QKXTM_Propagator_Kepler<Float>::rotateToPhysicalBase_host(int sign_int){
  if( (sign_int != +1) && (sign_int != -1) ) 
    errorQuda("The sign can be only +-1\n");
  
  std::complex<Float> sign;
  sign.real(1.0*sign_int);
  sign.imag(0.0);

  std::complex<Float> coeff;
  coeff.real(0.5);
  coeff.imag(0.0);
  
  std::complex<Float> P[4][4];
  std::complex<Float> PT[4][4];
  std::complex<Float> imag_unit;
  //imag_unit.real() = 0.;
  //imag_unit.imag() = 1.;
  imag_unit.real(0.0);
  imag_unit.imag(1.0);

  for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	      
	for(int mu = 0 ; mu < 4 ; mu++)
	  for(int nu = 0 ; nu < 4 ; nu++){
	    //P[mu][nu].real() = CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + nu*GK_nColor*GK_nColor*GK_localVolume + c1*GK_nColor*GK_localVolume + c2*GK_localVolume + iv)*2 + 0];
	    //P[mu][nu].imag() = CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + nu*GK_nColor*GK_nColor*GK_localVolume + c1*GK_nColor*GK_localVolume + c2*GK_localVolume + iv)*2 + 1]
	    P[mu][nu].real(CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + 
				       nu*GK_nColor*GK_nColor*GK_localVolume + 
				       c1*GK_nColor*GK_localVolume + 
				       c2*GK_localVolume + iv)*2 + 0]);
	    P[mu][nu].imag(CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + 
				       nu*GK_nColor*GK_nColor*GK_localVolume + 
				       c1*GK_nColor*GK_localVolume + 
				       c2*GK_localVolume + iv)*2 + 1]);
	  }
	
	PT[0][0] = coeff * (P[0][0] + sign * ( imag_unit * P[0][2] ) + sign * ( imag_unit * P[2][0] ) - P[2][2]);
	PT[0][1] = coeff * (P[0][1] + sign * ( imag_unit * P[0][3] ) + sign * ( imag_unit * P[2][1] ) - P[2][3]);
	PT[0][2] = coeff * (sign * ( imag_unit * P[0][0] ) + P[0][2] - P[2][0] + sign * ( imag_unit * P[2][2] ));
	PT[0][3] = coeff * (sign * ( imag_unit * P[0][1] ) + P[0][3] - P[2][1] + sign * ( imag_unit * P[2][3] ));
	
	PT[1][0] = coeff * (P[1][0] + sign * ( imag_unit * P[1][2] ) + sign * ( imag_unit * P[3][0] ) - P[3][2]);
	PT[1][1] = coeff * (P[1][1] + sign * ( imag_unit * P[1][3] ) + sign * ( imag_unit * P[3][1] ) - P[3][3]);
	PT[1][2] = coeff * (sign * ( imag_unit * P[1][0] ) + P[1][2] - P[3][0] + sign * ( imag_unit * P[3][2] ));
	PT[1][3] = coeff * (sign * ( imag_unit * P[1][1] ) + P[1][3] - P[3][1] + sign * ( imag_unit * P[3][3] ));
	
	PT[2][0] = coeff * (sign * ( imag_unit * P[0][0] ) - P[0][2] + P[2][0] + sign * ( imag_unit * P[2][2] ));
	PT[2][1] = coeff * (sign * ( imag_unit * P[0][1] ) - P[0][3] + P[2][1] + sign * ( imag_unit * P[2][3] ));
	PT[2][2] = coeff * (sign * ( imag_unit * P[0][2] ) - P[0][0] + sign * ( imag_unit * P[2][0] ) + P[2][2]);
	PT[2][3] = coeff * (sign * ( imag_unit * P[0][3] ) - P[0][1] + sign * ( imag_unit * P[2][1] ) + P[2][3]);

	PT[3][0] = coeff * (sign * ( imag_unit * P[1][0] ) - P[1][2] + P[3][0] + sign * ( imag_unit * P[3][2] ));
	PT[3][1] = coeff * (sign * ( imag_unit * P[1][1] ) - P[1][3] + P[3][1] + sign * ( imag_unit * P[3][3] ));
	PT[3][2] = coeff * (sign * ( imag_unit * P[1][2] ) - P[1][0] + sign * ( imag_unit * P[3][0] ) + P[3][2]);
	PT[3][3] = coeff * (sign * ( imag_unit * P[1][3] ) - P[1][1] + sign * ( imag_unit * P[3][1] ) + P[3][3]);

	for(int mu = 0 ; mu < 4 ; mu++)
	  for(int nu = 0 ; nu < 4 ; nu++){
	    CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + 
			nu*GK_nColor*GK_nColor*GK_localVolume + 
			c1*GK_nColor*GK_localVolume + 
			c2*GK_localVolume + iv)*2 + 0] = PT[mu][nu].real();
	    CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + 
			nu*GK_nColor*GK_nColor*GK_localVolume + 
			c1*GK_nColor*GK_localVolume + 
			c2*GK_localVolume + iv)*2 + 1] = PT[mu][nu].imag();
	  }
      }
}

// gpu collect ghost and send it to host
template<typename Float>
void QKXTM_Propagator_Kepler<Float>::ghostToHost(){   
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
    // set plus points to minus area
    position = (GK_localL[0]-1);
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int nu = 0 ; nu < GK_nSpin ; nu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++){
      d_elem_offset = (CC::d_elem + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       position*2);  
      h_elem_offset = (CC::h_elem + 
		       GK_minusGhost[0]*GK_nSpin*GK_nSpin*
		       GK_nColor*GK_nColor*2 + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_surface3D[0]*2 + 
		       nu*GK_nColor*GK_nColor*GK_surface3D[0]*2 + 
		       c1*GK_nColor*GK_surface3D[0]*2 + 
		       c2*GK_surface3D[0]*2);
      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		   spitch,width,height,cudaMemcpyDeviceToHost);
    }
    // set minus points to plus area
    position = 0;

    for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int nu = 0 ; nu < GK_nSpin ; nu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++){
      d_elem_offset = (CC::d_elem + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       position*2);  
      h_elem_offset = (CC::h_elem + 
		       GK_plusGhost[0]*GK_nSpin*GK_nSpin*
		       GK_nColor*GK_nColor*2 + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_surface3D[0]*2 + 
		       nu*GK_nColor*GK_nColor*GK_surface3D[0]*2 + 
		       c1*GK_nColor*GK_surface3D[0]*2 + 
		       c2*GK_surface3D[0]*2);
      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		   spitch,width,height,cudaMemcpyDeviceToHost);
    }
  }
  
  // direction y   
  if( GK_localL[1] < GK_totalL[1]){
    
    int position;
    int height = GK_localL[2] * GK_localL[3]; // number of blocks that we need
    size_t width = GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[1]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;

    // set plus points to minus area
    position = GK_localL[0]*(GK_localL[1]-1);
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int nu = 0 ; nu < GK_nSpin ; nu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++){
      d_elem_offset = (CC::d_elem + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       position*2);  
      h_elem_offset = (CC::h_elem + 
		       GK_minusGhost[1]*GK_nSpin*GK_nSpin*
		       GK_nColor*GK_nColor*2 + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_surface3D[1]*2 + 
		       nu*GK_nColor*GK_nColor*GK_surface3D[1]*2 + 
		       c1*GK_nColor*GK_surface3D[1]*2 + 
		       c2*GK_surface3D[1]*2);
      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		   spitch,width,height,cudaMemcpyDeviceToHost);
    }
    
    // set minus points to plus area
    position = 0;
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int nu = 0 ; nu < GK_nSpin ; nu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++){
      d_elem_offset = (CC::d_elem + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       position*2);  
      h_elem_offset = (CC::h_elem + 
		       GK_plusGhost[1]*GK_nSpin*GK_nSpin*
		       GK_nColor*GK_nColor*2 + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_surface3D[1]*2 + 
		       nu*GK_nColor*GK_nColor*GK_surface3D[1]*2 + 
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
    // position = GK_localL[0]*GK_localL[1]*(GK_localL[2]-1)*GK_localL[3];
    position = GK_localL[0]*GK_localL[1]*(GK_localL[2]-1);
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int nu = 0 ; nu < GK_nSpin ; nu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++){
      d_elem_offset = (CC::d_elem + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       position*2);  
      h_elem_offset = (CC::h_elem + 
		       GK_minusGhost[2]*GK_nSpin*GK_nSpin*
		       GK_nColor*GK_nColor*2 + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_surface3D[2]*2 + 
		       nu*GK_nColor*GK_nColor*GK_surface3D[2]*2 + 
		       c1*GK_nColor*GK_surface3D[2]*2 + 
		       c2*GK_surface3D[2]*2);
      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		   spitch,width,height,cudaMemcpyDeviceToHost);
    }

    // set minus points to plus area
    position = 0;

    for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int nu = 0 ; nu < GK_nSpin ; nu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++){
      d_elem_offset = (CC::d_elem + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       position*2);  
      h_elem_offset = (CC::h_elem + 
		       GK_plusGhost[2]*GK_nSpin*GK_nSpin*
		       GK_nColor*GK_nColor*2 + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_surface3D[2]*2 + 
		       nu*GK_nColor*GK_nColor*GK_surface3D[2]*2 + 
		       c1*GK_nColor*GK_surface3D[2]*2 + 
		       c2*GK_surface3D[2]*2);
      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		   spitch,width,height,cudaMemcpyDeviceToHost);
    }
  }
  
  // direction t 
  if( GK_localL[3] < GK_totalL[3]){
    int position;
    int height = GK_nSpin*GK_nSpin*GK_nColor*GK_nColor;
    size_t width = GK_localL[2]*GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[3]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;

    // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*GK_localL[2]*(GK_localL[3]-1);
    d_elem_offset = CC::d_elem + position*2;
    h_elem_offset = CC::h_elem + GK_minusGhost[3]*GK_nSpin*GK_nSpin*GK_nColor*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,
		 width,height,cudaMemcpyDeviceToHost);

    // set minus points to plus area
    position = 0;
    d_elem_offset = CC::d_elem + position*2;
    h_elem_offset = CC::h_elem + GK_plusGhost[3]*GK_nSpin*GK_nSpin*GK_nColor*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,
		 width,height,cudaMemcpyDeviceToHost);
    
    checkCudaError();
  }
}

template<typename Float>
void QKXTM_Propagator_Kepler<Float>::cpuExchangeGhost(){
  if( comm_size() > 1 ){
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_from_back[4];
    MsgHandle *mh_from_fwd[4];
    MsgHandle *mh_send_back[4];

    Float *pointer_receive = NULL;
    Float *pointer_send = NULL;

    for(int idim = 0 ; idim < GK_nDim; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	size_t nbytes = GK_surface3D[idim]*GK_nSpin*GK_nColor*GK_nSpin*GK_nColor*2*sizeof(Float);
	// send to plus
	pointer_receive = CC::h_ext_ghost + (GK_minusGhost[idim]-GK_localVolume)*GK_nSpin*GK_nColor*GK_nSpin*GK_nColor*2;
	pointer_send = CC::h_elem + GK_minusGhost[idim]*GK_nSpin*GK_nColor*GK_nSpin*GK_nColor*2;

	mh_from_back[idim] = comm_declare_receive_relative(pointer_receive,idim,-1,nbytes);
	mh_send_fwd[idim] = comm_declare_send_relative(pointer_send,idim,1,nbytes);
	comm_start(mh_from_back[idim]);
	comm_start(mh_send_fwd[idim]);
	comm_wait(mh_send_fwd[idim]);
	comm_wait(mh_from_back[idim]);
		
	// send to minus
	pointer_receive = CC::h_ext_ghost + (GK_plusGhost[idim]-GK_localVolume)*GK_nSpin*GK_nColor*GK_nSpin*GK_nColor*2;
	pointer_send = CC::h_elem + GK_plusGhost[idim]*GK_nSpin*GK_nColor*GK_nSpin*GK_nColor*2;

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
void QKXTM_Propagator_Kepler<Float>::ghostToDevice(){ 
  if(comm_size() > 1){
    Float *host = CC::h_ext_ghost;
    Float *device = CC::d_elem + GK_localVolume*GK_nSpin*GK_nColor*GK_nSpin*GK_nColor*2;
    cudaMemcpy(device,host,CC::bytes_ghost_length,cudaMemcpyHostToDevice);
    checkCudaError();
  }
}

template<typename Float>
void  QKXTM_Propagator_Kepler<Float>::conjugate(){
  run_conjugate_propagator((void*)CC::d_elem,sizeof(Float));
}

template<typename Float>
void  QKXTM_Propagator_Kepler<Float>::apply_gamma5(){
  run_apply_gamma5_propagator((void*)CC::d_elem,sizeof(Float));
}

//----------------------------------//
// class QKXTM_ Propagator3D_Kepler //
//----------------------------------//

template<typename Float>
QKXTM_Propagator3D_Kepler<Float>::
QKXTM_Propagator3D_Kepler(ALLOCATION_FLAG alloc_flag, 
			  CLASS_ENUM classT): 
  QKXTM_Field_Kepler<Float>(alloc_flag, classT){
  if(alloc_flag != BOTH)
    errorQuda("Propagator3D class is only implemented to allocate memory for both\n");
}

template<typename Float>
void QKXTM_Propagator3D_Kepler<Float>::
absorbTimeSliceFromHost(QKXTM_Propagator_Kepler<Float> &prop, 
			int timeslice){
  int V3 = GK_localVolume/GK_localL[3];
  
  for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++)
  for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
  for(int iv3 = 0 ; iv3 < V3 ; iv3++)
  for(int ipart = 0 ; ipart < 2 ; ipart++)
    CC::h_elem[ (mu*GK_nSpin*GK_nColor*GK_nColor*V3 + 
		 nu*GK_nColor*GK_nColor*V3 + 
		 c1*GK_nColor*V3 + 
		 c2*V3 + iv3)*2 + ipart] = 
      prop.H_elem()[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + 
		     nu*GK_nColor*GK_nColor*GK_localVolume + 
		     c1*GK_nColor*GK_localVolume + 
		     c2*GK_localVolume + 
		     timeslice*V3 + iv3)*2 + ipart];
  
  cudaMemcpy(CC::d_elem,CC::h_elem,
	     GK_nSpin*GK_nSpin*GK_nColor*GK_nColor*V3*2*sizeof(Float),
	     cudaMemcpyHostToDevice);
  checkCudaError();
}

template<typename Float>
void QKXTM_Propagator3D_Kepler<Float>::
absorbTimeSlice(QKXTM_Propagator_Kepler<Float> &prop, int timeslice){
  int V3 = GK_localVolume/GK_localL[3];
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;

  for(int mu=0; mu<4; mu++)
    for(int nu=0; nu<4; nu++)
      for(int c1=0; c1<3; c1++)
	for(int c2=0; c2<3; c2++){
	  pointer_dst = (CC::d_elem + mu*4*3*3*V3*2 + nu*3*3*V3*2 + 
			 c1*3*V3*2 + c2*V3*2);
	  pointer_src = (prop.D_elem() + mu*4*3*3*GK_localVolume*2 + 
			 nu*3*3*GK_localVolume*2 + c1*3*GK_localVolume*2 + 
			 c2*GK_localVolume*2 + timeslice*V3*2);
	  cudaMemcpy(pointer_dst, pointer_src, V3*2*sizeof(Float), 
		     cudaMemcpyDeviceToDevice);
	}
  checkCudaError();
  pointer_src = NULL;
  pointer_dst = NULL;
}

template<typename Float>
void QKXTM_Propagator3D_Kepler<Float>::
absorbVectorTimeSlice(QKXTM_Vector_Kepler<Float> &vec, 
		      int timeslice, int nu, int c2){
  int V3 = GK_localVolume/GK_localL[3];
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int c1 = 0 ; c1 < 3 ; c1++){
      pointer_dst = (CC::d_elem + mu*4*3*3*V3*2 + nu*3*3*V3*2 + 
		     c1*3*V3*2 + c2*V3*2);
      pointer_src = (vec.D_elem() + mu*3*GK_localVolume*2 + 
		     c1*GK_localVolume*2 + timeslice*V3*2);
      cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), 
		 cudaMemcpyDeviceToDevice);
    }
}

template<typename Float>
void QKXTM_Propagator3D_Kepler<Float>::broadcast(int tsink){
  cudaMemcpy(CC::h_elem , CC::d_elem , CC::bytes_total_length , 
	     cudaMemcpyDeviceToHost);
  checkCudaError();
  comm_barrier();
  int bcastRank = tsink/GK_localL[3];
  int V3 = GK_localVolume/GK_localL[3];
  if( typeid(Float) == typeid(float) ){
    int error = MPI_Bcast(CC::h_elem , 4*4*3*3*V3*2 , MPI_FLOAT , 
			  bcastRank , GK_timeComm );
    if(error != MPI_SUCCESS)errorQuda("Error in mpi broadcasting");
  }
  else if( typeid(Float) == typeid(double) ){
    int error = MPI_Bcast(CC::h_elem , 4*4*3*3*V3*2 , MPI_DOUBLE , 
			  bcastRank , GK_timeComm );
    if(error != MPI_SUCCESS)errorQuda("Error in mpi broadcasting");    
  }
  cudaMemcpy(CC::d_elem , CC::h_elem , CC::bytes_total_length, 
	     cudaMemcpyHostToDevice);
  checkCudaError();
}
