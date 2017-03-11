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

#ifdef HAVE_MKL
#include <mkl.h>
#endif

#ifdef HAVE_OPENBLAS
#include <cblas.h>
#include <common.h>
#endif

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

//---------------------------//
// class QKXTM_Vector_Kepler //
//---------------------------//

template<typename Float>
QKXTM_Vector_Kepler<Float>::QKXTM_Vector_Kepler(ALLOCATION_FLAG alloc_flag, 
						CLASS_ENUM classT): 
  QKXTM_Field_Kepler<Float>(alloc_flag, classT){ ; }

template<typename Float>
void QKXTM_Vector_Kepler<Float>::packVector(Float *vector){
  for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int mu = 0 ; mu < GK_nSpin ; mu++)  // always work with format colors inside spins
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int part = 0 ; part < 2 ; part++){
	  CC::h_elem[mu*GK_nColor*GK_localVolume*2 + 
		     c1*GK_localVolume*2 + iv*2 + part] = 
	    vector[iv*GK_nSpin*GK_nColor*2 + mu*GK_nColor*2 + c1*2 + part];
	}
}
 

template<typename Float>
void QKXTM_Vector_Kepler<Float>::unpackVector(){

  Float *vector_tmp = (Float*) malloc( CC::bytes_total_length );
  if(vector_tmp == NULL)
    errorQuda("Error in allocate memory of tmp vector in unpackVector\n");
  
  for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int mu = 0 ; mu < GK_nSpin ; mu++) // always work with format colors inside spins
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int part = 0 ; part < 2 ; part++){
	  vector_tmp[iv*GK_nSpin*GK_nColor*2 + mu*GK_nColor*2+c1*2+part] = 
	    CC::h_elem[mu*GK_nColor*GK_localVolume*2 + 
		       c1*GK_localVolume*2 + iv*2 + part];
	}
  
  memcpy(CC::h_elem,vector_tmp, CC::bytes_total_length);
  
  free(vector_tmp);
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::unpackVector(Float *vector){
  
  for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int mu = 0 ; mu < GK_nSpin ; mu++) // always work with format colors inside spins
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int part = 0 ; part < 2 ; part++){
	  CC::h_elem[iv*GK_nSpin*GK_nColor*2 + mu*GK_nColor*2+c1*2+part] = 
	    vector[mu*GK_nColor*GK_localVolume*2 + 
		   c1*GK_localVolume*2 + iv*2 + part];
	}
}


template<typename Float>
void QKXTM_Vector_Kepler<Float>::loadVector(){
  cudaMemcpy(CC::d_elem,CC::h_elem,CC::bytes_total_length, 
	     cudaMemcpyHostToDevice );
  checkCudaError();
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::unloadVector(){
  cudaMemcpy(CC::h_elem, CC::d_elem, CC::bytes_total_length, 
	     cudaMemcpyDeviceToHost);
  checkCudaError();
}


template<typename Float>
void QKXTM_Vector_Kepler<Float>::download(){

  cudaMemcpy(CC::h_elem, CC::d_elem, CC::bytes_total_length, 
	     cudaMemcpyDeviceToHost);
  checkCudaError();

  Float *vector_tmp = (Float*) malloc( CC::bytes_total_length );
  if(vector_tmp == NULL) errorQuda("Error in allocate memory of tmp vector");

  for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int mu = 0 ; mu < GK_nSpin ; mu++) // always work with format colors inside spins
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int part = 0 ; part < 2 ; part++){
	  vector_tmp[iv*GK_nSpin*GK_nColor*2 + mu*GK_nColor*2+c1*2+part] = 
	    CC::h_elem[mu*GK_nColor*GK_localVolume*2 + 
		       c1*GK_localVolume*2 + iv*2 + part];
	}
  
  memcpy(CC::h_elem, vector_tmp, CC::bytes_total_length);

  free(vector_tmp);
}


template<typename Float>
void QKXTM_Vector_Kepler<Float>::castDoubleToFloat(QKXTM_Vector_Kepler<double> &vecIn){
  if(typeid(Float) != typeid(float) )errorQuda("This method works only to convert double to single precision\n");
  run_castDoubleToFloat((void*)CC::d_elem, (void*)vecIn.D_elem());
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::castFloatToDouble(QKXTM_Vector_Kepler<float> &vecIn){
  if(typeid(Float) != typeid(double) )errorQuda("This method works only to convert single to double precision\n");
  run_castFloatToDouble((void*)CC::d_elem, (void*)vecIn.D_elem());
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::ghostToHost(){
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
      for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	d_elem_offset = (CC::d_elem + 
			 mu*GK_nColor*GK_localVolume*2 + 
			 c1*GK_localVolume*2 + 
			 position*2);
	h_elem_offset = (CC::h_elem + 
			 GK_minusGhost[0]*GK_nSpin*GK_nColor*2 + 
			 mu*GK_nColor*GK_surface3D[0]*2 + 
			 c1*GK_surface3D[0]*2);
	cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		     spitch,width,height,cudaMemcpyDeviceToHost);
      }
    // set minus points to plus area
    position = 0;
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	d_elem_offset = (CC::d_elem + 
			 mu*GK_nColor*GK_localVolume*2 + 
			 c1*GK_localVolume*2 + 
			 position*2);  
	h_elem_offset = (CC::h_elem + 
			 GK_plusGhost[0]*GK_nSpin*GK_nColor*2 + 
			 mu*GK_nColor*GK_surface3D[0]*2 + 
			 c1*GK_surface3D[0]*2);
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
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	d_elem_offset = (CC::d_elem + 
			 mu*GK_nColor*GK_localVolume*2 + 
			 c1*GK_localVolume*2 + 
			 position*2);  
	h_elem_offset = (CC::h_elem + 
			 GK_minusGhost[1]*GK_nSpin*GK_nColor*2 + 
			 mu*GK_nColor*GK_surface3D[1]*2 + 
			 c1*GK_surface3D[1]*2);
	cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		     spitch,width,height,cudaMemcpyDeviceToHost);
      }
    // set minus points to plus area
    position = 0;
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	d_elem_offset = (CC::d_elem + 
			 mu*GK_nColor*GK_localVolume*2 + 
			 c1*GK_localVolume*2 + 
			 position*2);  
	h_elem_offset = (CC::h_elem + 
			 GK_plusGhost[1]*GK_nSpin*GK_nColor*2 + 
			 mu*GK_nColor*GK_surface3D[1]*2 + 
			 c1*GK_surface3D[1]*2);
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
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	d_elem_offset = (CC::d_elem + 
			 mu*GK_nColor*GK_localVolume*2 + 
			 c1*GK_localVolume*2 + 
			 position*2);  
	h_elem_offset = (CC::h_elem + 
			 GK_minusGhost[2]*GK_nSpin*GK_nColor*2 + 
			 mu*GK_nColor*GK_surface3D[2]*2 + 
			 c1*GK_surface3D[2]*2);
	cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		     spitch,width,height,cudaMemcpyDeviceToHost);
      }
    // set minus points to plus area
    position = 0;
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	d_elem_offset = (CC::d_elem + 
			 mu*GK_nColor*GK_localVolume*2 + 
			 c1*GK_localVolume*2 + 
			 position*2);  
	h_elem_offset = (CC::h_elem + 
			 GK_plusGhost[2]*GK_nSpin*GK_nColor*2 + 
			 mu*GK_nColor*GK_surface3D[2]*2 + 
			 c1*GK_surface3D[2]*2);
	cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		     spitch,width,height,cudaMemcpyDeviceToHost);
      }
  }
  // direction t 
  if( GK_localL[3] < GK_totalL[3]){
    int position;
    int height = GK_nSpin*GK_nColor;
    size_t width = GK_localL[2]*GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[3]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*GK_localL[2]*(GK_localL[3]-1);
    d_elem_offset = CC::d_elem + position*2;
    h_elem_offset = CC::h_elem + GK_minusGhost[3]*GK_nSpin*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,
		 cudaMemcpyDeviceToHost);
    // set minus points to plus area
    position = 0;
    d_elem_offset = CC::d_elem + position*2;
    h_elem_offset = CC::h_elem + GK_plusGhost[3]*GK_nSpin*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,
		 cudaMemcpyDeviceToHost);
  }
}


template<typename Float>
void QKXTM_Vector_Kepler<Float>::cpuExchangeGhost(){
  if( comm_size() > 1 ){
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_from_back[4];
    MsgHandle *mh_from_fwd[4];
    MsgHandle *mh_send_back[4];

    Float *pointer_receive = NULL;
    Float *pointer_send = NULL;

    for(int idim = 0 ; idim < GK_nDim; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	size_t nbytes=GK_surface3D[idim]*GK_nSpin*GK_nColor*2*sizeof(Float);
	// send to plus
	pointer_receive = CC::h_ext_ghost + (GK_minusGhost[idim]-GK_localVolume)*GK_nSpin*GK_nColor*2;
	pointer_send = CC::h_elem + GK_minusGhost[idim]*GK_nSpin*GK_nColor*2;

	mh_from_back[idim] = comm_declare_receive_relative(pointer_receive,idim,-1,nbytes);
	mh_send_fwd[idim] = comm_declare_send_relative(pointer_send,idim,1,nbytes);
	comm_start(mh_from_back[idim]);
	comm_start(mh_send_fwd[idim]);
	comm_wait(mh_send_fwd[idim]);
	comm_wait(mh_from_back[idim]);
		
	// send to minus
	pointer_receive = CC::h_ext_ghost + (GK_plusGhost[idim]-GK_localVolume)*GK_nSpin*GK_nColor*2;
	pointer_send = CC::h_elem + GK_plusGhost[idim]*GK_nSpin*GK_nColor*2;

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
void QKXTM_Vector_Kepler<Float>::ghostToDevice(){ 
  if(comm_size() > 1){
    Float *host = CC::h_ext_ghost;
    Float *device = CC::d_elem + GK_localVolume*GK_nSpin*GK_nColor*2;
    cudaMemcpy(device,host,CC::bytes_ghost_length,cudaMemcpyHostToDevice);
    checkCudaError();
  }
}


template<typename Float>
void QKXTM_Vector_Kepler<Float>::gaussianSmearing(QKXTM_Vector_Kepler<Float> &vecIn,QKXTM_Gauge_Kepler<Float> &gaugeAPE){
  gaugeAPE.ghostToHost();
  gaugeAPE.cpuExchangeGhost();
  gaugeAPE.ghostToDevice();

  vecIn.ghostToHost();
  vecIn.cpuExchangeGhost();
  vecIn.ghostToDevice();

  cudaTextureObject_t texVecIn,texVecOut,texGauge;
  this->createTexObject(&texVecOut);
  vecIn.createTexObject(&texVecIn);
  gaugeAPE.createTexObject(&texGauge);
  
  for(int i = 0 ; i < GK_nsmearGauss ; i++){
    if( (i%2) == 0){
      run_GaussianSmearing((void*)this->D_elem(),texVecIn,texGauge, sizeof(Float));
      this->ghostToHost();
      this->cpuExchangeGhost();
      this->ghostToDevice();
    }
    else{
      run_GaussianSmearing((void*)vecIn.D_elem(),texVecOut,texGauge, sizeof(Float));
      vecIn.ghostToHost();
      vecIn.cpuExchangeGhost();
      vecIn.ghostToDevice();
    }
  }

  if( (GK_nsmearGauss%2) == 0) cudaMemcpy(this->D_elem(),vecIn.D_elem(),CC::bytes_total_length,cudaMemcpyDeviceToDevice);
  
  this->destroyTexObject(texVecOut);
  vecIn.destroyTexObject(texVecIn);
  gaugeAPE.destroyTexObject(texGauge);
  checkCudaError();
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::uploadToCuda(ColorSpinorField *qudaVector, 
					      bool isEv){
  run_UploadToCuda((void*) CC::d_elem, *qudaVector, sizeof(Float), isEv);
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::downloadFromCuda(ColorSpinorField *qudaVector, bool isEv){
  run_DownloadFromCuda((void*) CC::d_elem, *qudaVector, sizeof(Float), isEv);
}

template<typename Float>
void  QKXTM_Vector_Kepler<Float>::scaleVector(double a){
  run_ScaleVector(a,(void*)CC::d_elem,sizeof(Float));
}

template<typename Float>
void  QKXTM_Vector_Kepler<Float>::conjugate(){
  run_conjugate_vector((void*)CC::d_elem,sizeof(Float));
}

template<typename Float>
void  QKXTM_Vector_Kepler<Float>::apply_gamma5(){
  run_apply_gamma5_vector((void*)CC::d_elem,sizeof(Float));
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::norm2Host(){
  Float res = 0.;
  Float globalRes;

  for(int i = 0 ; i < GK_nSpin*GK_nColor*GK_localVolume ; i++){
    res += CC::h_elem[i*2 + 0]*CC::h_elem[i*2 + 0] + CC::h_elem[i*2 + 1]*CC::h_elem[i*2 + 1];
  }

  int rc = MPI_Allreduce(&res , &globalRes , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);
  if( rc != MPI_SUCCESS ) errorQuda("Error in MPI reduction for plaquette");
  printfQuda("Vector norm2 is %e\n",globalRes);
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::copyPropagator3D(QKXTM_Propagator3D_Kepler<Float> &prop, int timeslice, int nu , int c2){
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  int V3 = GK_localVolume/GK_localL[3];
  
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int c1 = 0 ; c1 < 3 ; c1++){
      pointer_dst = (CC::d_elem + 
		     mu*3*GK_localVolume*2 + 
		     c1*GK_localVolume*2 + 
		     timeslice*V3*2);
      pointer_src = (prop.D_elem() + 
		     mu*4*3*3*V3*2 + 
		     nu*3*3*V3*2 + 
		     c1*3*V3*2 + 
		     c2*V3*2);
      cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), 
		 cudaMemcpyDeviceToDevice);
    }

  pointer_src = NULL;
  pointer_dst = NULL;
  checkCudaError();

}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::copyPropagator(QKXTM_Propagator_Kepler<Float> &prop, int nu , int c2){
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int c1 = 0 ; c1 < 3 ; c1++){
      pointer_dst = (CC::d_elem + 
		     mu*3*GK_localVolume*2 + 
		     c1*GK_localVolume*2);
      pointer_src = (prop.D_elem() + 
		     mu*4*3*3*GK_localVolume*2 + 
		     nu*3*3*GK_localVolume*2 + 
		     c1*3*GK_localVolume*2 + 
		     c2*GK_localVolume*2);
      cudaMemcpy(pointer_dst, pointer_src, GK_localVolume*2 * sizeof(Float), cudaMemcpyDeviceToDevice);
    }
  
  pointer_src = NULL;
  pointer_dst = NULL;
  checkCudaError();

}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::write(char *filename){
  FILE *fid;
  int error_in_header=0;
  LimeWriter *limewriter;
  LimeRecordHeader *limeheader = NULL;
  int ME_flag=0, MB_flag=0, limeStatus;
  u_int64_t message_length;
  MPI_Offset offset;
  MPI_Datatype subblock;  //MPI-type, 5d subarray  
  MPI_File mpifid;
  MPI_Status status;
  int sizes[5], lsizes[5], starts[5];
  long int i;
  int chunksize,mu,c1;
  char *buffer;
  int x,y,z,t;
  char tmp_string[2048];

  if(comm_rank() == 0){ // master will write the lime header
    fid = fopen(filename,"w");
    if(fid == NULL){
      fprintf(stderr,"Error open file to write propagator in %s \n",__func__);
      comm_abort(-1);
    }
    else{
      limewriter = limeCreateWriter(fid);
      if(limewriter == (LimeWriter*)NULL) {
	fprintf(stderr, "Error in %s. LIME error in file for writing!\n", __func__);
	error_in_header=1;
	comm_abort(-1);
      }
      else
	{
	  sprintf(tmp_string, "DiracFermion_Sink");
	  message_length=(long int) strlen(tmp_string);
	  MB_flag=1; ME_flag=1;
	  limeheader = limeCreateHeader(MB_flag, ME_flag, "propagator-type", message_length);
	  if(limeheader == (LimeRecordHeader*)NULL)
	    {
	      fprintf(stderr, "Error in %s. LIME create header error.\n", __func__);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  limeStatus = limeWriteRecordHeader(limeheader, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header %d\n", __func__, limeStatus);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  limeDestroyHeader(limeheader);
	  limeStatus = limeWriteRecordData(tmp_string, &message_length, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
	      error_in_header=1;
	      comm_abort(-1);
	    }

	  if( typeid(Float) == typeid(double) )
	    sprintf(tmp_string, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<etmcFormat>\n\t<field>diracFermion</field>\n\t<precision>64</precision>\n\t<flavours>1</flavours>\n\t<lx>%d</lx>\n\t<ly>%d</ly>\n\t<lz>%d</lz>\n\t<lt>%d</lt>\n\t<spin>4</spin>\n\t<colour>3</colour>\n</etmcFormat>", GK_totalL[0], GK_totalL[1], GK_totalL[2], GK_totalL[3]);
	  else
	    sprintf(tmp_string, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<etmcFormat>\n\t<field>diracFermion</field>\n\t<precision>32</precision>\n\t<flavours>1</flavours>\n\t<lx>%d</lx>\n\t<ly>%d</ly>\n\t<lz>%d</lz>\n\t<lt>%d</lt>\n\t<spin>4</spin>\n\t<colour>3</colour>\n</etmcFormat>", GK_totalL[0], GK_totalL[1], GK_totalL[2], GK_totalL[3]);

	  message_length=(long int) strlen(tmp_string); 
	  MB_flag=1; ME_flag=1;

	  limeheader = limeCreateHeader(MB_flag, ME_flag, "quda-propagator-format", message_length);
	  if(limeheader == (LimeRecordHeader*)NULL)
	    {
	      fprintf(stderr, "Error in %s. LIME create header error.\n", __func__);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  limeStatus = limeWriteRecordHeader(limeheader, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header %d\n", __func__, limeStatus);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  limeDestroyHeader(limeheader);
	  limeStatus = limeWriteRecordData(tmp_string, &message_length, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  
	  message_length = GK_totalVolume*4*3*2*sizeof(Float);
	  MB_flag=1; ME_flag=1;
	  limeheader = limeCreateHeader(MB_flag, ME_flag, "scidac-binary-data", message_length);
	  limeStatus = limeWriteRecordHeader( limeheader, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
	      error_in_header=1;
	    }
	  limeDestroyHeader( limeheader );
	}
      message_length=1;
      limeWriteRecordData(tmp_string, &message_length, limewriter);
      limeDestroyWriter(limewriter);
      offset = ftell(fid)-1;
      fclose(fid);
    }
  }

  MPI_Bcast(&offset,sizeof(MPI_Offset),MPI_BYTE,0,MPI_COMM_WORLD);
  
  sizes[0]=GK_totalL[3];
  sizes[1]=GK_totalL[2];
  sizes[2]=GK_totalL[1];
  sizes[3]=GK_totalL[0];
  sizes[4]=4*3*2;
  lsizes[0]=GK_localL[3];
  lsizes[1]=GK_localL[2];
  lsizes[2]=GK_localL[1];
  lsizes[3]=GK_localL[0];
  lsizes[4]=sizes[4];
  starts[0]=comm_coords(default_topo)[3]*GK_localL[3];
  starts[1]=comm_coords(default_topo)[2]*GK_localL[2];
  starts[2]=comm_coords(default_topo)[1]*GK_localL[1];
  starts[3]=comm_coords(default_topo)[0]*GK_localL[0];
  starts[4]=0;  

  if( typeid(Float) == typeid(double) )
    MPI_Type_create_subarray(5,sizes,lsizes,starts,
			     MPI_ORDER_C,MPI_DOUBLE,&subblock);
  else
    MPI_Type_create_subarray(5,sizes,lsizes,starts,
			     MPI_ORDER_C,MPI_FLOAT,&subblock);

  MPI_Type_commit(&subblock);
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY, 
		MPI_INFO_NULL, &mpifid);
  MPI_File_set_view(mpifid, offset, MPI_FLOAT, subblock, 
		    "native", MPI_INFO_NULL);

  chunksize=4*3*2*sizeof(Float);
  buffer = (char*) malloc(chunksize*GK_localVolume);

  if(buffer==NULL)  
    {
      fprintf(stderr,"Error in %s! Out of memory\n", __func__);
      comm_abort(-1);
    }

  i=0;
                        
  for(t=0; t<GK_localL[3];t++)
  for(z=0; z<GK_localL[2];z++)
  for(y=0; y<GK_localL[1];y++)
  for(x=0; x<GK_localL[0];x++)
  for(mu=0; mu<4; mu++)
  for(c1=0; c1<3; c1++) 
    // works only for QUDA_DIRAC_ORDER (color inside spin)
    {
      ((Float *)buffer)[i] = 
	(CC::h_elem[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + 
		    z*GK_localL[1]*GK_localL[0]*4*3*2 + 
		    y*GK_localL[0]*4*3*2 + 
		    x*4*3*2 + mu*3*2 + c1*2 + 0]);
      
      ((Float *)buffer)[i+1] = 
	(CC::h_elem[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + 
		    z*GK_localL[1]*GK_localL[0]*4*3*2 + 
		    y*GK_localL[0]*4*3*2 + 
		    x*4*3*2 + mu*3*2 + c1*2 + 1]);
      i+=2;
    }
  if(!qcd_isBigEndian()){
    if( typeid(Float) == typeid(double) ) 
      qcd_swap_8((double*) buffer,2*4*3*GK_localVolume);
    else qcd_swap_4((float*) buffer,2*4*3*GK_localVolume);
  }
  if( typeid(Float) == typeid(double) )
    MPI_File_write_all(mpifid, buffer, 4*3*2*GK_localVolume, 
		       MPI_DOUBLE, &status);
  else
    MPI_File_write_all(mpifid, buffer, 4*3*2*GK_localVolume, 
		       MPI_FLOAT, &status);

  free(buffer);
  MPI_File_close(&mpifid);
  MPI_Type_free(&subblock);
}
