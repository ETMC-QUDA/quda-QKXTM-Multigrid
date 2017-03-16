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

//--------------------------------//
// class QKXTM_Contraction_Kepler //
//--------------------------------//

#define N_MESONS 10
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
contractMesons(QKXTM_Propagator_Kepler<Float> &prop1,
	       QKXTM_Propagator_Kepler<Float> &prop2, 
	       char *filename_out, int isource){
  
  errorQuda("contractMesons: This version of the function is obsolete. Cannot guarantee correct results. Please call the overloaded-updated version of this function with the corresponding list of arguments.\n");

  cudaTextureObject_t texProp1, texProp2;
  prop1.createTexObject(&texProp1);
  prop2.createTexObject(&texProp2);

  if( typeid(Float) == typeid(float))  
    printfQuda("contractMesons: Will perform in single precision\n");
  if( typeid(Float) == typeid(double)) 
    printfQuda("contractMesons: Will perform in double precision\n");
  
  Float (*corr_mom_local)[2][N_MESONS] =(Float(*)[2][N_MESONS]) calloc(GK_localL[3]*GK_Nmoms*2*N_MESONS*2,sizeof(Float));
  Float (*corr_mom_local_reduced)[2][N_MESONS] =(Float(*)[2][N_MESONS]) calloc(GK_localL[3]*GK_Nmoms*2*N_MESONS*2,sizeof(Float));
  Float (*corr_mom)[2][N_MESONS] = (Float(*)[2][N_MESONS]) calloc(GK_totalL[3]*GK_Nmoms*2*N_MESONS*2,sizeof(Float));
  
  if( corr_mom_local == NULL || 
      corr_mom_local_reduced == NULL || 
      corr_mom == NULL )
    errorQuda("Error problem to allocate memory");
  
  for(int it = 0 ; it < GK_localL[3] ; it++){
    run_contractMesons(texProp1,texProp2,(void*) corr_mom_local,it,isource,sizeof(Float),MOMENTUM_SPACE);
  }
  
  int error;

  if( typeid(Float) == typeid(float) ){
    MPI_Reduce(corr_mom_local, corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_FLOAT,MPI_SUM,0, GK_spaceComm);
    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_FLOAT,corr_mom,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_FLOAT,0,GK_timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
  }
  else{
    MPI_Reduce(corr_mom_local,corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_DOUBLE,MPI_SUM,0, GK_spaceComm);
    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_DOUBLE,corr_mom,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_DOUBLE,0,GK_timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
  }

  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out = fopen(filename_out,"w");
    if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
    for(int ip = 0 ; ip < N_MESONS ; ip++)
      for(int it = 0 ; it < GK_totalL[3] ; it++)
	for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	  int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	  fprintf(ptr_out,"%d \t %d \t %+d %+d %+d \t %+e %+e \t %+e %+e\n",
		  ip,it,
		  GK_moms[imom][0],GK_moms[imom][1],GK_moms[imom][2],
		  corr_mom[it_shift*GK_Nmoms*2+imom*2+0][0][ip], 
		  corr_mom[it_shift*GK_Nmoms*2+imom*2+1][0][ip], 
		  corr_mom[it_shift*GK_Nmoms*2+imom*2+0][1][ip], 
		  corr_mom[it_shift*GK_Nmoms*2+imom*2+1][1][ip]);
	}
    fclose(ptr_out);
  }

  free(corr_mom_local);
  free(corr_mom_local_reduced);
  free(corr_mom);
  prop1.destroyTexObject(texProp1);
  prop2.destroyTexObject(texProp2);
}

#define N_BARYONS 10
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
contractBaryons(QKXTM_Propagator_Kepler<Float> &prop1,
		QKXTM_Propagator_Kepler<Float> &prop2, 
		char *filename_out, int isource){
  
  errorQuda("contractBaryons: This version of the function is obsolete. Cannot guarantee correct results. Please call the overloaded-updated version of this function with the corresponding list of arguments.\n");
  
  cudaTextureObject_t texProp1, texProp2;
  prop1.createTexObject(&texProp1);
  prop2.createTexObject(&texProp2);

  if( typeid(Float) == typeid(float))  
    printfQuda("contractBaryons: Will perform in single precision\n");
  if( typeid(Float) == typeid(double)) 
    printfQuda("contractBaryons: Will perform in double precision\n");

  Float (*corr_mom_local)[2][N_BARYONS][4][4] = (Float(*)[2][N_BARYONS][4][4])calloc(GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,sizeof(Float));
  Float (*corr_mom_local_reduced)[2][N_BARYONS][4][4] = (Float(*)[2][N_BARYONS][4][4]) calloc(GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,sizeof(Float));
  Float (*corr_mom)[2][N_BARYONS][4][4] = (Float(*)[2][N_BARYONS][4][4]) calloc(GK_totalL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,sizeof(Float));
  
  if( corr_mom_local == NULL || 
      corr_mom_local_reduced == NULL || 
      corr_mom == NULL )
    errorQuda("Error problem to allocate memory");

  for(int it = 0 ; it < GK_localL[3] ; it++){
    run_contractBaryons(texProp1,texProp2,(void*) corr_mom_local,it,isource,sizeof(Float),MOMENTUM_SPACE);
  }

  int error;

  if( typeid(Float) == typeid(float) ){
    MPI_Reduce(corr_mom_local,corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_FLOAT,MPI_SUM,0, GK_spaceComm);
    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_FLOAT,corr_mom,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_FLOAT,0,GK_timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
  }
  else{
    MPI_Reduce(corr_mom_local,corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_DOUBLE,MPI_SUM,0, GK_spaceComm);
    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_DOUBLE,corr_mom,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_DOUBLE,0,GK_timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
  }

  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out = fopen(filename_out,"w");
    if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
    for(int ip = 0 ; ip < N_BARYONS ; ip++)
    for(int it = 0 ; it < GK_totalL[3] ; it++)
    for(int imom = 0 ; imom < GK_Nmoms ; imom++)
    for(int gamma = 0 ; gamma < 4 ; gamma++)
    for(int gammap = 0 ; gammap < 4 ; gammap++){
      int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
      int sign = (it+GK_sourcePosition[isource][3]) >= GK_totalL[3] ? -1 : +1;
      fprintf(ptr_out,"%d \t %d \t %+d %+d %+d \t %d %d \t %+e %+e \t %+e %+e\n",ip,it,GK_moms[imom][0],GK_moms[imom][1],GK_moms[imom][2],gamma,gammap,
	      sign*corr_mom[it_shift*GK_Nmoms*2+imom*2+0][0][ip][gamma][gammap], sign*corr_mom[it_shift*GK_Nmoms*2+imom*2+1][0][ip][gamma][gammap], sign*corr_mom[it_shift*GK_Nmoms*2+imom*2+0][1][ip][gamma][gammap], sign*corr_mom[it_shift*GK_Nmoms*2+imom*2+1][1][ip][gamma][gammap]);
    }
    fclose(ptr_out);
  }

  free(corr_mom_local);
  free(corr_mom_local_reduced);
  free(corr_mom);
  prop1.destroyTexObject(texProp1);
  prop2.destroyTexObject(texProp2);
}

//---------------------------------------//

template<typename Float>
void QKXTM_Contraction_Kepler<Float>::writeTwopBaryonsHDF5(void *twopBaryons, char *filename, qudaQKXTMinfo_Kepler info, int isource){

  if(info.CorrSpace==MOMENTUM_SPACE){
    if(info.HighMomForm){
      writeTwopBaryonsHDF5_MomSpace_HighMomForm((void*) twopBaryons, filename, info, isource);
    }
    else{
      writeTwopBaryonsHDF5_MomSpace((void*) twopBaryons, filename, info, isource);
    }
  }
  else if(info.CorrSpace==POSITION_SPACE) writeTwopBaryonsHDF5_PosSpace((void*) twopBaryons, filename, info, isource);
  else errorQuda("writeTwopBaryonsHDF5: Unsupported value for info.CorrSpace! Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

}

//-C.K. - New function to write the baryons two-point function in 
// HDF5 format, position-space
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
writeTwopBaryonsHDF5_PosSpace(void *twopBaryons, 
			      char *filename, 
			      qudaQKXTMinfo_Kepler info, 
			      int isource){

  if(info.CorrSpace!=POSITION_SPACE) errorQuda("writeTwopBaryonsHDF5_PosSpace: Support for writing the Baryon two-point function only in position-space!\n");

  hid_t DATATYPE_H5;
  if( typeid(Float) == typeid(float) ){
    DATATYPE_H5 = H5T_NATIVE_FLOAT;
    printfQuda("writeTwopBaryonsHDF5_PosSpace: Will write in single precision\n");
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE_H5 = H5T_NATIVE_DOUBLE;
    printfQuda("writeTwopBaryonsHDF5_PosSpace: Will write in double precision\n");
  }

  Float *writeTwopBuf;

  int Sdim = 7;
  int Sel = 16;
  int pc[4];
  int tL[4];
  int lL[4];
  for(int i=0;i<4;i++){
    pc[i] = comm_coords(default_topo)[i];
    tL[i] = GK_totalL[i];
    lL[i] = GK_localL[i];
  }
  int lV = GK_localVolume;
  // Size of the dataspace -> #Baryons, volume, spin, re-im
  hsize_t dims[7]  = {2,tL[3],tL[2],tL[1],tL[0],Sel,2};
  // Dimensions of the "local" dataspace, for each rank
  hsize_t ldims[7] = {2,lL[3],lL[2],lL[1],lL[0],Sel,2}; 
  // start position for each rank
  hsize_t start[7] = {0,pc[3]*lL[3],pc[2]*lL[2],pc[1]*lL[1],pc[0]*lL[0],0,0};


  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
  H5Pclose(fapl_id);

  char *group1_tag;
  asprintf(&group1_tag,"conf_%04d",info.traj);
  hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);

  char *group2_tag;
  asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);

  /* Attribute writing */
  //- Source position
  char *src_pos;
  asprintf(&src_pos," [x, y, z, t] = [%02d, %02d, %02d, %02d]\0",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  hid_t attrdat_id = H5Screate(H5S_SCALAR);
  hid_t type_id = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id, strlen(src_pos));
  hid_t attr_id = H5Acreate2(group2_id, "source-position", 
			     type_id, attrdat_id, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id, type_id, src_pos);
  H5Aclose(attr_id);
  H5Tclose(type_id);
  H5Sclose(attrdat_id);

  //- Index identification-ordering, precision
  char *corr_info;
  asprintf(&corr_info,"Position-space baryon 2pt-correlator\nIndex Order: [flav, t, z, y, x, spin, real/imag]\nSpin-index order: Row-major\nPrecision: %s\0",(typeid(Float) == typeid(float)) ? "single" : "double");
  hid_t attrdat_id_2 = H5Screate(H5S_SCALAR);
  hid_t type_id_2 = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id_2, strlen(corr_info));
  hid_t attr_id_2 = H5Acreate2(file_id, "Correlator-info", 
			       type_id_2, attrdat_id_2, 
			       H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id_2, type_id_2, corr_info);
  H5Aclose(attr_id_2);
  H5Tclose(type_id_2);
  H5Sclose(attrdat_id_2);
  //------------------------------------------------------------

  for(int bar=0;bar<N_BARYONS;bar++){
    char *group3_tag;
    asprintf(&group3_tag,"%s",info.baryon_type[bar]);
    hid_t group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);

    hid_t filespace  = H5Screate_simple(Sdim, dims,  NULL);
    hid_t subspace   = H5Screate_simple(Sdim, ldims, NULL);
    hid_t dataset_id = H5Dcreate(group3_id, "twop-baryon", 
				 DATATYPE_H5, filespace, H5P_DEFAULT, 
				 H5P_DEFAULT, H5P_DEFAULT);
    filespace = H5Dget_space(dataset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, ldims, NULL);
    hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    writeTwopBuf = &(((Float*)twopBaryons)[2*Sel*lV*2*bar]);

    herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, 
			     filespace, plist_id, writeTwopBuf);
    if(status<0) errorQuda("writeTwopBaryonsHDF5_PosSpace: Unsuccessful writing of the dataset. Exiting\n");

    H5Dclose(dataset_id);
    H5Pclose(plist_id);
    H5Sclose(subspace);
    H5Sclose(filespace);
    H5Gclose(group3_id);
  }

  H5Gclose(group2_id);
  H5Gclose(group1_id);
  H5Fclose(file_id);

}

//-C.K. - New function to write the baryons two-point function in 
//HDF5 format, momentum-space
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
writeTwopBaryonsHDF5_MomSpace(void *twopBaryons, 
			      char *filename, 
			      qudaQKXTMinfo_Kepler info, 
			      int isource){

  if(info.CorrSpace!=MOMENTUM_SPACE || info.HighMomForm) errorQuda("writeTwopBaryonsHDF5_MomSpace: Supports writing the Baryon two-point function only in momentum-space and for NOT High-Momenta Form!\n");

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){

    hid_t DATATYPE_H5;
    if( typeid(Float) == typeid(float) ){
      DATATYPE_H5 = H5T_NATIVE_FLOAT;
      printfQuda("writeTwopBaryonsHDF5_MomSpace: Will write in single precision\n");
    }
    if( typeid(Float) == typeid(double)){
      DATATYPE_H5 = H5T_NATIVE_DOUBLE;
      printfQuda("writeTwopBaryonsHDF5_MomSpace: Will write in double precision\n");
    }

    int t_src = GK_sourcePosition[isource][3];
    int Lt = GK_localL[3];
    int T  = GK_totalL[3];

    int src_rank = t_src/Lt;
    int sink_rank = ((t_src-1)%T)/Lt;
    int h = Lt - t_src%Lt;
    int tail = t_src%Lt;

    Float *writeTwopBuf;

    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, GK_timeComm, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    char *group1_tag;
    asprintf(&group1_tag,"conf_%04d",info.traj);
    hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);

    char *group2_tag;
    asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	     GK_sourcePosition[isource][0],
	     GK_sourcePosition[isource][1],
	     GK_sourcePosition[isource][2],
	     GK_sourcePosition[isource][3]);
    hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);

    hid_t group3_id;
    hid_t group4_id;

    hsize_t dims[3] = {T,16,2}; // Size of the dataspace

    //-Determine the ldims for each rank (tail not taken into account)
    hsize_t ldims[3];
    ldims[1] = dims[1];
    ldims[2] = dims[2];
    if(GK_timeRank==src_rank) ldims[0] = h;
    else ldims[0] = Lt;

    //-Determine the start position for each rank
    hsize_t start[3];
    // if src_rank = sink_rank then this is the same
    if(GK_timeRank==src_rank) start[0] = 0; 
    else{
      int offs;
      for(offs=0;offs<GK_nProc[3];offs++){
	if( GK_timeRank == ((src_rank+offs)%GK_nProc[3]) ) break;
      }
      offs--;
      start[0] = h + offs*Lt;
    }
    start[1] = 0; //
    start[2] = 0; //-These are common among all ranks

    for(int bar=0;bar<N_BARYONS;bar++){
      char *group3_tag;
      asprintf(&group3_tag,"%s",info.baryon_type[bar]);
      group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, 
			    H5P_DEFAULT, H5P_DEFAULT);

      for(int imom=0;imom<GK_Nmoms;imom++){
	char *group4_tag;
	asprintf(&group4_tag,"mom_xyz_%+d_%+d_%+d",
		 GK_moms[imom][0],
		 GK_moms[imom][1],
		 GK_moms[imom][2]);
	group4_id = H5Gcreate(group3_id, group4_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);
	
	hid_t filespace  = H5Screate_simple(3, dims, NULL);
	hid_t subspace   = H5Screate_simple(3, ldims, NULL);

	for(int ip=0;ip<2;ip++){
	  char *dset_tag;
	  asprintf(&dset_tag,"twop_baryon_%d",ip+1);

	  hid_t dataset_id = H5Dcreate(group4_id, dset_tag, DATATYPE_H5, 
				       filespace, H5P_DEFAULT, H5P_DEFAULT, 
				       H5P_DEFAULT);
	  filespace = H5Dget_space(dataset_id);
	  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, 
			      NULL, ldims, NULL);
	  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	  if(GK_timeRank==src_rank) writeTwopBuf = &(((Float*)twopBaryons)[2*16*tail + 2*16*Lt*imom + 2*16*Lt*GK_Nmoms*bar + 2*16*Lt*GK_Nmoms*N_BARYONS*ip]);
	  else writeTwopBuf = &(((Float*)twopBaryons)[2*16*Lt*imom + 2*16*Lt*GK_Nmoms*bar + 2*16*Lt*GK_Nmoms*N_BARYONS*ip]);
	
	  herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace,
				   filespace, plist_id, writeTwopBuf);
	  
	  H5Dclose(dataset_id);
	  H5Pclose(plist_id);
	}//-ip
	H5Sclose(subspace);
	H5Sclose(filespace);
	H5Gclose(group4_id);
      }//-imom
      H5Gclose(group3_id);
    }//-bar

    H5Gclose(group2_id);
    H5Gclose(group1_id);
    H5Fclose(file_id);

    //-Write the tail, sink_ranks's task
    if(tail!=0 && GK_timeRank==sink_rank){ 
      Float *tailBuf;

      hid_t file_idt = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

      ldims[0] = tail;
      ldims[1] = 16;
      ldims[2] = 2;
      start[0] = T-tail;
      start[1] = 0;
      start[2] = 0;

      for(int bar=0;bar<N_BARYONS;bar++){
	for(int imom=0;imom<GK_Nmoms;imom++){
	  char *group_tag;
	  asprintf(&group_tag,"conf_%04d/sx%02dsy%02dsz%02dst%02d/%s/mom_xyz_%+d_%+d_%+d",
		   info.traj,
		   GK_sourcePosition[isource][0],
		   GK_sourcePosition[isource][1],
		   GK_sourcePosition[isource][2],
		   GK_sourcePosition[isource][3],
		   info.baryon_type[bar],
		   GK_moms[imom][0],
		   GK_moms[imom][1],
		   GK_moms[imom][2]);  
	  hid_t group_id = H5Gopen(file_idt, group_tag, H5P_DEFAULT);
	  
	  for(int ip=0;ip<2;ip++){
	    char *dset_tag;
	    asprintf(&dset_tag,"twop_baryon_%d",ip+1);

	    hid_t dset_id  = H5Dopen(group_id, dset_tag, H5P_DEFAULT);
	    hid_t mspace_id  = H5Screate_simple(3, ldims, NULL);
	    hid_t dspace_id = H5Dget_space(dset_id);

	    H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, start, 
				NULL, ldims, NULL);
	  
	    tailBuf = &(((Float*)twopBaryons)[2*16*Lt*imom + 2*16*Lt*GK_Nmoms*bar + 2*16*Lt*GK_Nmoms*N_BARYONS*ip]);

	    herr_t status = H5Dwrite(dset_id, DATATYPE_H5, mspace_id, dspace_id, H5P_DEFAULT, tailBuf);

	    H5Dclose(dset_id);
	    H5Sclose(mspace_id);
	    H5Sclose(dspace_id);
	  }
	  H5Gclose(group_id);
	}//-imom
      }//-bar

      H5Fclose(file_idt);
    }//-tail!=0
  }//-if GK_timeRank >=0 && GK_timeRank < GK_nProc[3]

}


//-C.K. - New function to write the baryons two-point function in HDF5 format, momentum-space, High-Momenta Form
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::writeTwopBaryonsHDF5_MomSpace_HighMomForm(void *twopBaryons, char *filename, qudaQKXTMinfo_Kepler info, int isource){

  if(info.CorrSpace!=MOMENTUM_SPACE || !info.HighMomForm) errorQuda("writeTwopBaryonsHDF5_MomSpace_HighMomForm: Supports writing the Baryon two-point function only in momentum-space and for HighMomForm!\n");

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){

    hid_t DATATYPE_H5;
    if( typeid(Float) == typeid(float) ){
      DATATYPE_H5 = H5T_NATIVE_FLOAT;
      printfQuda("writeTwopBaryonsHDF5_MomSpace: Will write in single precision\n");
    }
    if( typeid(Float) == typeid(double)){
      DATATYPE_H5 = H5T_NATIVE_DOUBLE;
      printfQuda("writeTwopBaryonsHDF5_MomSpace: Will write in double precision\n");
    }

    int t_src = GK_sourcePosition[isource][3];
    int Lt = GK_localL[3];
    int T  = GK_totalL[3];
    int Nmoms = GK_Nmoms;

    int src_rank = t_src/Lt;
    int sink_rank = ((t_src-1)%T)/Lt;
    int h = Lt - t_src%Lt;
    int tail = t_src%Lt;

    Float *writeTwopBuf;

    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, GK_timeComm, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    char *group1_tag;
    asprintf(&group1_tag,"conf_%04d",info.traj);
    hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    char *group2_tag;
    asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",GK_sourcePosition[isource][0],GK_sourcePosition[isource][1],GK_sourcePosition[isource][2],GK_sourcePosition[isource][3]);
    hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hid_t group3_id;

    hsize_t dims[4] = {T, Nmoms ,16 ,2}; // Size of the dataspace

    //-Determine the ldims for each rank (tail not taken into account)
    hsize_t ldims[4];
    ldims[1] = dims[1]; //
    ldims[2] = dims[2]; //-These are common among all ranks
    ldims[3] = dims[3]; //
    if(GK_timeRank==src_rank) ldims[0] = h; // local-dimension size
    else ldims[0] = Lt;                     // for time for each rank

    //-Determine the start position for each rank
    hsize_t start[4];
    start[1] = 0; //
    start[2] = 0; // -These are common among all ranks
    start[3] = 0; //
    if(GK_timeRank==src_rank) start[0] = 0; // if src_rank == sink_rank then this holds for the sink rank as well
    else{
      int offs;
      for(offs=0;offs<GK_nProc[3];offs++){
        if( GK_timeRank == ((src_rank+offs)%GK_nProc[3]) ) break;
      }
      offs--;
      start[0] = h + offs*Lt;
    }

    for(int bar=0;bar<N_BARYONS;bar++){
      char *group3_tag;
      asprintf(&group3_tag,"%s",info.baryon_type[bar]);
      group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      
      hid_t filespace  = H5Screate_simple(4, dims, NULL);
      hid_t subspace   = H5Screate_simple(4, ldims, NULL);

      for(int ip=0;ip<2;ip++){
        char *dset_tag;
        asprintf(&dset_tag,"twop_baryon_%d",ip+1);
        
        hid_t dataset_id = H5Dcreate(group3_id, dset_tag, DATATYPE_H5, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        filespace = H5Dget_space(dataset_id);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, ldims, NULL);
        hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        
        if(GK_timeRank==src_rank) writeTwopBuf = &(((Float*)twopBaryons)[2*16*Nmoms*tail + 2*16*Nmoms*Lt*bar + 2*16*Nmoms*Lt*N_BARYONS*ip]);
        else writeTwopBuf = &(((Float*)twopBaryons)[2*16*Nmoms*Lt*bar + 2*16*Nmoms*Lt*N_BARYONS*ip]);
        
        herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, filespace, plist_id, writeTwopBuf);
        
        H5Dclose(dataset_id);
        H5Pclose(plist_id);
      }//-ip
      H5Sclose(subspace);
      H5Sclose(filespace);
      H5Gclose(group3_id);
    }//-bar

    H5Gclose(group2_id);
    H5Gclose(group1_id);
    H5Fclose(file_id);

    //-Write the tail, sink_ranks's task
    if(tail!=0 && GK_timeRank==sink_rank){ 
      Float *tailBuf;

      hid_t file_idt = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

      ldims[0] = tail;
      ldims[1] = Nmoms;
      ldims[2] = 16;
      ldims[3] = 2;

      start[0] = T-tail;
      start[1] = 0;
      start[2] = 0;
      start[3] = 0;

      for(int bar=0;bar<N_BARYONS;bar++){
        char *group_tag;
        asprintf(&group_tag,"conf_%04d/sx%02dsy%02dsz%02dst%02d/%s",info.traj,GK_sourcePosition[isource][0],GK_sourcePosition[isource][1],GK_sourcePosition[isource][2],GK_sourcePosition[isource][3],info.baryon_type[bar]);  
        hid_t group_id = H5Gopen(file_idt, group_tag, H5P_DEFAULT);

        for(int ip=0;ip<2;ip++){
          char *dset_tag;
          asprintf(&dset_tag,"twop_baryon_%d",ip+1);

          hid_t dset_id   = H5Dopen(group_id, dset_tag, H5P_DEFAULT);
          hid_t mspace_id = H5Screate_simple(4, ldims, NULL);
          hid_t dspace_id = H5Dget_space(dset_id);
          
          H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, start, NULL, ldims, NULL);
          
          tailBuf = &(((Float*)twopBaryons)[2*16*Nmoms*Lt*bar + 2*16*Nmoms*Lt*N_BARYONS*ip]);
          
          herr_t status = H5Dwrite(dset_id, DATATYPE_H5, mspace_id, dspace_id, H5P_DEFAULT, tailBuf);
          
          H5Dclose(dset_id);
          H5Sclose(mspace_id);
          H5Sclose(dspace_id);
        }
        H5Gclose(group_id);
      }//-bar

      H5Fclose(file_idt);
    }//-tail!=0

    //-Write the momenta in a separate dataset (including some attributes)
    if(GK_timeRank==sink_rank){
      hid_t file_idt   = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

      char *cNmoms, *cQsq,*corr_info,*ens_info;
      asprintf(&cNmoms,"%d\0",GK_Nmoms);
      asprintf(&cQsq,"%d\0",info.Q_sq);
      asprintf(&corr_info,
	       "Momentum-space baryon 2pt-correlator\nQuark field basis: Physical\nIndex Order: [t, mom-index, spin, real/imag]\nSpin-index order: Row-major\nPrecision: %s\nInversion tolerance = %e\0",
	       (typeid(Float) == typeid(float)) ? "single" : "double", info.inv_tol);
      asprintf(&ens_info,"kappa = %10.8f\nmu = %8.6f\nCsw = %8.6f\0",info.kappa, info.mu, info.csw);
      hid_t attrdat_id1 = H5Screate(H5S_SCALAR);
      hid_t attrdat_id2 = H5Screate(H5S_SCALAR);
      hid_t attrdat_id3 = H5Screate(H5S_SCALAR);
      hid_t attrdat_id4 = H5Screate(H5S_SCALAR);
      hid_t type_id1 = H5Tcopy(H5T_C_S1);
      hid_t type_id2 = H5Tcopy(H5T_C_S1);
      hid_t type_id3 = H5Tcopy(H5T_C_S1);
      hid_t type_id4 = H5Tcopy(H5T_C_S1);
      H5Tset_size(type_id1, strlen(cNmoms));
      H5Tset_size(type_id2, strlen(cQsq));
      H5Tset_size(type_id3, strlen(corr_info));
      H5Tset_size(type_id4, strlen(ens_info));
      hid_t attr_id1 = H5Acreate2(file_idt, "Nmoms", type_id1, attrdat_id1, H5P_DEFAULT, H5P_DEFAULT);
      hid_t attr_id2 = H5Acreate2(file_idt, "Qsq"  , type_id2, attrdat_id2, H5P_DEFAULT, H5P_DEFAULT);
      hid_t attr_id3 = H5Acreate2(file_idt, "Correlator-info", type_id3, attrdat_id3, H5P_DEFAULT, H5P_DEFAULT);
      hid_t attr_id4 = H5Acreate2(file_idt, "Ensemble-info", type_id4, attrdat_id4, H5P_DEFAULT, H5P_DEFAULT);
      H5Awrite(attr_id1, type_id1, cNmoms);
      H5Awrite(attr_id2, type_id2, cQsq);
      H5Awrite(attr_id3, type_id3, corr_info);
      H5Awrite(attr_id4, type_id4, ens_info);
      H5Aclose(attr_id1);
      H5Aclose(attr_id2);
      H5Aclose(attr_id3);
      H5Aclose(attr_id4);
      H5Tclose(type_id1);
      H5Tclose(type_id2);
      H5Tclose(type_id3);
      H5Tclose(type_id4);
      H5Sclose(attrdat_id1);
      H5Sclose(attrdat_id2);
      H5Sclose(attrdat_id3);
      H5Sclose(attrdat_id4);

      hid_t MOMTYPE_H5 = H5T_NATIVE_INT;
      char *dset_tag;
      asprintf(&dset_tag,"Momenta_list_xyz");

      hsize_t Mdims[2] = {(hsize_t)Nmoms,3};
      hid_t filespace  = H5Screate_simple(2, Mdims, NULL);
      hid_t dataset_id = H5Dcreate(file_idt, dset_tag, MOMTYPE_H5, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      int *Moms_H5 = (int*) malloc(GK_Nmoms*3*sizeof(int));
      for(int im=0;im<GK_Nmoms;im++){
        for(int d=0;d<3;d++) Moms_H5[d + 3*im] = GK_moms[im][d];
      }

      herr_t status = H5Dwrite(dataset_id, MOMTYPE_H5, H5S_ALL, filespace, H5P_DEFAULT, Moms_H5);

      H5Dclose(dataset_id);
      H5Sclose(filespace);
      H5Fclose(file_idt);

      free(Moms_H5);
    }//-if GK_timeRank==0

  }//-if GK_timeRank >=0 && GK_timeRank < GK_nProc[3]

}


//-C.K. - New function to copy the baryon two-point functions into write 
// Buffers for writing in HDF5 format
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
copyTwopBaryonsToHDF5_Buf(void *Twop_baryons_HDF5, 
			  void *corrBaryons, 
			  int isource, 
			  CORR_SPACE CorrSpace, bool HighMomForm){

  int Lt = GK_localL[3];
  int SpVol = GK_localVolume/Lt;
  int t_src = GK_sourcePosition[isource][3];
  int Nmoms = GK_Nmoms;

  if(CorrSpace==MOMENTUM_SPACE){

    if(HighMomForm){
      if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){      
        
        for(int ip=0;ip<2;ip++){
          for(int bar=0;bar<N_BARYONS;bar++){
            for(int imom=0;imom<GK_Nmoms;imom++){
              for(int it=0;it<Lt;it++){
                int t_glob = GK_timeRank*Lt+it;
                int sign = t_glob < t_src ? -1 : +1;
                for(int ga=0;ga<4;ga++){
                  for(int gap=0;gap<4;gap++){
                    int im=gap+4*ga;
                    ((Float*)Twop_baryons_HDF5)[0 + 2*im + 2*16*imom + 2*16*Nmoms*it + 2*16*Nmoms*Lt*bar + 2*16*Nmoms*Lt*N_BARYONS*ip] = sign*((Float(*)[2][N_BARYONS][4][4])corrBaryons)[0 + 2*imom + 2*Nmoms*it][ip][bar][ga][gap];
                    ((Float*)Twop_baryons_HDF5)[1 + 2*im + 2*16*imom + 2*16*Nmoms*it + 2*16*Nmoms*Lt*bar + 2*16*Nmoms*Lt*N_BARYONS*ip] = sign*((Float(*)[2][N_BARYONS][4][4])corrBaryons)[1 + 2*imom + 2*Nmoms*it][ip][bar][ga][gap];
                  }}}}}
        }//-ip
        
      }//-if GK_timeRank
    }//-if HighMomForm
    else{
      if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){      
        
        for(int ip=0;ip<2;ip++){
          for(int bar=0;bar<N_BARYONS;bar++){
            for(int imom=0;imom<GK_Nmoms;imom++){
              for(int it=0;it<Lt;it++){
                int t_glob = GK_timeRank*Lt+it;
                int sign = t_glob < t_src ? -1 : +1;
                for(int ga=0;ga<4;ga++){
                  for(int gap=0;gap<4;gap++){
                    int im=gap+4*ga;
                    ((Float*)Twop_baryons_HDF5)[0 + 2*im + 2*16*it + 2*16*Lt*imom + 2*16*Lt*Nmoms*bar + 2*16*Lt*Nmoms*N_BARYONS*ip] = sign*((Float(*)[2][N_BARYONS][4][4])corrBaryons)[0 + 2*imom + 2*Nmoms*it][ip][bar][ga][gap];
                    ((Float*)Twop_baryons_HDF5)[1 + 2*im + 2*16*it + 2*16*Lt*imom + 2*16*Lt*Nmoms*bar + 2*16*Lt*Nmoms*N_BARYONS*ip] = sign*((Float(*)[2][N_BARYONS][4][4])corrBaryons)[1 + 2*imom + 2*Nmoms*it][ip][bar][ga][gap];
                  }}}}}
        }//-ip
        
      }//-if GK_timeRank
    }//-else HighMomForm

  }//-if CorrSpace
  else if(CorrSpace==POSITION_SPACE){

    for(int ip=0;ip<2;ip++){
      for(int bar=0;bar<N_BARYONS;bar++){
        for(int ga=0;ga<4;ga++){
          for(int gap=0;gap<4;gap++){
            int im=gap+4*ga;
            for(int it=0;it<Lt;it++){
              int t_glob = comm_coords(default_topo)[3]*Lt+it;
              int sign = t_glob < t_src ? -1 : +1;
              for(int sv=0;sv<SpVol;sv++){
                ((Float*)Twop_baryons_HDF5)[0 + 2*im + 2*16*sv + 2*16*SpVol*it + 2*16*SpVol*Lt*ip + 2*16*SpVol*Lt*2*bar] = sign*((Float(*)[2][N_BARYONS][4][4])corrBaryons)[0 + 2*sv + 2*SpVol*it][ip][bar][ga][gap];
                ((Float*)Twop_baryons_HDF5)[1 + 2*im + 2*16*sv + 2*16*SpVol*it + 2*16*SpVol*Lt*ip + 2*16*SpVol*Lt*2*bar] = sign*((Float(*)[2][N_BARYONS][4][4])corrBaryons)[1 + 2*sv + 2*SpVol*it][ip][bar][ga][gap];
              }}}}}
    }//-ip

  }//-else CorrSpace

}


//-C.K. New function to write the baryons two-point function in ASCII format
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
writeTwopBaryons_ASCII(void *corrBaryons, 
		       char *filename_out, 
		       int isource, 
		       CORR_SPACE CorrSpace){

  if(CorrSpace!=MOMENTUM_SPACE) errorQuda("writeTwopBaryons_ASCII: Supports writing only in momentum-space!\n");

  Float (*GLcorrBaryons)[2][N_BARYONS][4][4] = (Float(*)[2][N_BARYONS][4][4]) calloc(GK_totalL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,sizeof(Float));
  if( GLcorrBaryons == NULL )errorQuda("writeTwopBaryons_ASCII: Cannot allocate memory for Baryon two-point function buffer.");

  MPI_Datatype DATATYPE = -1;
  if( typeid(Float) == typeid(float)){
    DATATYPE = MPI_FLOAT;
    printfQuda("writeTwopBaryons_ASCII: Will write in single precision\n");
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE = MPI_DOUBLE;
    printfQuda("writeTwopBaryons_ASCII: Will write in double precision\n");
  }

  int error;
  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
    error = MPI_Gather((Float*)corrBaryons,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,
		       DATATYPE,GLcorrBaryons,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,
		       DATATYPE,0,GK_timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
  }

  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out = fopen(filename_out,"w");
    if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
    for(int ip = 0 ; ip < N_BARYONS ; ip++)
    for(int it = 0 ; it < GK_totalL[3] ; it++)
    for(int imom = 0 ; imom < GK_Nmoms ; imom++)
    for(int gamma = 0 ; gamma < 4 ; gamma++)
    for(int gammap = 0 ; gammap < 4 ; gammap++){
      int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
      int sign = (it+GK_sourcePosition[isource][3]) 
	>= GK_totalL[3] ? -1 : +1;
      fprintf(ptr_out,
	      "%d \t %d \t %+d %+d %+d \t %d %d \t %+e %+e \t %+e %+e\n",
	      ip,it,
	      GK_moms[imom][0],
	      GK_moms[imom][1],
	      GK_moms[imom][2],gamma,gammap,
     sign*GLcorrBaryons[it_shift*GK_Nmoms*2+imom*2+0][0][ip][gamma][gammap], 
     sign*GLcorrBaryons[it_shift*GK_Nmoms*2+imom*2+1][0][ip][gamma][gammap],
     sign*GLcorrBaryons[it_shift*GK_Nmoms*2+imom*2+0][1][ip][gamma][gammap], 
     sign*GLcorrBaryons[it_shift*GK_Nmoms*2+imom*2+1][1][ip][gamma][gammap]);
    }
    fclose(ptr_out);
  }
  
  free(GLcorrBaryons);
}

//-C.K. Overloaded function to perform the baryon contractions without writing the data
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
contractBaryons(QKXTM_Propagator_Kepler<Float> &prop1,
		QKXTM_Propagator_Kepler<Float> &prop2, 
		void *corrBaryons, int isource, 
		CORR_SPACE CorrSpace){
  cudaTextureObject_t texProp1, texProp2;
  prop1.createTexObject(&texProp1);
  prop2.createTexObject(&texProp2);

  if( typeid(Float) == typeid(float))  
    printfQuda("contractBaryons: Will perform in single precision\n");
  if( typeid(Float) == typeid(double)) 
    printfQuda("contractBaryons: Will perform in double precision\n");
  
  if(CorrSpace==POSITION_SPACE){
    for(int it = 0 ; it < GK_localL[3] ; it++) 
      run_contractBaryons(texProp1,texProp2,
			  (void*) corrBaryons,it,
			  isource,sizeof(Float),CorrSpace);
  }
  else if(CorrSpace==MOMENTUM_SPACE){
    Float (*corrBaryons_local)[2][N_BARYONS][4][4] = (Float(*)[2][N_BARYONS][4][4]) calloc(GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,sizeof(Float));
    if( corrBaryons_local == NULL ) errorQuda("contractBaryons: Cannot allocate memory for Baryon two-point function contract buffer.\n");

    for(int it = 0 ; it < GK_localL[3] ; it++) 
      run_contractBaryons(texProp1,texProp2,
			  (void*) corrBaryons_local,it,
			  isource,sizeof(Float),CorrSpace);
    
    MPI_Datatype DATATYPE = -1;
    if( typeid(Float) == typeid(float))  DATATYPE = MPI_FLOAT;
    if( typeid(Float) == typeid(double)) DATATYPE = MPI_DOUBLE;

    MPI_Reduce(corrBaryons_local, (Float*) corrBaryons,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,DATATYPE,MPI_SUM,0, GK_spaceComm);

    free(corrBaryons_local);
  }
  else errorQuda("contractBaryons: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

  prop1.destroyTexObject(texProp1);
  prop2.destroyTexObject(texProp2);
}

//--------------------------------------------------------//
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
writeTwopMesonsHDF5(void *twopMesons, 
		    char *filename, 
		    qudaQKXTMinfo_Kepler info, 
		    int isource){

  if(info.CorrSpace==MOMENTUM_SPACE){
    if(info.HighMomForm){
      writeTwopMesonsHDF5_MomSpace_HighMomForm((void*)twopMesons, filename, info, isource);
    }
    else{
      writeTwopMesonsHDF5_MomSpace((void*)twopMesons, filename, info, isource);
    }
  }
  else if(info.CorrSpace==POSITION_SPACE) writeTwopMesonsHDF5_PosSpace((void*)twopMesons, filename, info, isource);
  else errorQuda("writeTwopMesonsHDF5: Unsupported value for info.CorrSpace! Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

}

//-C.K. - New function to write the mesons two-point function in HDF5 format, position-space
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
writeTwopMesonsHDF5_PosSpace(void *twopMesons, 
			     char *filename, 
			     qudaQKXTMinfo_Kepler info, 
			     int isource){
  
  if(info.CorrSpace!=POSITION_SPACE) errorQuda("writeTwopMesonsHDF5_PosSpace: Support for writing the Meson two-point function only in position-space!\n");

  hid_t DATATYPE_H5;
  if( typeid(Float) == typeid(float) ){
    DATATYPE_H5 = H5T_NATIVE_FLOAT;
    printfQuda("writeTwopMesonsHDF5_PosSpace: Will write in single precision\n");
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE_H5 = H5T_NATIVE_DOUBLE;
    printfQuda("writeTwopMesonsHDF5_PosSpace: Will write in double precision\n");
  }

  Float *writeTwopBuf;

  int Sdim = 6;
  int pc[4];
  int tL[4];
  int lL[4];
  for(int i=0;i<4;i++){
    pc[i] = comm_coords(default_topo)[i];
    tL[i] = GK_totalL[i];
    lL[i] = GK_localL[i];
  }
  int lV = GK_localVolume;

  // Size of the dataspace -> #Baryons, volume, spin, re-im
  hsize_t dims[6]  = {2,tL[3],tL[2],tL[1],tL[0],2}; 
  // Dimensions of the "local" dataspace, for each rank
  hsize_t ldims[6] = {2,lL[3],lL[2],lL[1],lL[0],2}; 
  // start position for each rank
  hsize_t start[6] = {0,pc[3]*lL[3],pc[2]*lL[2],pc[1]*lL[1],pc[0]*lL[0],0};
  
  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
  H5Pclose(fapl_id);

  char *group1_tag;
  asprintf(&group1_tag,"conf_%04d",info.traj);
  hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);

  char *group2_tag;
  asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);

  /* Attribute writing */
  //- Source position
  char *src_pos;
  asprintf(&src_pos," [x, y, z, t] = [%02d, %02d, %02d, %02d]\0",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  hid_t attrdat_id = H5Screate(H5S_SCALAR);
  hid_t type_id = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id, strlen(src_pos));
  hid_t attr_id = H5Acreate2(group2_id, "source-position", type_id, 
			     attrdat_id, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id, type_id, src_pos);
  H5Aclose(attr_id);
  H5Tclose(type_id);
  H5Sclose(attrdat_id);

  //- Index identification-ordering, precision
  char *corr_info;
  asprintf(&corr_info,"Position-space meson 2pt-correlator\nIndex Order: [flav, t, z, y, x, real/imag]\nPrecision: %s\0",(typeid(Float) == typeid(float)) ? "single" : "double");
  hid_t attrdat_id_2 = H5Screate(H5S_SCALAR);
  hid_t type_id_2 = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id_2, strlen(corr_info));
  hid_t attr_id_2 = H5Acreate2(file_id, "Correlator-info", type_id_2, 
			       attrdat_id_2, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id_2, type_id_2, corr_info);
  H5Aclose(attr_id_2);
  H5Tclose(type_id_2);
  H5Sclose(attrdat_id_2);
  //------------------------------------------------------------

  for(int mes=0;mes<N_MESONS;mes++){
    char *group3_tag;
    asprintf(&group3_tag,"%s",info.meson_type[mes]);
    hid_t group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);

    hid_t filespace  = H5Screate_simple(Sdim, dims,  NULL);
    hid_t subspace   = H5Screate_simple(Sdim, ldims, NULL);
    hid_t dataset_id = H5Dcreate(group3_id, "twop-meson", DATATYPE_H5, 
				 filespace, H5P_DEFAULT, H5P_DEFAULT, 
				 H5P_DEFAULT);
    filespace = H5Dget_space(dataset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, ldims, NULL);
    hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    writeTwopBuf = &(((Float*)twopMesons)[2*lV*2*mes]);

    herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, filespace, 
			     plist_id, writeTwopBuf);
    if(status<0) errorQuda("writeTwopMesonsHDF5_PosSpace: Unsuccessful writing of the dataset. Exiting\n");

    H5Dclose(dataset_id);
    H5Pclose(plist_id);
    H5Sclose(subspace);
    H5Sclose(filespace);
    H5Gclose(group3_id);
  }

  H5Gclose(group2_id);
  H5Gclose(group1_id);
  H5Fclose(file_id);

}


//-C.K. - New function to write the mesons two-point function in 
// HDF5 format, momentum-space
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
writeTwopMesonsHDF5_MomSpace(void *twopMesons, 
			     char *filename, 
			     qudaQKXTMinfo_Kepler info, 
			     int isource){

  if(info.CorrSpace!=MOMENTUM_SPACE || info.HighMomForm) errorQuda("writeTwopMesonsHDF5_MomSpace: Supports writing the Meson two-point function only in momentum-space and for NOT High-Momenta Form!\n");

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){

    hid_t DATATYPE_H5;
    if( typeid(Float) == typeid(float) ){
      DATATYPE_H5 = H5T_NATIVE_FLOAT;
      printfQuda("writeTwopMesonsHDF5_MomSpace: Will write in single precision\n");
    }
    if( typeid(Float) == typeid(double)){
      DATATYPE_H5 = H5T_NATIVE_DOUBLE;
      printfQuda("writeTwopMesons_HDF5_MomSpace: Will write in double precision\n");
    }

    int t_src = GK_sourcePosition[isource][3];
    int Lt = GK_localL[3];
    int T  = GK_totalL[3];

    int src_rank = t_src/Lt;
    int sink_rank = ((t_src-1)%T)/Lt;
    int h = Lt - t_src%Lt;
    int tail = t_src%Lt;

    Float *writeTwopBuf;

    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, GK_timeComm, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    char *group1_tag;
    asprintf(&group1_tag,"conf_%04d",info.traj);
    hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);
    
    char *group2_tag;
    asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	     GK_sourcePosition[isource][0],
	     GK_sourcePosition[isource][1],
	     GK_sourcePosition[isource][2],
	     GK_sourcePosition[isource][3]);
    hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);

    hid_t group3_id;
    hid_t group4_id;

    hsize_t dims[2] = {T,2}; // Size of the dataspace

    //-Determine the ldims for each rank (tail not taken into account)
    hsize_t ldims[2];
    ldims[1] = dims[1];
    if(GK_timeRank==src_rank) ldims[0] = h;
    else ldims[0] = Lt;

    //-Determine the start position for each rank
    hsize_t start[2];
    // if src_rank = sink_rank then this is the same
    if(GK_timeRank==src_rank) start[0] = 0;
    else{
      int offs;
      for(offs=0;offs<GK_nProc[3];offs++){
	if( GK_timeRank == ((src_rank+offs)%GK_nProc[3]) ) break;
      }
      offs--;
      start[0] = h + offs*Lt;
    }
    start[1] = 0; //-This is common among all ranks

    for(int mes=0;mes<N_MESONS;mes++){
      char *group3_tag;
      asprintf(&group3_tag,"%s",info.meson_type[mes]);
      group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, 
			    H5P_DEFAULT, H5P_DEFAULT);

      for(int imom=0;imom<GK_Nmoms;imom++){
	char *group4_tag;
	asprintf(&group4_tag,"mom_xyz_%+d_%+d_%+d",
		 GK_moms[imom][0],
		 GK_moms[imom][1],
		 GK_moms[imom][2]);
	group4_id = H5Gcreate(group3_id, group4_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);
	
	hid_t filespace  = H5Screate_simple(2, dims, NULL);
	hid_t subspace   = H5Screate_simple(2, ldims, NULL);

	for(int ip=0;ip<2;ip++){
	  char *dset_tag;
	  asprintf(&dset_tag,"twop_meson_%d",ip+1);

	  hid_t dataset_id = H5Dcreate(group4_id, dset_tag, DATATYPE_H5, 
				       filespace, H5P_DEFAULT, H5P_DEFAULT, 
				       H5P_DEFAULT);
	  filespace = H5Dget_space(dataset_id);
	  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, 
			      NULL, ldims, NULL);
	  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	  if(GK_timeRank==src_rank) writeTwopBuf = &(((Float*)twopMesons)[2*tail + 2*Lt*imom + 2*Lt*GK_Nmoms*mes + 2*Lt*GK_Nmoms*N_MESONS*ip]);
	  else writeTwopBuf = &(((Float*)twopMesons)[2*Lt*imom + 2*Lt*GK_Nmoms*mes + 2*Lt*GK_Nmoms*N_MESONS*ip]);
	
	  herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, 
				   filespace, plist_id, writeTwopBuf);
	  
	  H5Dclose(dataset_id);
	  H5Pclose(plist_id);
	}//-ip
	H5Sclose(subspace);
	H5Sclose(filespace);
	H5Gclose(group4_id);
      }//-imom
      H5Gclose(group3_id);
    }//-mes

    H5Gclose(group2_id);
    H5Gclose(group1_id);
    H5Fclose(file_id);

    //-Write the tail, sink_ranks's task
    if(tail!=0 && GK_timeRank==sink_rank){ 
      Float *tailBuf;

      hid_t file_idt = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

      ldims[0] = tail;
      ldims[1] = 2;
      start[0] = T-tail;
      start[1] = 0;

      for(int mes=0;mes<N_MESONS;mes++){
	for(int imom=0;imom<GK_Nmoms;imom++){
	  char *group_tag;
	  asprintf(&group_tag,"conf_%04d/sx%02dsy%02dsz%02dst%02d/%s/mom_xyz_%+d_%+d_%+d",
		   info.traj,
		   GK_sourcePosition[isource][0],
		   GK_sourcePosition[isource][1],
		   GK_sourcePosition[isource][2],
		   GK_sourcePosition[isource][3],
		   info.meson_type[mes],
		   GK_moms[imom][0],
		   GK_moms[imom][1],
		   GK_moms[imom][2]);  
	  hid_t group_id = H5Gopen(file_idt, group_tag, H5P_DEFAULT);
	  
	  for(int ip=0;ip<2;ip++){
	    char *dset_tag;
	    asprintf(&dset_tag,"twop_meson_%d",ip+1);
	    
	    hid_t dset_id  = H5Dopen(group_id, dset_tag, H5P_DEFAULT);
	    hid_t mspace_id  = H5Screate_simple(2, ldims, NULL);
	    hid_t dspace_id = H5Dget_space(dset_id);
	    
	    H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, start, NULL, ldims, NULL);
	  
	    tailBuf = &(((Float*)twopMesons)[2*Lt*imom + 2*Lt*GK_Nmoms*mes + 2*Lt*GK_Nmoms*N_MESONS*ip]);

	    herr_t status = H5Dwrite(dset_id, DATATYPE_H5, mspace_id, dspace_id, H5P_DEFAULT, tailBuf);
	    
	    H5Dclose(dset_id);
	    H5Sclose(mspace_id);
	    H5Sclose(dspace_id);
	  }
	  H5Gclose(group_id);
	}//-imom
      }//-mes

      H5Fclose(file_idt);
    }//-tail!=0

  }//-if GK_timeRank >=0 && GK_timeRank < GK_nProc[3]

}


//-C.K. - New function to write the mesons two-point function in HDF5 format, momentum-space, High-Momenta Form
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::writeTwopMesonsHDF5_MomSpace_HighMomForm(void *twopMesons, char *filename, qudaQKXTMinfo_Kepler info, int isource){

  if(info.CorrSpace!=MOMENTUM_SPACE || !info.HighMomForm) errorQuda("writeTwopMesonsHDF5_MomSpace_HighMomForm: Supports writing the Meson two-point function only in momentum-space and for HighMomForm!\n");

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){

    hid_t DATATYPE_H5;
    if( typeid(Float) == typeid(float) ){
      DATATYPE_H5 = H5T_NATIVE_FLOAT;
      printfQuda("writeTwopMesonsHDF5_MomSpace: Will write in single precision\n");
    }
    if( typeid(Float) == typeid(double)){
      DATATYPE_H5 = H5T_NATIVE_DOUBLE;
      printfQuda("writeTwopMesons_HDF5_MomSpace: Will write in double precision\n");
    }

    int t_src = GK_sourcePosition[isource][3];
    int Lt = GK_localL[3];
    int T  = GK_totalL[3];
    int Nmoms = GK_Nmoms;

    int src_rank = t_src/Lt;
    int sink_rank = ((t_src-1)%T)/Lt;
    int h = Lt - t_src%Lt;
    int tail = t_src%Lt;

    Float *writeTwopBuf;

    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, GK_timeComm, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    char *group1_tag;
    asprintf(&group1_tag,"conf_%04d",info.traj);
    hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    char *group2_tag;
    asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",GK_sourcePosition[isource][0],GK_sourcePosition[isource][1],GK_sourcePosition[isource][2],GK_sourcePosition[isource][3]);
    hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hid_t group3_id;

    hsize_t dims[3] = {T,Nmoms,2}; // Size of the dataspace

    //-Determine the ldims for each rank (tail not taken into account)
    hsize_t ldims[3];
    ldims[1] = dims[1];
    ldims[2] = dims[2];
    if(GK_timeRank==src_rank) ldims[0] = h;
    else ldims[0] = Lt;

    //-Determine the start position for each rank
    hsize_t start[3];
    if(GK_timeRank==src_rank) start[0] = 0; // if src_rank = sink_rank then this is the same for both
    else{
      int offs;
      for(offs=0;offs<GK_nProc[3];offs++){
        if( GK_timeRank == ((src_rank+offs)%GK_nProc[3]) ) break;
      }
      offs--;
      start[0] = h + offs*Lt;
    }
    start[1] = 0; //-This is common among all ranks
    start[2] = 0; //

    for(int mes=0;mes<N_MESONS;mes++){
      char *group3_tag;
      asprintf(&group3_tag,"%s",info.meson_type[mes]);
      group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
      hid_t filespace  = H5Screate_simple(3, dims, NULL);
      hid_t subspace   = H5Screate_simple(3, ldims, NULL);
      
      for(int ip=0;ip<2;ip++){
        char *dset_tag;
        asprintf(&dset_tag,"twop_meson_%d",ip+1);
        
        hid_t dataset_id = H5Dcreate(group3_id, dset_tag, DATATYPE_H5, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        filespace = H5Dget_space(dataset_id);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, ldims, NULL);
        hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        
        if(GK_timeRank==src_rank) writeTwopBuf = &(((Float*)twopMesons)[2*Nmoms*tail + 2*Nmoms*Lt*mes + 2*Nmoms*Lt*N_MESONS*ip]);
        else writeTwopBuf = &(((Float*)twopMesons)[2*Nmoms*Lt*mes + 2*Nmoms*Lt*N_MESONS*ip]);
        
        herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, filespace, plist_id, writeTwopBuf);
        
        H5Dclose(dataset_id);
        H5Pclose(plist_id);
      }//-ip
      H5Sclose(subspace);
      H5Sclose(filespace);
      H5Gclose(group3_id);
    }//-mes

    H5Gclose(group2_id);
    H5Gclose(group1_id);
    H5Fclose(file_id);

    //-Write the tail, sink_ranks's task
    if(tail!=0 && GK_timeRank==sink_rank){ 
      Float *tailBuf;

      hid_t file_idt = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

      ldims[0] = tail;
      ldims[1] = Nmoms;
      ldims[2] = 2;
      start[0] = T-tail;
      start[1] = 0;
      start[2] = 0;

      for(int mes=0;mes<N_MESONS;mes++){
        char *group_tag;
        asprintf(&group_tag,"conf_%04d/sx%02dsy%02dsz%02dst%02d/%s",info.traj,GK_sourcePosition[isource][0],GK_sourcePosition[isource][1],GK_sourcePosition[isource][2],GK_sourcePosition[isource][3],info.meson_type[mes]);  
        hid_t group_id = H5Gopen(file_idt, group_tag, H5P_DEFAULT);
        
        for(int ip=0;ip<2;ip++){
          char *dset_tag;
          asprintf(&dset_tag,"twop_meson_%d",ip+1);
          
          hid_t dset_id  = H5Dopen(group_id, dset_tag, H5P_DEFAULT);
          hid_t mspace_id  = H5Screate_simple(3, ldims, NULL);
          hid_t dspace_id = H5Dget_space(dset_id);
          
          H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, start, NULL, ldims, NULL);
          
          tailBuf = &(((Float*)twopMesons)[2*Nmoms*Lt*mes + 2*Nmoms*Lt*N_MESONS*ip]);
          
          herr_t status = H5Dwrite(dset_id, DATATYPE_H5, mspace_id, dspace_id, H5P_DEFAULT, tailBuf);
          
          H5Dclose(dset_id);
          H5Sclose(mspace_id);
          H5Sclose(dspace_id);
        }
        H5Gclose(group_id);
      }//-mes

      H5Fclose(file_idt);
    }//-tail!=0

    //-Write the momenta in a separate dataset (including some attributes)
    if(GK_timeRank==sink_rank){
      hid_t file_idt   = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

      char *cNmoms, *cQsq,*corr_info,*ens_info;;
      asprintf(&cNmoms,"%d\0",GK_Nmoms);
      asprintf(&cQsq,"%d\0",info.Q_sq);
      asprintf(&corr_info,
               "Momentum-space meson 2pt-correlator\nQuark field basis: Physical\nIndex Order: [t, mom-index, real/imag]\nPrecision: %s\nInversion tolerance = %e\0",
               (typeid(Float) == typeid(float)) ? "single" : "double", info.inv_tol);
      asprintf(&ens_info,"kappa = %10.8f\nmu = %8.6f\nCsw = %8.6f\0",info.kappa, info.mu, info.csw);
      hid_t attrdat_id1 = H5Screate(H5S_SCALAR);
      hid_t attrdat_id2 = H5Screate(H5S_SCALAR);
      hid_t attrdat_id3 = H5Screate(H5S_SCALAR);
      hid_t attrdat_id4 = H5Screate(H5S_SCALAR);
      hid_t type_id1 = H5Tcopy(H5T_C_S1);
      hid_t type_id2 = H5Tcopy(H5T_C_S1);
      hid_t type_id3 = H5Tcopy(H5T_C_S1);
      hid_t type_id4 = H5Tcopy(H5T_C_S1);
      H5Tset_size(type_id1, strlen(cNmoms));
      H5Tset_size(type_id2, strlen(cQsq));
      H5Tset_size(type_id3, strlen(corr_info));
      H5Tset_size(type_id4, strlen(ens_info));
      hid_t attr_id1 = H5Acreate2(file_idt, "Nmoms", type_id1, attrdat_id1, H5P_DEFAULT, H5P_DEFAULT);
      hid_t attr_id2 = H5Acreate2(file_idt, "Qsq"  , type_id2, attrdat_id2, H5P_DEFAULT, H5P_DEFAULT);
      hid_t attr_id3 = H5Acreate2(file_idt, "Correlator-info", type_id3, attrdat_id3, H5P_DEFAULT, H5P_DEFAULT);
      hid_t attr_id4 = H5Acreate2(file_idt, "Ensemble-info", type_id4, attrdat_id4, H5P_DEFAULT, H5P_DEFAULT);
      H5Awrite(attr_id1, type_id1, cNmoms);
      H5Awrite(attr_id2, type_id2, cQsq);
      H5Awrite(attr_id3, type_id3, corr_info);
      H5Awrite(attr_id4, type_id4, ens_info);
      H5Aclose(attr_id1);
      H5Aclose(attr_id2);
      H5Aclose(attr_id3);
      H5Aclose(attr_id4);
      H5Tclose(type_id1);
      H5Tclose(type_id2);
      H5Tclose(type_id3);
      H5Tclose(type_id4);
      H5Sclose(attrdat_id1);
      H5Sclose(attrdat_id2);
      H5Sclose(attrdat_id3);
      H5Sclose(attrdat_id4);

      hid_t MOMTYPE_H5 = H5T_NATIVE_INT;
      char *dset_tag;
      asprintf(&dset_tag,"Momenta_list_xyz");

      hsize_t Mdims[2] = {(hsize_t)Nmoms,3};
      hid_t filespace  = H5Screate_simple(2, Mdims, NULL);
      hid_t dataset_id = H5Dcreate(file_idt, dset_tag, MOMTYPE_H5, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      int *Moms_H5 = (int*) malloc(GK_Nmoms*3*sizeof(int));
      for(int im=0;im<GK_Nmoms;im++){
        for(int d=0;d<3;d++) Moms_H5[d + 3*im] = GK_moms[im][d];
      }

      herr_t status = H5Dwrite(dataset_id, MOMTYPE_H5, H5S_ALL, filespace, H5P_DEFAULT, Moms_H5);

      H5Dclose(dataset_id);
      H5Sclose(filespace);
      H5Fclose(file_idt);

      free(Moms_H5);
    }//-if GK_timeRank==0


  }//-if GK_timeRank >=0 && GK_timeRank < GK_nProc[3]

}



//-C.K. - New function to copy the meson two-point functions into write 
//Buffers for writing in HDF5 format
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
copyTwopMesonsToHDF5_Buf(void *Twop_mesons_HDF5, 
			 void *corrMesons, 
			 CORR_SPACE CorrSpace, bool HighMomForm){

  int Lt = GK_localL[3];
  int Nmoms = GK_Nmoms;

  if(CorrSpace==MOMENTUM_SPACE){

    if(HighMomForm){
      if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
        
        for(int ip=0;ip<2;ip++){
          for(int mes=0;mes<N_MESONS;mes++){
            for(int imom=0;imom<Nmoms;imom++){
              for(int it=0;it<Lt;it++){
                ((Float*)Twop_mesons_HDF5)[0 + 2*imom + 2*Nmoms*it + 2*Nmoms*Lt*mes + 2*Nmoms*Lt*N_MESONS*ip] = ((Float(*)[2][N_MESONS])corrMesons)[0 + 2*imom + 2*Nmoms*it][ip][mes];
                ((Float*)Twop_mesons_HDF5)[1 + 2*imom + 2*Nmoms*it + 2*Nmoms*Lt*mes + 2*Nmoms*Lt*N_MESONS*ip] = ((Float(*)[2][N_MESONS])corrMesons)[1 + 2*imom + 2*Nmoms*it][ip][mes];
              }}}
        }//-ip
        
      }//-if GK_timeRank
    }//-if HighMomForm
    else{
      if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
        
        for(int ip=0;ip<2;ip++){
          for(int mes=0;mes<N_MESONS;mes++){
            for(int imom=0;imom<Nmoms;imom++){
              for(int it=0;it<Lt;it++){
                ((Float*)Twop_mesons_HDF5)[0 + 2*it + 2*Lt*imom + 2*Lt*Nmoms*mes + 2*Lt*Nmoms*N_MESONS*ip] = ((Float(*)[2][N_MESONS])corrMesons)[0 + 2*imom + 2*Nmoms*it][ip][mes];
                ((Float*)Twop_mesons_HDF5)[1 + 2*it + 2*Lt*imom + 2*Lt*Nmoms*mes + 2*Lt*Nmoms*N_MESONS*ip] = ((Float(*)[2][N_MESONS])corrMesons)[1 + 2*imom + 2*Nmoms*it][ip][mes];
              }}}
        }//-ip
        
      }//-if GK_timeRank
    }//-else HighMomForm

  }//-if CorrSpace
  else if(CorrSpace==POSITION_SPACE){
    int Lv = GK_localVolume;

    for(int ip=0;ip<2;ip++){
      for(int mes=0;mes<N_MESONS;mes++){
        for(int v=0;v<Lv;v++){
          ((Float*)Twop_mesons_HDF5)[0 + 2*v + 2*Lv*ip + 2*Lv*2*mes] = ((Float(*)[2][N_MESONS])corrMesons)[0 + 2*v][ip][mes];
          ((Float*)Twop_mesons_HDF5)[1 + 2*v + 2*Lv*ip + 2*Lv*2*mes] = ((Float(*)[2][N_MESONS])corrMesons)[1 + 2*v][ip][mes];
        }}
    }//-ip

  }//-else if

}


//-C.K. New function to write the mesons two-point function in ASCII format
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::writeTwopMesons_ASCII(void *corrMesons, char *filename_out, int isource, CORR_SPACE CorrSpace){

  if(CorrSpace!=MOMENTUM_SPACE) errorQuda("writeTwopMesons_ASCII: Supports writing only in momentum-space!\n");

  Float (*GLcorrMesons)[2][N_MESONS] = (Float(*)[2][N_MESONS]) calloc(GK_totalL[3]*GK_Nmoms*2*N_MESONS*2,sizeof(Float));;
  if( GLcorrMesons == NULL )errorQuda("writeTwopMesons_ASCII: Cannot allocate memory for Meson two-point function buffer.\n");

  MPI_Datatype DATATYPE = -1;
  if( typeid(Float) == typeid(float)){
    DATATYPE = MPI_FLOAT;
    printfQuda("writeTwopMesons_ASCII: Will write in single precision\n");
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE = MPI_DOUBLE;
    printfQuda("writeTwopMesons_ASCII: Will write in double precision\n");
  }

  int error;
  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
    error = MPI_Gather((Float*) corrMesons,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,DATATYPE,GLcorrMesons,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,DATATYPE,0,GK_timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
  }

  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out = fopen(filename_out,"w");
    if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
    for(int ip = 0 ; ip < N_MESONS ; ip++)
      for(int it = 0 ; it < GK_totalL[3] ; it++)
	for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	  int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	  fprintf(ptr_out,"%d \t %d \t %+d %+d %+d \t %+e %+e \t %+e %+e\n",ip,it,GK_moms[imom][0],GK_moms[imom][1],GK_moms[imom][2],
		  GLcorrMesons[it_shift*GK_Nmoms*2+imom*2+0][0][ip], GLcorrMesons[it_shift*GK_Nmoms*2+imom*2+1][0][ip],
		  GLcorrMesons[it_shift*GK_Nmoms*2+imom*2+0][1][ip], GLcorrMesons[it_shift*GK_Nmoms*2+imom*2+1][1][ip]);
	}
    fclose(ptr_out);
  }

  free(GLcorrMesons);
}

//-C.K. Overloaded function to perform the meson contractions without 
//writing the data
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
contractMesons(QKXTM_Propagator_Kepler<Float> &prop1,
	       QKXTM_Propagator_Kepler<Float> &prop2, 
	       void *corrMesons, 
	       int isource, 
	       CORR_SPACE CorrSpace){
  cudaTextureObject_t texProp1, texProp2;
  prop1.createTexObject(&texProp1);
  prop2.createTexObject(&texProp2);

  if( typeid(Float) == typeid(float))  
    printfQuda("contractMesons: Will perform in single precision\n");
  if( typeid(Float) == typeid(double)) 
    printfQuda("contractMesons: Will perform in double precision\n");

  if(CorrSpace==POSITION_SPACE){
    for(int it = 0 ; it < GK_localL[3] ; it++) {
      run_contractMesons(texProp1,
			 texProp2,
			 (void*) corrMesons,
			 it,isource,
			 sizeof(Float),CorrSpace);
    }
  }
  else if( CorrSpace==MOMENTUM_SPACE ){
    Float (*corrMesons_local)[2][N_MESONS] = (Float(*)[2][N_MESONS]) calloc(GK_localL[3]*GK_Nmoms*2*N_MESONS*2,sizeof(Float));
    if( corrMesons_local == NULL )errorQuda("contractMesons: Cannot allocate memory for Meson two-point function contract buffer.\n");

    for(int it = 0 ; it < GK_localL[3] ; it++) run_contractMesons(texProp1,texProp2,(void*) corrMesons_local,it,isource,sizeof(Float),CorrSpace);

    MPI_Datatype DATATYPE = -1;
    if( typeid(Float) == typeid(float))  DATATYPE = MPI_FLOAT;
    if( typeid(Float) == typeid(double)) DATATYPE = MPI_DOUBLE;
    
    MPI_Reduce(corrMesons_local, (Float*)corrMesons,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,DATATYPE,MPI_SUM,0, GK_spaceComm);

    free(corrMesons_local);
  }
  else errorQuda("contractMesons: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

  prop1.destroyTexObject(texProp1);
  prop2.destroyTexObject(texProp2);
}

//--------------------------------------------------------//

template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
seqSourceFixSinkPart1(QKXTM_Vector_Kepler<Float> &vec, 
		      QKXTM_Propagator3D_Kepler<Float> &prop1, 
		      QKXTM_Propagator3D_Kepler<Float> &prop2, 
		      int tsinkMtsource, int nu, int c2, 
		      WHICHPROJECTOR PID, 
		      WHICHPARTICLE testParticle){
  
  cudaTextureObject_t tex1,tex2;
  prop1.createTexObject(&tex1);
  prop2.createTexObject(&tex2);

  run_seqSourceFixSinkPart1(vec.D_elem(), tsinkMtsource, tex1, 
			    tex2, nu, c2, PID, testParticle, sizeof(Float));
  
  prop1.destroyTexObject(tex1);
  prop2.destroyTexObject(tex2);
  checkCudaError();
  
}

template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
seqSourceFixSinkPart2(QKXTM_Vector_Kepler<Float> &vec, 
		      QKXTM_Propagator3D_Kepler<Float> &prop, 
		      int tsinkMtsource, int nu, int c2, 
		      WHICHPROJECTOR PID, 
		      WHICHPARTICLE testParticle){
  cudaTextureObject_t tex;
  prop.createTexObject(&tex);
  
  run_seqSourceFixSinkPart2(vec.D_elem(), tsinkMtsource, tex, 
			    nu, c2, PID, testParticle, sizeof(Float));
  
  prop.destroyTexObject(tex);
  
  checkCudaError();
}

template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
writeThrpHDF5(void *Thrp_local_HDF5, 
	      void *Thrp_noether_HDF5, 
	      void **Thrp_oneD_HDF5, 
	      char *filename, 
	      qudaQKXTMinfo_Kepler info, 
	      int isource, 
	      WHICHPARTICLE NUCLEON){

  if(info.CorrSpace==MOMENTUM_SPACE){
    if(info.HighMomForm){
      writeThrpHDF5_MomSpace_HighMomForm((void*) Thrp_local_HDF5, (void*) Thrp_noether_HDF5, (void**)Thrp_oneD_HDF5, filename, info, isource, NUCLEON);
    }
    else{
      writeThrpHDF5_MomSpace((void*) Thrp_local_HDF5, (void*) Thrp_noether_HDF5, (void**)Thrp_oneD_HDF5, filename, info, isource, NUCLEON);
    }
  }
  else if(info.CorrSpace==POSITION_SPACE) writeThrpHDF5_PosSpace((void*) Thrp_local_HDF5, (void*) Thrp_noether_HDF5, (void**)Thrp_oneD_HDF5, filename, info, isource, NUCLEON);
  else errorQuda("writeThrpHDF5: Unsupported value for info.CorrSpace! Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

}


//-C.K. - New function to write the three-point function in HDF5 format, position-space
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
writeThrpHDF5_PosSpace(void *Thrp_local_HDF5, 
		       void *Thrp_noether_HDF5, 
		       void **Thrp_oneD_HDF5, 
		       char *filename, 
		       qudaQKXTMinfo_Kepler info, 
		       int isource, 
		       WHICHPARTICLE NUCLEON){
  
  if(info.CorrSpace!=POSITION_SPACE) errorQuda("writeThrpHDF5_PosSpace: Support for writing the three-point function only in position-space!\n");

  hid_t DATATYPE_H5;
  if( typeid(Float) == typeid(float) ){
    DATATYPE_H5 = H5T_NATIVE_FLOAT;
    printfQuda("writeThrpHDF5_PosSpace: Will write in single precision\n");
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE_H5 = H5T_NATIVE_DOUBLE;
    printfQuda("writeThrp_HDF5_PosSpace: Will write in double precision\n");
  }

  Float *writeThrpBuf;

  int Nsink = info.Ntsink;
  int pc[4];
  int tL[4];
  int lL[4];
  for(int i=0;i<4;i++){
    pc[i] = comm_coords(default_topo)[i];
    tL[i] = GK_totalL[i];
    lL[i] = GK_localL[i];
  }
  int lV = GK_localVolume;

  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
  H5Pclose(fapl_id);

  char *group1_tag;
  asprintf(&group1_tag,"conf_%04d",info.traj);
  hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  char *group2_tag;
  asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  /* Attribute writing */
  //- Need these further on
  char *dir_order,*operator_list;
  asprintf(&dir_order,"[0,1,2,3] = [x,y,z,t]");
  asprintf(&operator_list,"0 = g5\n1 = gx\n2 = gy\n3 = gz\n4 = g0\n5 = Unity\n6 = g5gx\n7 = g5gy\n8 = g5gz\n9 = g5g0\n10 = g5sixy\n11 = g5sixz\n12 = g5siyz\n13 = g5si0x\n14 = g5si0y\n15 = g5si0z\n");

  //- Write the Source position
  char *src_pos;
  asprintf(&src_pos," [x, y, z, t] = [%02d, %02d, %02d, %02d]\0",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  hid_t attrdat_id = H5Screate(H5S_SCALAR);
  hid_t type_id = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id, strlen(src_pos));
  hid_t attr_id = H5Acreate2(group2_id, "source-position", 
			     type_id, attrdat_id, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id, type_id, src_pos);
  H5Aclose(attr_id);
  H5Tclose(type_id);
  H5Sclose(attrdat_id);
  
  //- Write general Correlator Info
  char *corr_info;
  asprintf(&corr_info,"Position-space %s 3pt-correlator\nIncludes ultra-local and one-derivative operators, noether current\nPrecision: %s\0",
	   (NUCLEON==PROTON)?"proton":"neutron",
	   (typeid(Float) == typeid(float)) ? "single" : "double");
  hid_t attrdat_id_2 = H5Screate(H5S_SCALAR);
  hid_t type_id_2 = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id_2, strlen(corr_info));
  hid_t attr_id_2 = H5Acreate2(file_id, "Correlator-info", type_id_2, 
			       attrdat_id_2, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id_2, type_id_2, corr_info);
  H5Aclose(attr_id_2);
  H5Tclose(type_id_2);
  H5Sclose(attrdat_id_2);
  //------------------------------------------------------------

  hid_t group3_id;
  hid_t group4_id;
  hid_t group5_id;

  for(int its=0;its<Nsink;its++){
    int tsink = info.tsinkSource[its];
    char *group3_tag;
    asprintf(&group3_tag,"tsink_%02d",tsink);
    group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, 
			  H5P_DEFAULT, H5P_DEFAULT);
    
    for(int ipr=0;ipr<info.Nproj[its];ipr++){
      char *group4_tag;
      asprintf(&group4_tag,"proj_%s",
	       info.thrp_proj_type[info.proj_list[its][ipr]]);
      group4_id = H5Gcreate(group3_id, group4_tag, H5P_DEFAULT, 
			    H5P_DEFAULT, H5P_DEFAULT);

      for(int thrp_int=0;thrp_int<3;thrp_int++){
	THRP_TYPE type = (THRP_TYPE) thrp_int;

	char *group5_tag;
	asprintf(&group5_tag,"%s", info.thrp_type[thrp_int]);
	group5_id = H5Gcreate(group4_id, group5_tag, 
			      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	if(type==THRP_LOCAL){
	  char *attr_info;
	  asprintf(&attr_info,"Ultra-local operators:\nIndex-order: [operator, up-0/down-1, t, z, y, x, real/imag]\nOperator list:\n%s\0",operator_list);
	  hid_t attrdat_id_c = H5Screate(H5S_SCALAR);
	  hid_t type_id_c = H5Tcopy(H5T_C_S1);
	  H5Tset_size(type_id_c, strlen(attr_info));
	  hid_t attr_id_c = H5Acreate2(group5_id, "Ultra-local Info", 
				       type_id_c, attrdat_id_c, 
				       H5P_DEFAULT, H5P_DEFAULT);
	  H5Awrite(attr_id_c, type_id_c, attr_info);
	  H5Aclose(attr_id_c);
	  H5Tclose(type_id_c);
	  H5Sclose(attrdat_id_c);

	  int Mel = 16;
	  int Sdim = 7;
	  // Size of the dataspace -> Operator, up-down, localVolume, Re-Im
	  hsize_t dims[7]  = {Mel, 2, tL[3], tL[2], tL[1], tL[0], 2};
	  // Dimensions of the local dataspace for each rank
	  hsize_t ldims[7] = {Mel, 2, lL[3], lL[2], lL[1], lL[0], 2}; 
	  hsize_t start[7] = {0,0,pc[3]*lL[3],pc[2]*lL[2],pc[1]*lL[1],pc[0]*lL[0],0}; // start position for each rank

	  hid_t filespace  = H5Screate_simple(Sdim, dims,  NULL);
	  hid_t subspace   = H5Screate_simple(Sdim, ldims, NULL);
	  hid_t dataset_id = H5Dcreate(group5_id, "threep", DATATYPE_H5, 
				       filespace, H5P_DEFAULT, 
				       H5P_DEFAULT, H5P_DEFAULT);
	  filespace = H5Dget_space(dataset_id);
	  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, 
			      ldims, NULL);
	  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
	  
	  writeThrpBuf = &(((Float*)Thrp_local_HDF5)[2*lV*2*Mel*its + 
						     2*lV*2*Mel*Nsink*ipr]);
	  
	  herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, 
				   filespace, plist_id, writeThrpBuf);
	  if(status<0) errorQuda("writeThrpHDF5_PosSpace: Unsuccessful writing of the %s dataset. Exiting\n",info.thrp_type[thrp_int]);

	  H5Sclose(subspace);
	  H5Dclose(dataset_id);
	  H5Sclose(filespace);
	  H5Pclose(plist_id);
	}//-ultra_local
	else if(type==THRP_NOETHER){
	  char *attr_info;
	  asprintf(&attr_info,"Noether current:\nIndex-order: [direction, up-0/down-1, t, z, y, x, real/imag]\nDirection order:\n%s\0",dir_order);
	  hid_t attrdat_id_c = H5Screate(H5S_SCALAR);
	  hid_t type_id_c = H5Tcopy(H5T_C_S1);
	  H5Tset_size(type_id_c, strlen(attr_info));
	  hid_t attr_id_c = H5Acreate2(group5_id, "Noether current Info", type_id_c, attrdat_id_c, H5P_DEFAULT, H5P_DEFAULT);
	  H5Awrite(attr_id_c, type_id_c, attr_info);
	  H5Aclose(attr_id_c);
	  H5Tclose(type_id_c);
	  H5Sclose(attrdat_id_c);

	  int Mel = 4;
	  int Sdim = 7;
	  // Size of the dataspace -> Operator, up-down, localVolume, Re-Im
	  hsize_t dims[7]  = {Mel,2,tL[3],tL[2],tL[1],tL[0],2}; 
	  // Dimensions of the local dataspace for each rank
	  hsize_t ldims[7] = {Mel,2,lL[3],lL[2],lL[1],lL[0],2}; 
	  hsize_t start[7] = {0,0,pc[3]*lL[3],pc[2]*lL[2],pc[1]*lL[1],pc[0]*lL[0],0}; // start position for each rank

	  hid_t filespace  = H5Screate_simple(Sdim, dims,  NULL);
	  hid_t subspace   = H5Screate_simple(Sdim, ldims, NULL);
	  hid_t dataset_id = H5Dcreate(group5_id, "threep", DATATYPE_H5, 
				       filespace, H5P_DEFAULT, 
				       H5P_DEFAULT, H5P_DEFAULT);
	  filespace = H5Dget_space(dataset_id);
	  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, 
			      ldims, NULL);
	  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	  writeThrpBuf=&(((Float*)Thrp_noether_HDF5)[2*lV*2*Mel*its + 
						     2*lV*2*Mel*Nsink*ipr]);
	  
	  herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, 
				   filespace, plist_id, writeThrpBuf);
	  if(status<0) errorQuda("writeThrpHDF5_PosSpace: Unsuccessful writing of the %s dataset. Exiting\n",info.thrp_type[thrp_int]);

	  H5Sclose(subspace);
	  H5Dclose(dataset_id);
	  H5Sclose(filespace);
	  H5Pclose(plist_id);
	}//- noether
	else if(type==THRP_ONED){
	  char *attr_info;
	  asprintf(&attr_info,"One-derivative operators:\nIndex-order: [direction, operator, up-0/down-1, t, z, y, x, real/imag]\nOperator list:%s\nDirection order:\n%s\0", operator_list,dir_order);
	  hid_t attrdat_id_c = H5Screate(H5S_SCALAR);
	  hid_t type_id_c = H5Tcopy(H5T_C_S1);
	  H5Tset_size(type_id_c, strlen(attr_info));
	  hid_t attr_id_c = H5Acreate2(group5_id, "One-derivative Info", 
				       type_id_c, attrdat_id_c, 
				       H5P_DEFAULT, H5P_DEFAULT);
	  H5Awrite(attr_id_c, type_id_c, attr_info);
	  H5Aclose(attr_id_c);
	  H5Tclose(type_id_c);
	  H5Sclose(attrdat_id_c);

	  int Mel = 16;
	  int Sdim = 8;
	  
	  hsize_t dims[8]  = {4,Mel,2,tL[3],tL[2],tL[1],tL[0],2}; // Size of the dataspace -> Direction, Operator, up-down, localVolume, Re-Im
	  hsize_t ldims[8] = {4,Mel,2,lL[3],lL[2],lL[1],lL[0],2}; // Dimensions of the local dataspace for each rank
	  hsize_t start[8] = {0,0,0,pc[3]*lL[3],pc[2]*lL[2],pc[1]*lL[1],pc[0]*lL[0],0}; // start position for each rank

	  hid_t filespace  = H5Screate_simple(Sdim, dims,  NULL);
	  hid_t subspace   = H5Screate_simple(Sdim, ldims, NULL);
	  hid_t dataset_id = H5Dcreate(group5_id, "threep", DATATYPE_H5, 
				       filespace, H5P_DEFAULT, 
				       H5P_DEFAULT, H5P_DEFAULT);
	  filespace = H5Dget_space(dataset_id);
	  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, 
			      ldims, NULL);
	  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	  writeThrpBuf = NULL;
	  if( (writeThrpBuf = (Float*) malloc(2*lV*2*Mel*4*sizeof(Float))) == NULL ) 
	    errorQuda("writeThrpHDF5_PosSpace: Cannot allocate writeBuffer for one-derivative three-point correlator\n");

	  for(int dir=0;dir<4;dir++) memcpy(&(writeThrpBuf[2*lV*2*Mel*dir]), &(((Float*)Thrp_oneD_HDF5[dir])[2*lV*2*Mel*its + 2*lV*2*Mel*Nsink*ipr]), 2*lV*2*Mel*sizeof(Float));

	  herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, filespace, plist_id, writeThrpBuf);
	  if(status<0) errorQuda("writeThrpHDF5_PosSpace: Unsuccessful writing of the %s dataset. Exiting\n",info.thrp_type[thrp_int]);

	  free(writeThrpBuf);

	  H5Sclose(subspace);
	  H5Dclose(dataset_id);
	  H5Sclose(filespace);
	  H5Pclose(plist_id);
	}//- oneD

	H5Gclose(group5_id);
      }//-thrp_int
      H5Gclose(group4_id);
    }//-ipr
    H5Gclose(group3_id);
  }//-its

  H5Gclose(group2_id);
  H5Gclose(group1_id);
  H5Fclose(file_id);


  return;
}


//-C.K. - New function to write the three-point function in HDF5 format, 
// momentum-space
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
writeThrpHDF5_MomSpace(void *Thrp_local_HDF5, 
		       void *Thrp_noether_HDF5, 
		       void **Thrp_oneD_HDF5, 
		       char *filename, 
		       qudaQKXTMinfo_Kepler info, 
		       int isource, 
		       WHICHPARTICLE NUCLEON){
  
  if(info.CorrSpace!=MOMENTUM_SPACE || info.HighMomForm) errorQuda("writeThrpHDF5_MomSpace: Supports writing the three-point function only in momentum-space and for NOT High-Momenta Form!\n");

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){

    hid_t DATATYPE_H5;
    if( typeid(Float) == typeid(float) ){
      DATATYPE_H5 = H5T_NATIVE_FLOAT;
      printfQuda("writeThrpHDF5_MomSpace: Will write in single precision\n");
    }
    if( typeid(Float) == typeid(double)){
      DATATYPE_H5 = H5T_NATIVE_DOUBLE;
      printfQuda("writeThrp_HDF5_MomSpace: Will write in double precision\n");
    }

    Float *writeThrpBuf;

    int Nsink = info.Ntsink;
    int t_src = GK_sourcePosition[isource][3];
    int Lt = GK_localL[3];
    int T  = GK_totalL[3];
    int Mel;

    int src_rank = t_src/Lt;
    int h = Lt - t_src%Lt;
    int w = t_src%Lt;

    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, GK_timeComm, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    char *group1_tag;
    asprintf(&group1_tag,"conf_%04d",info.traj);
    hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);

    char *group2_tag;
    asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	     GK_sourcePosition[isource][0],
	     GK_sourcePosition[isource][1],
	     GK_sourcePosition[isource][2],
	     GK_sourcePosition[isource][3]);

    hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);

    hid_t group3_id;
    hid_t group4_id;
    hid_t group5_id;
    hid_t group6_id;
    hid_t group7_id;
    hid_t group8_id;

    hsize_t dims[3],ldims[3],start[3];

    for(int its=0;its<Nsink;its++){
      int tsink = info.tsinkSource[its];
      char *group3_tag;
      asprintf(&group3_tag,"tsink_%02d",tsink);
      group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, 
			    H5P_DEFAULT, H5P_DEFAULT);

      bool all_print = false;
      if( tsink >= (T - t_src%Lt) ) all_print = true;

      int sink_rank = ((t_src+tsink)%T)/Lt;
      int l = ((t_src+tsink)%T)%Lt + 1; //-Significant only for sink_rank

      //-Determine which processes will print for this tsink
      bool print_rank;
      if(all_print) print_rank = true;
      else{
	print_rank = false;
	for(int i=0;i<GK_nProc[3];i++){
	  if( GK_timeRank == ((src_rank+i)%GK_nProc[3]) ) print_rank = true;
	  if( ((src_rank+i)%GK_nProc[3]) == sink_rank ) break;
	}
      }
      
      //-Determine the start position for each rank
      if(print_rank){
	// if src_rank = sink_rank then this is the same
	if(GK_timeRank==src_rank) start[0] = 0; 
	else{
	  int offs;
	  for(offs=0;offs<GK_nProc[3];offs++){
	    if( GK_timeRank == ((src_rank+offs)%GK_nProc[3]) ) break;
	  }
	  offs--;
	  start[0] = h + offs*Lt;
	}
      }
      else start[0] = 0; // Need to set this to zero when a given rank does not print. Otherwise the dimensions will not fit   
      start[1] = 0; //
      start[2] = 0; //-These are common among all ranks

      for(int ipr=0;ipr<info.Nproj[its];ipr++){
	char *group4_tag;
	asprintf(&group4_tag,"proj_%s",
		 info.thrp_proj_type[info.proj_list[its][ipr]]);
	group4_id = H5Gcreate(group3_id, group4_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);
      
	for(int part=0;part<2;part++){
	  char *group5_tag;
	  asprintf(&group5_tag,"%s", (part==0) ? "up" : "down");
	  group5_id = H5Gcreate(group4_id, group5_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);
	
	  for(int thrp_int=0;thrp_int<3;thrp_int++){
	    THRP_TYPE type = (THRP_TYPE) thrp_int;

	    char *group6_tag;
	    asprintf(&group6_tag,"%s", info.thrp_type[thrp_int]);
	    group6_id = H5Gcreate(group5_id, group6_tag, H5P_DEFAULT, 
				  H5P_DEFAULT, H5P_DEFAULT);
	  
	    //-Determine the global dimensions
	    if(type==THRP_LOCAL || type==THRP_ONED) Mel = 16;
	    else if (type==THRP_NOETHER) Mel = 4;
	    else errorQuda("writeThrpHDF5_MomSpace: Undefined three-point function type.\n");
	    dims[0] = tsink+1;
	    dims[1] = Mel;
	    dims[2] = 2;

	    //-Determine ldims for print ranks
	    if(all_print){
	      ldims[1] = dims[1];
	      ldims[2] = dims[2];
	      if(GK_timeRank==src_rank) ldims[0] = h;
	      else ldims[0] = Lt;
	    }
	    else{
	      if(print_rank){
		ldims[1] = dims[1];
		ldims[2] = dims[2];
		if(src_rank != sink_rank){
		  if(GK_timeRank==src_rank) ldims[0] = h;
		  else if(GK_timeRank==sink_rank) ldims[0] = l;
		  else ldims[0] = Lt;
		}
		else ldims[0] = dims[0];
	      }
	      //- Non-print ranks get zero space
	      else for(int i=0;i<3;i++) ldims[i] = 0; 
	    }
	    
	    for(int imom=0;imom<GK_Nmoms;imom++){
	      char *group7_tag;
	      asprintf(&group7_tag,"mom_xyz_%+d_%+d_%+d",
		       GK_moms[imom][0],
		       GK_moms[imom][1],
		       GK_moms[imom][2]);
	      group7_id = H5Gcreate(group6_id, group7_tag, H5P_DEFAULT, 
				    H5P_DEFAULT, H5P_DEFAULT);
	    
	      if(type==THRP_ONED){
		for(int mu=0;mu<4;mu++){
		  char *group8_tag;
		  asprintf(&group8_tag,"dir_%02d",mu);
		  group8_id = H5Gcreate(group7_id, group8_tag, H5P_DEFAULT, 
					H5P_DEFAULT, H5P_DEFAULT);

		  hid_t filespace  = H5Screate_simple(3, dims, NULL);
		  hid_t dataset_id = H5Dcreate(group8_id, "threep", 
					       DATATYPE_H5, filespace, 
					       H5P_DEFAULT, H5P_DEFAULT, 
					       H5P_DEFAULT);
		  hid_t subspace   = H5Screate_simple(3, ldims, NULL);
		  filespace = H5Dget_space(dataset_id);
		  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, 
				      NULL, ldims, NULL);
		  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
		  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

		  if(GK_timeRank==src_rank) writeThrpBuf = &(((Float*)Thrp_oneD_HDF5[mu])[2*Mel*w + 2*Mel*Lt*imom + 2*Mel*Lt*GK_Nmoms*part + 2*Mel*Lt*GK_Nmoms*2*its + 2*Mel*Lt*GK_Nmoms*2*Nsink*ipr]);
		  else writeThrpBuf = &(((Float*)Thrp_oneD_HDF5[mu])[2*Mel*Lt*imom + 2*Mel*Lt*GK_Nmoms*part + 2*Mel*Lt*GK_Nmoms*2*its + 2*Mel*Lt*GK_Nmoms*2*Nsink*ipr]);

		  herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, 
					   subspace, filespace, 
					   plist_id, writeThrpBuf);

		  H5Sclose(subspace);
		  H5Dclose(dataset_id);
		  H5Sclose(filespace);
		  H5Pclose(plist_id);

		  H5Gclose(group8_id);
		}//-mu	      
	      }//-if
	      else{
		Float *thrpBuf;
		if(type==THRP_LOCAL) thrpBuf = (Float*)Thrp_local_HDF5;
		else if(type==THRP_NOETHER) 
		  thrpBuf = (Float*)Thrp_noether_HDF5;
		
		hid_t filespace  = H5Screate_simple(3, dims, NULL);
		hid_t dataset_id = H5Dcreate(group7_id, "threep", 
					     DATATYPE_H5, filespace, 
					     H5P_DEFAULT, H5P_DEFAULT, 
					     H5P_DEFAULT);
		hid_t subspace   = H5Screate_simple(3, ldims, NULL);
		filespace = H5Dget_space(dataset_id);
		H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL,
				    ldims, NULL);
		hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
		H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

		if(GK_timeRank==src_rank) writeThrpBuf = &(thrpBuf[2*Mel*w + 2*Mel*Lt*imom + 2*Mel*Lt*GK_Nmoms*part + 2*Mel*Lt*GK_Nmoms*2*its + 2*Mel*Lt*GK_Nmoms*2*Nsink*ipr]);
		else writeThrpBuf = &(thrpBuf[2*Mel*Lt*imom + 2*Mel*Lt*GK_Nmoms*part + 2*Mel*Lt*GK_Nmoms*2*its + 2*Mel*Lt*GK_Nmoms*2*Nsink*ipr]);

		herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, filespace, plist_id, writeThrpBuf);
	      
		H5Sclose(subspace);
		H5Dclose(dataset_id);
		H5Sclose(filespace);
		H5Pclose(plist_id);
	      }//-else	  
	      H5Gclose(group7_id);
	    }//-imom	 
	    H5Gclose(group6_id);
	  }//-thrp_int
	  H5Gclose(group5_id);
	}//-part
	H5Gclose(group4_id);
      }//-projector
      H5Gclose(group3_id);
    }//-its
    
    H5Gclose(group2_id);
    H5Gclose(group1_id);
    H5Fclose(file_id);


    for(int its=0;its<Nsink;its++){
      int tsink = info.tsinkSource[its];
      int l = ((t_src+tsink)%T)%Lt + 1;
      
      int sink_rank = ((t_src+tsink)%T)/Lt;
      
      // No need to write something else
      if( tsink < (T - t_src%Lt) ) continue; 
      
      if(GK_timeRank==sink_rank){
	Float *tailBuf;
    
	hid_t file_idt = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

	start[0] = tsink + 1 - l;
	start[1] = 0;
	start[2] = 0;

	for(int ipr=0;ipr<info.Nproj[its];ipr++){
	  for(int part=0;part<2;part++){
	    for(int thrp_int=0;thrp_int<3;thrp_int++){
	      THRP_TYPE type = (THRP_TYPE) thrp_int;
	    
	      //-Determine the global dimensions
	      if(type==THRP_LOCAL || type==THRP_ONED) Mel = 16;
	      else if (type==THRP_NOETHER) Mel = 4;
	      else errorQuda("writeThrp_HDF5: Undefined three-point function type.\n");
	      dims[0] = tsink+1;
	      dims[1] = Mel;
	      dims[2] = 2;

	      ldims[0] = l;
	      ldims[1] = Mel;
	      ldims[2] = 2;

	      for(int imom=0;imom<GK_Nmoms;imom++){
		if(type==THRP_ONED){
		  for(int mu=0;mu<4;mu++){
		    char *group_tag;
		    asprintf(&group_tag,"conf_%04d/sx%02dsy%02dsz%02dst%02d/tsink_%02d/proj_%s/%s/%s/mom_xyz_%+d_%+d_%+d/dir_%02d",
			     info.traj,
			     GK_sourcePosition[isource][0],
			     GK_sourcePosition[isource][1],
			     GK_sourcePosition[isource][2],
			     GK_sourcePosition[isource][3],
			     tsink, 
			     info.thrp_proj_type[info.proj_list[its][ipr]], 
			     (part==0) ? "up" : "down", 
			     info.thrp_type[thrp_int], 
			     GK_moms[imom][0], 
			     GK_moms[imom][1], 
			     GK_moms[imom][2], mu);
		    hid_t group_id = H5Gopen(file_idt, group_tag, 
					     H5P_DEFAULT);

		    hid_t dset_id  = H5Dopen(group_id, "threep", 
					     H5P_DEFAULT);
		    hid_t mspace_id = H5Screate_simple(3, ldims, NULL);
		    hid_t dspace_id = H5Dget_space(dset_id);

		    H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, start, 
					NULL, ldims, NULL);

		    tailBuf = &(((Float*)Thrp_oneD_HDF5[mu])[2*Mel*Lt*imom + 2*Mel*Lt*GK_Nmoms*part + 2*Mel*Lt*GK_Nmoms*2*its + 2*Mel*Lt*GK_Nmoms*2*Nsink*ipr]);

		    herr_t status = H5Dwrite(dset_id, DATATYPE_H5, 
					     mspace_id, dspace_id, 
					     H5P_DEFAULT, tailBuf);

		    H5Dclose(dset_id);
		    H5Sclose(mspace_id);
		    H5Sclose(dspace_id);
		    H5Gclose(group_id);
		  }//-mu
		}
		else{
		  Float *thrpBuf;
		  if(type==THRP_LOCAL) thrpBuf = (Float*)Thrp_local_HDF5;
		  else if(type==THRP_NOETHER) 
		    thrpBuf = (Float*)Thrp_noether_HDF5;

		  char *group_tag;
		  asprintf(&group_tag,"conf_%04d/sx%02dsy%02dsz%02dst%02d/tsink_%02d/proj_%s/%s/%s/mom_xyz_%+d_%+d_%+d",info.traj,
			   GK_sourcePosition[isource][0],
			   GK_sourcePosition[isource][1],
			   GK_sourcePosition[isource][2],
			   GK_sourcePosition[isource][3],
			   tsink, 
			   info.thrp_proj_type[info.proj_list[its][ipr]], 
			   (part==0) ? "up" : "down", 
			   info.thrp_type[thrp_int], 
			   GK_moms[imom][0], 
			   GK_moms[imom][1], 
			   GK_moms[imom][2]);
		  hid_t group_id = H5Gopen(file_idt, group_tag, H5P_DEFAULT);
		  hid_t dset_id  = H5Dopen(group_id, "threep", H5P_DEFAULT);
		  hid_t mspace_id  = H5Screate_simple(3, ldims, NULL);
		  hid_t dspace_id = H5Dget_space(dset_id);

		  H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, 
				      start, NULL, ldims, NULL);
		
		  tailBuf = &(thrpBuf[2*Mel*Lt*imom + 
				      2*Mel*Lt*GK_Nmoms*part + 
				      2*Mel*Lt*GK_Nmoms*2*its + 
				      2*Mel*Lt*GK_Nmoms*2*Nsink*ipr]);
		  
		  herr_t status = H5Dwrite(dset_id, DATATYPE_H5, 
					   mspace_id, dspace_id, 
					   H5P_DEFAULT, tailBuf);
		  
		  H5Dclose(dset_id);
		  H5Sclose(mspace_id);
		  H5Sclose(dspace_id);
		  H5Gclose(group_id);
		}
	      }//-imom
	    }//-thrp_int
	  }//-part
	}//-projector
	H5Fclose(file_idt);
      }//-if GK_timeRank==sink_rank
    }//-its

  }//-if
}

//-C.K. - New function to write the three-point function in HDF5 format, momentum-space, in High-Momenta Form
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::writeThrpHDF5_MomSpace_HighMomForm(void *Thrp_local_HDF5, void *Thrp_noether_HDF5, void **Thrp_oneD_HDF5, char *filename, qudaQKXTMinfo_Kepler info, int isource, WHICHPARTICLE NUCLEON){

  if(info.CorrSpace!=MOMENTUM_SPACE && !info.HighMomForm) errorQuda("writeThrpHDF5_MomSpace: Support for writing the three-point function only in momentum-space for high # of momenta!\n");

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){

    hid_t DATATYPE_H5;
    if( typeid(Float) == typeid(float) ){
      DATATYPE_H5 = H5T_NATIVE_FLOAT;
      printfQuda("writeThrpHDF5_MomSpace: Will write in single precision\n");
    }
    if( typeid(Float) == typeid(double)){
      DATATYPE_H5 = H5T_NATIVE_DOUBLE;
      printfQuda("writeThrp_HDF5_MomSpace: Will write in double precision\n");
    }

    Float *writeThrpBuf;

    int Nsink = info.Ntsink;
    int t_src = GK_sourcePosition[isource][3];
    int Lt = GK_localL[3];
    int T  = GK_totalL[3];
    int Mel;
    int Nmoms = GK_Nmoms;

    int src_rank = t_src/Lt;
    int h = Lt - t_src%Lt;
    int w = t_src%Lt;

    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, GK_timeComm, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    char *group1_tag;
    asprintf(&group1_tag,"conf_%04d",info.traj);
    hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    char *group2_tag;
    asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",GK_sourcePosition[isource][0],GK_sourcePosition[isource][1],GK_sourcePosition[isource][2],GK_sourcePosition[isource][3]);
    hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hid_t group3_id;
    hid_t group4_id;
    hid_t group5_id;
    hid_t group6_id;
    hid_t group7_id;

    hsize_t dims[4],ldims[4],start[4];

    for(int its=0;its<Nsink;its++){
      int tsink = info.tsinkSource[its];
      char *group3_tag;
      asprintf(&group3_tag,"tsink_%02d",tsink);
      group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      bool all_print = false;
      if( tsink >= (T - t_src%Lt) ) all_print = true;

      int sink_rank = ((t_src+tsink)%T)/Lt;
      int l = ((t_src+tsink)%T)%Lt + 1; //-Significant only for sink_rank

      //-Determine which processes will print for this tsink
      bool print_rank;
      if(all_print) print_rank = true;
      else{
        print_rank = false;
        for(int i=0;i<GK_nProc[3];i++){
          if( GK_timeRank == ((src_rank+i)%GK_nProc[3]) ) print_rank = true;
          if( ((src_rank+i)%GK_nProc[3]) == sink_rank ) break;
        }
      }
      
      //-Determine the start position for each rank
      if(print_rank){
        if(GK_timeRank==src_rank) start[0] = 0; // if src_rank = sink_rank then this is the same for sink_rank as well
        else{
          int offs;
          for(offs=0;offs<GK_nProc[3];offs++){
            if( GK_timeRank == ((src_rank+offs)%GK_nProc[3]) ) break;
          }
          offs--;
          start[0] = h + offs*Lt;
        }
      }
      else start[0] = 0; // Need to set this to zero when a given rank does not print. Otherwise the dimensions will not fit   
      start[1] = 0; //
      start[2] = 0; //-These are common among all ranks
      start[3] = 0; //

      for(int ipr=0;ipr<info.Nproj[its];ipr++){
        char *group4_tag;
        asprintf(&group4_tag,"proj_%s",info.thrp_proj_type[info.proj_list[its][ipr]]);
        group4_id = H5Gcreate(group3_id, group4_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      
        for(int part=0;part<2;part++){
          char *group5_tag;
          asprintf(&group5_tag,"%s", (part==0) ? "up" : "down");
          group5_id = H5Gcreate(group4_id, group5_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
          for(int thrp_int=0;thrp_int<3;thrp_int++){
            THRP_TYPE type = (THRP_TYPE) thrp_int;

            char *group6_tag;
            asprintf(&group6_tag,"%s", info.thrp_type[thrp_int]);
            group6_id = H5Gcreate(group5_id, group6_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          
            //-Determine the global dimensions
            if(type==THRP_LOCAL || type==THRP_ONED) Mel = 16;
            else if (type==THRP_NOETHER) Mel = 4;
            else errorQuda("writeThrpHDF5_MomSpace: Undefined three-point function type.\n");
            dims[0] = tsink+1;
            dims[1] = Nmoms;
            dims[2] = Mel;
            dims[3] = 2;

            //-Determine ldims for print ranks
            if(all_print){
              ldims[1] = dims[1];
              ldims[2] = dims[2];
              ldims[3] = dims[3];
              if(GK_timeRank==src_rank) ldims[0] = h;
              else ldims[0] = Lt;
            }
            else{
              if(print_rank){
                ldims[1] = dims[1];
                ldims[2] = dims[2];
                ldims[3] = dims[3];
                if(src_rank != sink_rank){
                  if(GK_timeRank==src_rank) ldims[0] = h;
                  else if(GK_timeRank==sink_rank) ldims[0] = l;
                  else ldims[0] = Lt;
                }
                else ldims[0] = dims[0];
              }
              else for(int i=0;i<4;i++) ldims[i] = 0; //- Non-print ranks get zero space
            }

            if(type==THRP_ONED){
              for(int mu=0;mu<4;mu++){
                char *group7_tag;
                asprintf(&group7_tag,"dir_%02d",mu);
                group7_id = H5Gcreate(group6_id, group7_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                hid_t filespace  = H5Screate_simple(4, dims, NULL);
                hid_t dataset_id = H5Dcreate(group7_id, "threep", DATATYPE_H5, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                hid_t subspace   = H5Screate_simple(4, ldims, NULL);
                filespace = H5Dget_space(dataset_id);
                H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, ldims, NULL);
                hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
                H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

                if(GK_timeRank==src_rank) writeThrpBuf = &(((Float*)Thrp_oneD_HDF5[mu])[2*Mel*Nmoms*w + 2*Mel*Nmoms*Lt*part + 2*Mel*Nmoms*Lt*2*its + 2*Mel*Nmoms*Lt*2*Nsink*ipr]);
                else writeThrpBuf = &(((Float*)Thrp_oneD_HDF5[mu])[2*Mel*Nmoms*Lt*part + 2*Mel*Nmoms*Lt*2*its + 2*Mel*Nmoms*Lt*2*Nsink*ipr]);

                herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, filespace, plist_id, writeThrpBuf);

                H5Sclose(subspace);
                H5Dclose(dataset_id);
                H5Sclose(filespace);
                H5Pclose(plist_id);

                H5Gclose(group7_id);
              }//-mu          
            }//-if
            else{
              Float *thrpBuf;
              if(type==THRP_LOCAL) thrpBuf = (Float*)Thrp_local_HDF5;
              else if(type==THRP_NOETHER) thrpBuf = (Float*)Thrp_noether_HDF5;

              hid_t filespace  = H5Screate_simple(4, dims, NULL);
              hid_t dataset_id = H5Dcreate(group6_id, "threep", DATATYPE_H5, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
              hid_t subspace   = H5Screate_simple(4, ldims, NULL);
              filespace = H5Dget_space(dataset_id);
              H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, ldims, NULL);
              hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
              H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

              if(GK_timeRank==src_rank) writeThrpBuf = &(thrpBuf[2*Mel*Nmoms*w + 2*Mel*Nmoms*Lt*part + 2*Mel*Nmoms*Lt*2*its + 2*Mel*Nmoms*Lt*2*Nsink*ipr]);
              else writeThrpBuf = &(thrpBuf[2*Mel*Nmoms*Lt*part + 2*Mel*Nmoms*Lt*2*its + 2*Mel*Nmoms*Lt*2*Nsink*ipr]);

              herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, filespace, plist_id, writeThrpBuf);
              
              H5Sclose(subspace);
              H5Dclose(dataset_id);
              H5Sclose(filespace);
              H5Pclose(plist_id);
            }//-else      
            H5Gclose(group6_id);
          }//-thrp_int
          H5Gclose(group5_id);
        }//-part
        H5Gclose(group4_id);
      }//-projector
      H5Gclose(group3_id);
    }//-its
    
    H5Gclose(group2_id);
    H5Gclose(group1_id);
    H5Fclose(file_id);

    //-Write the tail buffer
    for(int its=0;its<Nsink;its++){
      int tsink = info.tsinkSource[its];
      int l = ((t_src+tsink)%T)%Lt + 1;
      
      int sink_rank = ((t_src+tsink)%T)/Lt;
      
      if( tsink < (T - t_src%Lt) ) continue; // No need to write something else

      if(GK_timeRank==sink_rank){
        Float *tailBuf;
    
        hid_t file_idt = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

        start[0] = tsink + 1 - l;
        start[1] = 0;
        start[2] = 0;
        start[3] = 0;

        for(int ipr=0;ipr<info.Nproj[its];ipr++){
          for(int part=0;part<2;part++){
            for(int thrp_int=0;thrp_int<3;thrp_int++){
              THRP_TYPE type = (THRP_TYPE) thrp_int;
            
              //-Determine the global dimensions
              if(type==THRP_LOCAL || type==THRP_ONED) Mel = 16;
              else if (type==THRP_NOETHER) Mel = 4;
              else errorQuda("writeThrp_HDF5: Undefined three-point function type.\n");
              dims[0] = tsink+1;
              dims[1] = Nmoms;
              dims[2] = Mel;
              dims[3] = 2;

              ldims[0] = l;
              ldims[1] = Nmoms;
              ldims[2] = Mel;
              ldims[3] = 2;

              for(int imom=0;imom<GK_Nmoms;imom++){
                if(type==THRP_ONED){
                  for(int mu=0;mu<4;mu++){
                    char *group_tag;
                    asprintf(&group_tag,"conf_%04d/sx%02dsy%02dsz%02dst%02d/tsink_%02d/proj_%s/%s/%s/dir_%02d",info.traj,
                             GK_sourcePosition[isource][0],GK_sourcePosition[isource][1],GK_sourcePosition[isource][2],GK_sourcePosition[isource][3],
                             tsink, info.thrp_proj_type[info.proj_list[its][ipr]], (part==0) ? "up" : "down", info.thrp_type[thrp_int], mu);
                    hid_t group_id = H5Gopen(file_idt, group_tag, H5P_DEFAULT);

                    hid_t dset_id   = H5Dopen(group_id, "threep", H5P_DEFAULT);
                    hid_t mspace_id = H5Screate_simple(4, ldims, NULL);
                    hid_t dspace_id = H5Dget_space(dset_id);

                    H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, start, NULL, ldims, NULL);

                    tailBuf = &(((Float*)Thrp_oneD_HDF5[mu])[2*Mel*Nmoms*Lt*part + 2*Mel*Nmoms*Lt*2*its + 2*Mel*Nmoms*Lt*2*Nsink*ipr]);

                    herr_t status = H5Dwrite(dset_id, DATATYPE_H5, mspace_id, dspace_id, H5P_DEFAULT, tailBuf);

                    H5Dclose(dset_id);
                    H5Sclose(mspace_id);
                    H5Sclose(dspace_id);
                    H5Gclose(group_id);
                  }//-mu
                }
                else{
                  Float *thrpBuf;
                  if(type==THRP_LOCAL) thrpBuf = (Float*)Thrp_local_HDF5;
                  else if(type==THRP_NOETHER) thrpBuf = (Float*)Thrp_noether_HDF5;

                  char *group_tag;
                  asprintf(&group_tag,"conf_%04d/sx%02dsy%02dsz%02dst%02d/tsink_%02d/proj_%s/%s/%s",info.traj,
                           GK_sourcePosition[isource][0],GK_sourcePosition[isource][1],GK_sourcePosition[isource][2],GK_sourcePosition[isource][3],
                           tsink, info.thrp_proj_type[info.proj_list[its][ipr]], (part==0) ? "up" : "down", info.thrp_type[thrp_int]);
                  hid_t group_id = H5Gopen(file_idt, group_tag, H5P_DEFAULT);

                  hid_t dset_id   = H5Dopen(group_id, "threep", H5P_DEFAULT);
                  hid_t mspace_id = H5Screate_simple(4, ldims, NULL);
                  hid_t dspace_id = H5Dget_space(dset_id);

                  H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, start, NULL, ldims, NULL);
                
                  tailBuf = &(thrpBuf[2*Mel*Nmoms*Lt*part + 2*Mel*Nmoms*Lt*2*its + 2*Mel*Nmoms*Lt*2*Nsink*ipr]);

                  herr_t status = H5Dwrite(dset_id, DATATYPE_H5, mspace_id, dspace_id, H5P_DEFAULT, tailBuf);
                
                  H5Dclose(dset_id);
                  H5Sclose(mspace_id);
                  H5Sclose(dspace_id);
                  H5Gclose(group_id);
                }
              }//-imom
            }//-thrp_int
          }//-part
        }//-projector
        H5Fclose(file_idt);
      }//-if GK_timeRank==sink_rank
    }//-its

    //-Write the momenta in a separate dataset (including some attributes)
    if(GK_timeRank==src_rank){
      hid_t file_idt   = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

      char *operator_list,*ultra_local_info,*noether_info,*oneD_info;
      asprintf(&operator_list,"0 = g5\n1 = gx\n2 = gy\n3 = gz\n4 = g0\n5 = Unity\n6 = g5gx\n7 = g5gy\n8 = g5gz\n9 = g5g0\n10 = g5sixy\n11 = g5sixz\n12 = g5siyz\n13 = g5si0x\n14 = g5si0y\n15 = g5si0z\0");
      asprintf(&ultra_local_info,"Index-order: [t, mom-index, optr-index, real/imag]\0");
      asprintf(&noether_info,"Index-order: [t, mom-index, direction: [0,1,2,3]=[x,y,z,t], real/imag]\0");
      asprintf(&oneD_info,"Direction: [0,1,2,3] = [x,y,z,t], Index-order: [t, mom-index, optr-index, real/imag]\0");

      char *thrp_str1,*thrp_str2,*thrp_str3,*thrp_str4,*thrp_str5,*thrp_str6,*thrp_str7,*thrp_str8;
      asprintf(&thrp_str1,"Momentum-space nucleon 3pt-correlator\nQuark field and operator basis: Physical\n");
      asprintf(&thrp_str2,"Includes ultra-local and one-derivative operators, noether current\n");
      asprintf(&thrp_str3,"Precision: %s\n",(typeid(Float) == typeid(float)) ? "single" : "double");
      asprintf(&thrp_str4,"Inversion tolerance = %e\n",info.inv_tol);
      asprintf(&thrp_str5,"Operator list:\n%s\n\n",operator_list);
      asprintf(&thrp_str6,"Ultra-local correlators:\n  %s\n",ultra_local_info);
      asprintf(&thrp_str7,"Noether current correlators:\n  %s\n",noether_info);
      asprintf(&thrp_str8,"One-dericative correlators:\n  %s\n",oneD_info);

      char *cNmoms, *cQsq,*corr_info,*ens_info;
      asprintf(&cNmoms,"%d\0",GK_Nmoms);
      asprintf(&cQsq,"%d\0",info.Q_sq);
      asprintf(&corr_info,"%s%s%s%s%s%s%s%s\0",thrp_str1,thrp_str2,thrp_str3,thrp_str4,thrp_str5,thrp_str6,thrp_str7,thrp_str8);
      asprintf(&ens_info,"kappa = %10.8f\nmu = %8.6f\nCsw = %8.6f\0",info.kappa, info.mu, info.csw);
      hid_t attrdat_id1 = H5Screate(H5S_SCALAR);
      hid_t attrdat_id2 = H5Screate(H5S_SCALAR);
      hid_t attrdat_id3 = H5Screate(H5S_SCALAR);
      hid_t attrdat_id4 = H5Screate(H5S_SCALAR);
      hid_t type_id1 = H5Tcopy(H5T_C_S1);
      hid_t type_id2 = H5Tcopy(H5T_C_S1);
      hid_t type_id3 = H5Tcopy(H5T_C_S1);
      hid_t type_id4 = H5Tcopy(H5T_C_S1);
      H5Tset_size(type_id1, strlen(cNmoms));
      H5Tset_size(type_id2, strlen(cQsq));
      H5Tset_size(type_id3, strlen(corr_info));
      H5Tset_size(type_id4, strlen(ens_info));
      hid_t attr_id1 = H5Acreate2(file_idt, "Nmoms", type_id1, attrdat_id1, H5P_DEFAULT, H5P_DEFAULT);
      hid_t attr_id2 = H5Acreate2(file_idt, "Qsq"  , type_id2, attrdat_id2, H5P_DEFAULT, H5P_DEFAULT);
      hid_t attr_id3 = H5Acreate2(file_idt, "Correlator-info", type_id3, attrdat_id3, H5P_DEFAULT, H5P_DEFAULT);
      hid_t attr_id4 = H5Acreate2(file_idt, "Ensemble-info", type_id4, attrdat_id4, H5P_DEFAULT, H5P_DEFAULT);
      H5Awrite(attr_id1, type_id1, cNmoms);
      H5Awrite(attr_id2, type_id2, cQsq);
      H5Awrite(attr_id3, type_id3, corr_info);
      H5Awrite(attr_id4, type_id4, ens_info);
      H5Aclose(attr_id1);
      H5Aclose(attr_id2);
      H5Aclose(attr_id3);
      H5Aclose(attr_id4);
      H5Tclose(type_id1);
      H5Tclose(type_id2);
      H5Tclose(type_id3);
      H5Tclose(type_id4);
      H5Sclose(attrdat_id1);
      H5Sclose(attrdat_id2);
      H5Sclose(attrdat_id3);
      H5Sclose(attrdat_id4);


      hid_t MOMTYPE_H5 = H5T_NATIVE_INT;
      char *dset_tag;
      asprintf(&dset_tag,"Momenta_list_xyz");

      hsize_t Mdims[2] = {(hsize_t)Nmoms,3};
      hid_t filespace  = H5Screate_simple(2, Mdims, NULL);
      hid_t dataset_id = H5Dcreate(file_idt, dset_tag, MOMTYPE_H5, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      int *Moms_H5 = (int*) malloc(GK_Nmoms*3*sizeof(int));
      for(int im=0;im<GK_Nmoms;im++){
        for(int d=0;d<3;d++) Moms_H5[d + 3*im] = GK_moms[im][d];
      }

      herr_t status = H5Dwrite(dataset_id, MOMTYPE_H5, H5S_ALL, filespace, H5P_DEFAULT, Moms_H5);

      H5Dclose(dataset_id);
      H5Sclose(filespace);
      H5Fclose(file_idt);

      free(Moms_H5);
    }//-if GK_timeRank==0

  }//-if GK_timeRank >= 0 && GK_timeRank < GK_nProc[3]
}




//-C.K. - New function to copy the three-point data into write Buffers 
// for writing in HDF5 format
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
copyThrpToHDF5_Buf(void *Thrp_HDF5, 
		   void *corrThp,  
		   int mu, int uORd, 
		   int its, int Nsink, 
		   int pr, int sign, 
		   THRP_TYPE type, 
		   CORR_SPACE CorrSpace, bool HighMomForm){

  int Mel;
  if(type==THRP_LOCAL || type==THRP_ONED) Mel = 16;
  else if(type==THRP_NOETHER) Mel = 4;
  else errorQuda("Undefined THRP_TYPE passed to copyThrpToHDF5_Buf.\n");

  int Lt = GK_localL[3];
  int Nmoms = GK_Nmoms;

  if(CorrSpace==MOMENTUM_SPACE){

    if(HighMomForm){
      if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
        if(type==THRP_LOCAL || type==THRP_NOETHER){
          for(int it = 0; it<Lt; it++){
            for(int imom = 0; imom<Nmoms; imom++){
              for(int im = 0; im<Mel; im++){
                ((Float*)Thrp_HDF5)[0 + 2*im + 2*Mel*imom + 2*Mel*Nmoms*it + 2*Mel*Nmoms*Lt*uORd + 2*Mel*Nmoms*Lt*2*its + 2*Mel*Nmoms*Lt*2*Nsink*pr] = sign*((Float*)corrThp)[0 + 2*im + 2*Mel*imom + 2*Mel*Nmoms*it];
                ((Float*)Thrp_HDF5)[1 + 2*im + 2*Mel*imom + 2*Mel*Nmoms*it + 2*Mel*Nmoms*Lt*uORd + 2*Mel*Nmoms*Lt*2*its + 2*Mel*Nmoms*Lt*2*Nsink*pr] = sign*((Float*)corrThp)[1 + 2*im + 2*Mel*imom + 2*Mel*Nmoms*it];
              }
            }
          }
        }
        else if(type==THRP_ONED){
          for(int it = 0; it<Lt; it++){
            for(int imom = 0; imom<Nmoms; imom++){
              for(int im = 0; im<Mel; im++){
                ((Float*)Thrp_HDF5)[0 + 2*im + 2*Mel*imom + 2*Mel*Nmoms*it + 2*Mel*Nmoms*Lt*uORd + 2*Mel*Nmoms*Lt*2*its + 2*Mel*Nmoms*Lt*2*Nsink*pr] = sign*((Float*)corrThp)[0 + 2*im + 2*Mel*mu + 2*Mel*4*imom + 2*Mel*4*Nmoms*it];
                ((Float*)Thrp_HDF5)[1 + 2*im + 2*Mel*imom + 2*Mel*Nmoms*it + 2*Mel*Nmoms*Lt*uORd + 2*Mel*Nmoms*Lt*2*its + 2*Mel*Nmoms*Lt*2*Nsink*pr] = sign*((Float*)corrThp)[1 + 2*im + 2*Mel*mu + 2*Mel*4*imom + 2*Mel*4*Nmoms*it];
              }
            }
          }      
        }
      }//-if GK_timeRank

    }//-if HighMomForm
    else{
      if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
        if(type==THRP_LOCAL || type==THRP_NOETHER){
          for(int it = 0; it<Lt; it++){
            for(int imom = 0; imom<Nmoms; imom++){
              for(int im = 0; im<Mel; im++){
                ((Float*)Thrp_HDF5)[0 + 2*im + 2*Mel*it + 2*Mel*Lt*imom + 2*Mel*Lt*Nmoms*uORd + 2*Mel*Lt*Nmoms*2*its + 2*Mel*Lt*Nmoms*2*Nsink*pr] = sign*((Float*)corrThp)[0 + 2*im + 2*Mel*imom + 2*Mel*Nmoms*it];
                ((Float*)Thrp_HDF5)[1 + 2*im + 2*Mel*it + 2*Mel*Lt*imom + 2*Mel*Lt*Nmoms*uORd + 2*Mel*Lt*Nmoms*2*its + 2*Mel*Lt*Nmoms*2*Nsink*pr] = sign*((Float*)corrThp)[1 + 2*im + 2*Mel*imom + 2*Mel*Nmoms*it];
              }
            }
          }
        }
        else if(type==THRP_ONED){
          for(int it = 0; it<Lt; it++){
            for(int imom = 0; imom<Nmoms; imom++){
              for(int im = 0; im<Mel; im++){
                ((Float*)Thrp_HDF5)[0 + 2*im + 2*Mel*it + 2*Mel*Lt*imom + 2*Mel*Lt*Nmoms*uORd + 2*Mel*Lt*Nmoms*2*its + 2*Mel*Lt*Nmoms*2*Nsink*pr] = sign*((Float*)corrThp)[0 + 2*im + 2*Mel*mu + 2*Mel*4*imom + 2*Mel*4*Nmoms*it];
                ((Float*)Thrp_HDF5)[1 + 2*im + 2*Mel*it + 2*Mel*Lt*imom + 2*Mel*Lt*Nmoms*uORd + 2*Mel*Lt*Nmoms*2*its + 2*Mel*Lt*Nmoms*2*Nsink*pr] = sign*((Float*)corrThp)[1 + 2*im + 2*Mel*mu + 2*Mel*4*imom + 2*Mel*4*Nmoms*it];
              }
            }
          }      
        }
      }//-if GK_timeRank    
    }//-else HighMomForm

  }//-if CorrSpace
  else if(CorrSpace==POSITION_SPACE){
    int lV = GK_localVolume;
    Float *tmp3pt;
    if(type==THRP_LOCAL || type==THRP_NOETHER) tmp3pt = ((Float*)corrThp);
    else if(type==THRP_ONED) tmp3pt = &(((Float*)corrThp)[2*16*lV*mu]);

    for(int v = 0; v<lV; v++){
      for(int im = 0; im<Mel; im++){
        ((Float*)Thrp_HDF5)[0 + 2*v + 2*lV*uORd + 2*lV*2*im + 2*lV*2*Mel*its + 2*lV*2*Mel*Nsink*pr] = sign*tmp3pt[0 + 2*im + 2*Mel*v];
        ((Float*)Thrp_HDF5)[1 + 2*v + 2*lV*uORd + 2*lV*2*im + 2*lV*2*Mel*its + 2*lV*2*Mel*Nsink*pr] = sign*tmp3pt[1 + 2*im + 2*Mel*v];
      }
    }
  }//-else if

}


//-C.K. - New function to write the three-point function in ASCII format
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
writeThrp_ASCII(void *corrThp_local, 
		void *corrThp_noether,
		void *corrThp_oneD, 
		WHICHPARTICLE testParticle, 
		int partflag , 
		char *filename_out, 
		int isource, 
		int tsinkMtsource, 
		CORR_SPACE CorrSpace){
  
  if(CorrSpace!=MOMENTUM_SPACE) 
    errorQuda("writeThrp_ASCII: Supports writing only in momentum-space!\n");
  
  Float *GLcorrThp_local   = 
    (Float*) calloc(GK_totalL[3]*GK_Nmoms*16  *2,sizeof(Float));
  Float *GLcorrThp_noether = 
    (Float*) calloc(GK_totalL[3]*GK_Nmoms   *4*2,sizeof(Float));
  Float *GLcorrThp_oneD    = 
    (Float*) calloc(GK_totalL[3]*GK_Nmoms*16*4*2,sizeof(Float));
  if(GLcorrThp_local == NULL || 
     GLcorrThp_noether == NULL || 
     GLcorrThp_oneD == NULL) 
    errorQuda("writeThrp_ASCII: Cannot allocate memory for write Buffers.");

  MPI_Datatype DATATYPE = -1;
  if( typeid(Float) == typeid(float)){
    DATATYPE = MPI_FLOAT;
    printfQuda("writeThrp_ASCII: Will write in single precision\n");
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE = MPI_DOUBLE;
    printfQuda("writeThrp_ASCII: Will write in double precision\n");
  }

  int error;
  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
    error = MPI_Gather((Float*)corrThp_local,GK_localL[3]*GK_Nmoms*16*2, 
		       DATATYPE, GLcorrThp_local, 
		       GK_localL[3]*GK_Nmoms*16*2, 
		       DATATYPE, 0, GK_timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");

    error = MPI_Gather((Float*)corrThp_noether,GK_localL[3]*GK_Nmoms*4*2, 
		       DATATYPE, GLcorrThp_noether, 
		       GK_localL[3]*GK_Nmoms*4*2, 
		       DATATYPE, 0, GK_timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");

    error = MPI_Gather((Float*)corrThp_oneD,GK_localL[3]*GK_Nmoms*4*16*2, 
		       DATATYPE, GLcorrThp_oneD, 
		       GK_localL[3]*GK_Nmoms*4*16*2, 
		       DATATYPE, 0, GK_timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
  }

  char fname_local[257];
  char fname_noether[257];
  char fname_oneD[257];
  char fname_particle[257];
  char fname_upORdown[257];

  if(testParticle == PROTON){
    strcpy(fname_particle,"proton");
    if(partflag == 1)strcpy(fname_upORdown,"up");
    else if(partflag == 2)strcpy(fname_upORdown,"down");
    else errorQuda("writeThrp_ASCII: Got the wrong part! Should be either 1 or 2.");
  }
  else{
    strcpy(fname_particle,"neutron");
    if(partflag == 1)strcpy(fname_upORdown,"down");
    else if(partflag == 2)strcpy(fname_upORdown,"up");
    else errorQuda("writeThrp_ASCII: Got the wrong part! Should be either 1 or 2.");
  }

  sprintf(fname_local,"%s.%s.%s.%s.SS.%02d.%02d.%02d.%02d.dat",
	  filename_out,fname_particle,fname_upORdown,"ultra_local",
	  GK_sourcePosition[isource][0],
	  GK_sourcePosition[isource][1],
	  GK_sourcePosition[isource][2],
	  GK_sourcePosition[isource][3]);
  sprintf(fname_noether,"%s.%s.%s.%s.SS.%02d.%02d.%02d.%02d.dat",
	  filename_out,fname_particle,fname_upORdown,"noether",
	  GK_sourcePosition[isource][0],
	  GK_sourcePosition[isource][1],
	  GK_sourcePosition[isource][2],
	  GK_sourcePosition[isource][3]);
  sprintf(fname_oneD,"%s.%s.%s.%s.SS.%02d.%02d.%02d.%02d.dat",
	  filename_out,fname_particle,fname_upORdown,"oneD",
	  GK_sourcePosition[isource][0],
	  GK_sourcePosition[isource][1],
	  GK_sourcePosition[isource][2],
	  GK_sourcePosition[isource][3]);

  FILE *ptr_local = NULL;
  FILE *ptr_noether = NULL;
  FILE *ptr_oneD = NULL;

  if( comm_rank() == 0 ){
    ptr_local = fopen(fname_local,"w");
    ptr_noether = fopen(fname_noether,"w");
    ptr_oneD = fopen(fname_oneD,"w");
    // local //
    for(int iop = 0 ; iop < 16 ; iop++)
      for(int it = 0 ; it < GK_totalL[3] ; it++)
	for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	  int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	  int sign = (tsinkMtsource+GK_sourcePosition[isource][3]) 
	    >= GK_totalL[3] ? -1 : +1;
	  fprintf(ptr_local,"%d \t %d \t %+d %+d %+d \t %+e %+e\n", 
		  iop, it, 
		  GK_moms[imom][0],
		  GK_moms[imom][1],
		  GK_moms[imom][2],
		  sign*GLcorrThp_local[it_shift*GK_Nmoms*16*2 + 
				       imom*16*2 + iop*2 + 0], 
		  sign*GLcorrThp_local[it_shift*GK_Nmoms*16*2 + 
				       imom*16*2 + iop*2 + 1]);
	}
    // noether //
    for(int iop = 0 ; iop < 4 ; iop++)
      for(int it = 0 ; it < GK_totalL[3] ; it++)
	for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	  int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	  int sign = (tsinkMtsource+GK_sourcePosition[isource][3]) 
	    >= GK_totalL[3] ? -1 : +1;
	  fprintf(ptr_noether,"%d \t %d \t %+d %+d %+d \t %+e %+e\n", 
		  iop, it, 
		  GK_moms[imom][0],
		  GK_moms[imom][1],
		  GK_moms[imom][2],
		  sign*GLcorrThp_noether[it_shift*GK_Nmoms*4*2 + 
					 imom*4*2 + iop*2 + 0], 
		  sign*GLcorrThp_noether[it_shift*GK_Nmoms*4*2 + 
					 imom*4*2 + iop*2 + 1]);
	}
    // oneD //
    for(int iop = 0 ; iop < 16 ; iop++)
      for(int dir = 0 ; dir < 4 ; dir++)
	for(int it = 0 ; it < GK_totalL[3] ; it++)
	  for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	    int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	    int sign = (tsinkMtsource+GK_sourcePosition[isource][3]) 
	      >= GK_totalL[3] ? -1 : +1;
	    fprintf(ptr_oneD,"%d \t %d \t %d \t %+d %+d %+d \t %+e %+e\n", 
		    iop, dir, it, 
		    GK_moms[imom][0],
		    GK_moms[imom][1],
		    GK_moms[imom][2],
		    sign*GLcorrThp_oneD[it_shift*GK_Nmoms*4*16*2 + 
					imom*4*16*2 + dir*16*2 + iop*2 + 0], 
		    sign*GLcorrThp_oneD[it_shift*GK_Nmoms*4*16*2 + 
					imom*4*16*2 + dir*16*2 + iop*2 + 1]);
	  }
    fclose(ptr_local);
    fclose(ptr_noether);
    fclose(ptr_oneD);
  }

  free(GLcorrThp_local);
  free(GLcorrThp_noether);
  free(GLcorrThp_oneD);
}

//-C.K. Overloaded function to perform the contractions without 
// writing the data
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
contractFixSink(QKXTM_Propagator_Kepler<Float> &seqProp,
		QKXTM_Propagator_Kepler<Float> &prop, 
		QKXTM_Gauge_Kepler<Float> &gauge, 
		void *corrThp_local, void *corrThp_noether, 
		void *corrThp_oneD, 
		WHICHPROJECTOR typeProj , 
		WHICHPARTICLE testParticle, 
		int partflag, int isource, 
		CORR_SPACE CorrSpace){
  
  if( typeid(Float) == typeid(float))  
    printfQuda("contractFixSink: Will perform in single precision\n");
  if( typeid(Float) == typeid(double)) 
    printfQuda("contractFixSink: Will perform in double precision\n");
  
  // seq prop apply gamma5 and conjugate
  seqProp.apply_gamma5();
  seqProp.conjugate();

  gauge.ghostToHost();
  // communicate gauge
  gauge.cpuExchangeGhost(); 
  gauge.ghostToDevice();
  comm_barrier();

  prop.ghostToHost();
  // communicate forward propagator
  prop.cpuExchangeGhost(); 
  prop.ghostToDevice();
  comm_barrier();

  seqProp.ghostToHost();
  // communicate sequential propagator
  seqProp.cpuExchangeGhost();
  seqProp.ghostToDevice();
  comm_barrier();

  cudaTextureObject_t seqTex, fwdTex, gaugeTex;
  seqProp.createTexObject(&seqTex);
  prop.createTexObject(&fwdTex);
  gauge.createTexObject(&gaugeTex);

  if(CorrSpace==POSITION_SPACE){
    for(int it = 0 ; it < GK_localL[3] ; it++)
      run_fixSinkContractions((void*)corrThp_local, 
			      (void*)corrThp_noether, 
			      (void*)corrThp_oneD, 
			      fwdTex, seqTex, gaugeTex, 
			      testParticle, 
			      partflag, it, isource, 
			      sizeof(Float), CorrSpace);
  }
  else if(CorrSpace==MOMENTUM_SPACE){
    Float *corrThp_local_local   = (Float*) calloc(GK_localL[3]*
						   GK_Nmoms*16*2,
						   sizeof(Float));

    Float *corrThp_noether_local = (Float*) calloc(GK_localL[3]*
						   GK_Nmoms*4*2,
						   sizeof(Float));

    Float *corrThp_oneD_local    = (Float*) calloc(GK_localL[3]*
						   GK_Nmoms*16*4*2,
						   sizeof(Float));
    
    if(corrThp_local_local == NULL || 
       corrThp_noether_local == NULL || 
       corrThp_oneD_local == NULL) 
      errorQuda("contractFixSink: Cannot allocate memory for three-point function contract buffers.\n");
    
    for(int it = 0 ; it < GK_localL[3] ; it++)
      run_fixSinkContractions(corrThp_local_local, 
			      corrThp_noether_local, 
			      corrThp_oneD_local, 
			      fwdTex, seqTex, gaugeTex, 
			      testParticle, 
			      partflag, it, isource, 
			      sizeof(Float), CorrSpace);

    MPI_Datatype DATATYPE = -1;
    if( typeid(Float) == typeid(float))  DATATYPE = MPI_FLOAT;
    if( typeid(Float) == typeid(double)) DATATYPE = MPI_DOUBLE;
    
    MPI_Reduce(corrThp_local_local, (Float*)corrThp_local, 
	       GK_localL[3]*GK_Nmoms*16*2, DATATYPE, 
	       MPI_SUM, 0, GK_spaceComm);

    MPI_Reduce(corrThp_noether_local, (Float*)corrThp_noether, 
	       GK_localL[3]*GK_Nmoms*4*2, DATATYPE, 
	       MPI_SUM, 0, GK_spaceComm);

    MPI_Reduce(corrThp_oneD_local, (Float*)corrThp_oneD, 
	       GK_localL[3]*GK_Nmoms*16*4*2, DATATYPE, 
	       MPI_SUM, 0, GK_spaceComm);
    
    free(corrThp_local_local);
    free(corrThp_noether_local);
    free(corrThp_oneD_local);
  }
  else errorQuda("contractFixSink: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

  seqProp.destroyTexObject(seqTex);
  prop.destroyTexObject(fwdTex);
  gauge.destroyTexObject(gaugeTex);

}

//---------------------//

template<typename Float>
void QKXTM_Contraction_Kepler<Float>::
contractFixSink(QKXTM_Propagator_Kepler<Float> &seqProp,
		QKXTM_Propagator_Kepler<Float> &prop, 
		QKXTM_Gauge_Kepler<Float> &gauge, 
		WHICHPROJECTOR typeProj , 
		WHICHPARTICLE testParticle, 
		int partflag , 
		char *filename_out, 
		int isource, 
		int tsinkMtsource){
  
  errorQuda("contractFixSink: This version of the function is obsolete. Cannot guarantee correct results. Please call the overloaded-updated version of this function with the corresponding list of arguments.\n");

  if( typeid(Float) == typeid(float))  
    printfQuda("contractFixSink: Will perform in single precision\n");
  if( typeid(Float) == typeid(double)) 
    printfQuda("contractFixSink: Will perform in double precision\n");

  // seq prop apply gamma5 and conjugate
  // do the communication for gauge, prop and seqProp
  seqProp.apply_gamma5();
  seqProp.conjugate();

  gauge.ghostToHost();

  // communicate gauge
  gauge.cpuExchangeGhost();
  gauge.ghostToDevice();
  comm_barrier();

  prop.ghostToHost();

  // communicate forward propagator
  prop.cpuExchangeGhost(); 
  prop.ghostToDevice();
  comm_barrier();

  seqProp.ghostToHost();

  // communicate sequential propagator
  seqProp.cpuExchangeGhost(); 
  seqProp.ghostToDevice();
  comm_barrier();

  cudaTextureObject_t seqTex, fwdTex, gaugeTex;
  seqProp.createTexObject(&seqTex);
  prop.createTexObject(&fwdTex);
  gauge.createTexObject(&gaugeTex);

  Float *corrThp_local_local = 
    (Float*) calloc(GK_localL[3]*GK_Nmoms*16*2,sizeof(Float));
  Float *corrThp_noether_local = 
    (Float*) calloc(GK_localL[3]*GK_Nmoms*4*2,sizeof(Float));
  Float *corrThp_oneD_local = 
    (Float*) calloc(GK_localL[3]*GK_Nmoms*4*16*2,sizeof(Float));
  if(corrThp_local_local == NULL || 
     corrThp_noether_local == NULL || 
     corrThp_oneD_local == NULL) 
    errorQuda("Error problem to allocate memory");

  Float *corrThp_local_reduced = 
    (Float*) calloc(GK_localL[3]*GK_Nmoms*16*2,sizeof(Float));
  Float *corrThp_noether_reduced = 
    (Float*) calloc(GK_localL[3]*GK_Nmoms*4*2,sizeof(Float));
  Float *corrThp_oneD_reduced = 
    (Float*) calloc(GK_localL[3]*GK_Nmoms*4*16*2,sizeof(Float));
  if(corrThp_local_reduced == NULL || 
     corrThp_noether_reduced == NULL || 
     corrThp_oneD_reduced == NULL) 
    errorQuda("Error problem to allocate memory");

  Float *corrThp_local = 
    (Float*) calloc(GK_totalL[3]*GK_Nmoms*16*2,sizeof(Float));
  Float *corrThp_noether = 
    (Float*) calloc(GK_totalL[3]*GK_Nmoms*4*2,sizeof(Float));
  Float *corrThp_oneD = 
    (Float*) calloc(GK_totalL[3]*GK_Nmoms*4*16*2,sizeof(Float));
  if(corrThp_local == NULL || 
     corrThp_noether == NULL || 
     corrThp_oneD == NULL) 
    errorQuda("Error problem to allocate memory");

  for(int it = 0 ; it < GK_localL[3] ; it++)
    run_fixSinkContractions(corrThp_local_local,
			    corrThp_noether_local,
			    corrThp_oneD_local,
			    fwdTex,seqTex,gaugeTex,
			    testParticle,partflag,it,
			    isource,sizeof(Float),MOMENTUM_SPACE);
  
  int error;
  if( typeid(Float) == typeid(float)){
    MPI_Reduce(corrThp_local_local, 
	       corrThp_local_reduced, 
	       GK_localL[3]*GK_Nmoms*16*2, 
	       MPI_FLOAT, MPI_SUM, 0, GK_spaceComm);

    MPI_Reduce(corrThp_noether_local, 
	       corrThp_noether_reduced, 
	       GK_localL[3]*GK_Nmoms*4*2, 
	       MPI_FLOAT, MPI_SUM, 0, GK_spaceComm);

    MPI_Reduce(corrThp_oneD_local, 
	       corrThp_oneD_reduced, 
	       GK_localL[3]*GK_Nmoms*4*16*2, 
	       MPI_FLOAT, MPI_SUM, 0, GK_spaceComm);

    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corrThp_local_reduced,
			 GK_localL[3]*GK_Nmoms*16*2, 
			 MPI_FLOAT, corrThp_local, 
			 GK_localL[3]*GK_Nmoms*16*2, 
			 MPI_FLOAT, 0, GK_timeComm);
      if(error != MPI_SUCCESS) 
	errorQuda("Error in MPI_gather");

      error = MPI_Gather(corrThp_noether_reduced,
			 GK_localL[3]*GK_Nmoms*4*2, 
			 MPI_FLOAT, corrThp_noether, 
			 GK_localL[3]*GK_Nmoms*4*2, 
			 MPI_FLOAT, 0, GK_timeComm);
      if(error != MPI_SUCCESS) 
	errorQuda("Error in MPI_gather");

      error = MPI_Gather(corrThp_oneD_reduced,
			 GK_localL[3]*GK_Nmoms*4*16*2, 
			 MPI_FLOAT, corrThp_oneD, 
			 GK_localL[3]*GK_Nmoms*4*16*2, 
			 MPI_FLOAT, 0, GK_timeComm);
      if(error != MPI_SUCCESS) 
	errorQuda("Error in MPI_gather");
    }
  }
  else{
    MPI_Reduce(corrThp_local_local, corrThp_local_reduced, 
	       GK_localL[3]*GK_Nmoms*16*2, 
	       MPI_DOUBLE, MPI_SUM, 0, GK_spaceComm);
    MPI_Reduce(corrThp_noether_local, corrThp_noether_reduced, 
	       GK_localL[3]*GK_Nmoms*4*2, 
	       MPI_DOUBLE, MPI_SUM, 0, GK_spaceComm);
    MPI_Reduce(corrThp_oneD_local, corrThp_oneD_reduced, 
	       GK_localL[3]*GK_Nmoms*4*16*2, 
	       MPI_DOUBLE, MPI_SUM, 0, GK_spaceComm);

    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corrThp_local_reduced,
			 GK_localL[3]*GK_Nmoms*16*2, 
			 MPI_DOUBLE, corrThp_local, 
			 GK_localL[3]*GK_Nmoms*16*2, 
			 MPI_DOUBLE, 0, GK_timeComm);
      if(error != MPI_SUCCESS) 
	errorQuda("Error in MPI_gather");

      error = MPI_Gather(corrThp_noether_reduced,
			 GK_localL[3]*GK_Nmoms*4*2, 
			 MPI_DOUBLE, corrThp_noether, 
			 GK_localL[3]*GK_Nmoms*4*2, 
			 MPI_DOUBLE, 0, GK_timeComm);
      if(error != MPI_SUCCESS) 
	errorQuda("Error in MPI_gather");

      error = MPI_Gather(corrThp_oneD_reduced,
			 GK_localL[3]*GK_Nmoms*4*16*2, 
			 MPI_DOUBLE, corrThp_oneD, 
			 GK_localL[3]*GK_Nmoms*4*16*2, 
			 MPI_DOUBLE, 0, GK_timeComm);
      if(error != MPI_SUCCESS) 
	errorQuda("Error in MPI_gather");
    }
  }
  char fname_local[257];
  char fname_noether[257];
  char fname_oneD[257];
  char fname_particle[257];
  char fname_upORdown[257];

  if(testParticle == PROTON){
    strcpy(fname_particle,"proton");
    if(partflag == 1)strcpy(fname_upORdown,"up");
    else if(partflag == 2)strcpy(fname_upORdown,"down");
    else errorQuda("Error wrong part got");
  }
  else{
    strcpy(fname_particle,"neutron");
    if(partflag == 1)strcpy(fname_upORdown,"down");
    else if(partflag == 2)strcpy(fname_upORdown,"up");
    else errorQuda("Error wrong part got");
  }

  sprintf(fname_local,"%s.%s.%s.%s.SS.%02d.%02d.%02d.%02d.dat",
	  filename_out,fname_particle,fname_upORdown,"ultra_local",
	  GK_sourcePosition[isource][0],
	  GK_sourcePosition[isource][1],
	  GK_sourcePosition[isource][2],
	  GK_sourcePosition[isource][3]);
  sprintf(fname_noether,"%s.%s.%s.%s.SS.%02d.%02d.%02d.%02d.dat",
	  filename_out,fname_particle,fname_upORdown,"noether",
	  GK_sourcePosition[isource][0],
	  GK_sourcePosition[isource][1],
	  GK_sourcePosition[isource][2],
	  GK_sourcePosition[isource][3]);
  sprintf(fname_oneD,"%s.%s.%s.%s.SS.%02d.%02d.%02d.%02d.dat",
	  filename_out,fname_particle,fname_upORdown,"oneD",
	  GK_sourcePosition[isource][0],
	  GK_sourcePosition[isource][1],
	  GK_sourcePosition[isource][2],
	  GK_sourcePosition[isource][3]);
  
  FILE *ptr_local = NULL;
  FILE *ptr_noether = NULL;
  FILE *ptr_oneD = NULL;

  if( comm_rank() == 0 ){
    ptr_local = fopen(fname_local,"w");
    ptr_noether = fopen(fname_noether,"w");
    ptr_oneD = fopen(fname_oneD,"w");
    // local //
    for(int iop = 0 ; iop < 16 ; iop++)
      for(int it = 0 ; it < GK_totalL[3] ; it++)
	for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	  int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	  int sign = (tsinkMtsource+GK_sourcePosition[isource][3]) 
	    >= GK_totalL[3] ? -1 : +1;
	  fprintf(ptr_local,"%d \t %d \t %+d %+d %+d \t %+e %+e\n", 
		  iop, it, 
		  GK_moms[imom][0],
		  GK_moms[imom][1],
		  GK_moms[imom][2],
		  sign*corrThp_local[it_shift*GK_Nmoms*16*2 + 
				     imom*16*2 + iop*2 + 0], 
		  sign*corrThp_local[it_shift*GK_Nmoms*16*2 + 
				     imom*16*2 + iop*2 + 1]);
	}
    // noether //
    for(int iop = 0 ; iop < 4 ; iop++)
      for(int it = 0 ; it < GK_totalL[3] ; it++)
	for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	  int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	  int sign = (tsinkMtsource+GK_sourcePosition[isource][3]) 
	    >= GK_totalL[3] ? -1 : +1;
	  fprintf(ptr_noether,"%d \t %d \t %+d %+d %+d \t %+e %+e\n", 
		  iop, it, 
		  GK_moms[imom][0],
		  GK_moms[imom][1],
		  GK_moms[imom][2],
		  sign*corrThp_noether[it_shift*GK_Nmoms*4*2 + 
				       imom*4*2 + iop*2 + 0], 
		  sign*corrThp_noether[it_shift*GK_Nmoms*4*2 + 
				       imom*4*2 + iop*2 + 1]);
	}
    // oneD //
    for(int iop = 0 ; iop < 16 ; iop++)
      for(int dir = 0 ; dir < 4 ; dir++)
	for(int it = 0 ; it < GK_totalL[3] ; it++)
	  for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	    int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	    int sign = (tsinkMtsource+GK_sourcePosition[isource][3]) 
	      >= GK_totalL[3] ? -1 : +1;
	    fprintf(ptr_oneD,"%d \t %d \t %d \t %+d %+d %+d \t %+e %+e\n", 
		    iop, dir, it, 
		    GK_moms[imom][0],
		    GK_moms[imom][1],
		    GK_moms[imom][2],
		    sign*corrThp_oneD[it_shift*GK_Nmoms*4*16*2 + 
				      imom*4*16*2 + dir*16*2 + iop*2 + 0], 
		    sign*corrThp_oneD[it_shift*GK_Nmoms*4*16*2 + 
				      imom*4*16*2 + dir*16*2 + iop*2 + 1]);
	  }
    fclose(ptr_local);
    fclose(ptr_noether);
    fclose(ptr_oneD);
  }

  free(corrThp_local_local);
  free(corrThp_local_reduced);
  free(corrThp_local);

  free(corrThp_noether_local);
  free(corrThp_noether_reduced);
  free(corrThp_noether);

  free(corrThp_oneD_local);
  free(corrThp_oneD_reduced);
  free(corrThp_oneD);

  seqProp.destroyTexObject(seqTex);
  prop.destroyTexObject(fwdTex);
  gauge.destroyTexObject(gaugeTex);

}
