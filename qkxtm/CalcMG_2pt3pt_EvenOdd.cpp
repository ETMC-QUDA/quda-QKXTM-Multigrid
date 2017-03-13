#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <QKXTM_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#include "face_quda.h"

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <gauge_qio.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <qudaQKXTM_Kepler.h>
// Wilson, clover-improved Wilson, twisted mass, and domain wall are 
// supported.

//========================================================================//
//====== P A R A M E T E R   S E T T I N G S   A N D   C H E C K S =======//
//========================================================================//

extern QudaDslashType dslash_type;
extern bool tune;
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int Lsdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaPrecision  prec_sloppy;
extern QudaPrecision  prec_precondition;
extern QudaReconstructType link_recon_sloppy;
extern QudaReconstructType link_recon_precondition;
extern double mass; // mass of Dirac operator
extern double mu;
//QKXTM: DMH Experimental MG additions
extern double delta_muPR;
extern double delta_kappaPR;
extern double delta_cswPR;
extern double delta_muCG;
extern double delta_kappaCG;
extern double delta_cswCG;

extern double anisotropy;
extern double tol; // tolerance for inverter
extern double tol_hq; // heavy-quark tolerance for inverter
extern QudaMassNormalization normalization; // mass normalization of Dirac operators

extern int mg_setup_maxiter;
extern double mg_setup_tol;

extern int niter;
extern int nvec[];
extern int mg_levels;

extern bool generate_nullspace;
extern bool generate_all_levels;
extern int nu_pre;
extern int nu_post;
extern int geo_block_size[QUDA_MAX_MG_LEVEL][QUDA_MAX_DIM];

extern QudaInverterType smoother_type;

extern QudaMatPCType matpc_type;
extern QudaSolveType solve_type;

extern char vec_infile[];
extern char vec_outfile[];

//Twisted mass flavor type
extern QudaTwistFlavorType twist_flavor;

extern void usage(char** );

extern double clover_coeff;
extern bool compute_clover;

//QKXTM: DMH extra inputs
extern char latfile[];
extern char latfile_smeared[];
extern char verbosity_level[];
extern int traj;
extern bool isEven;
extern int src[];
extern int Ntsink;
extern char pathList_tsink[];
extern int Q_sq;
extern int nsmearAPE;
extern int nsmearGauss;
extern double alphaAPE;
extern double alphaGauss;
extern char twop_filename[];
extern char threep_filename[];

extern double kappa;
extern char prop_path[];
extern double csw;

extern int numSourcePositions;
extern char pathListSourcePositions[];
extern char pathListRun3pt[];
extern char run3pt[];
extern char *corr_file_format;
extern char check_file_exist[];

extern int Nproj;
extern char proj_list_file[];

extern char *corr_write_space;

namespace quda {
  extern void setTransferGPU(bool);
}

void display_test_info() {
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n",
	     get_prec_str(prec),get_prec_str(prec_sloppy),
	     get_recon_str(link_recon), 
	     get_recon_str(link_recon_sloppy),  xdim, ydim, zdim, tdim, Lsdim);     

  printfQuda("MG parameters\n");
  printfQuda(" - number of levels %d\n", mg_levels);
  for (int i=0; i<mg_levels-1; i++) printfQuda(" - level %d number of null-space vectors %d\n", i+1, nvec[i]);
  printfQuda(" - number of pre-smoother applications %d\n", nu_pre);
  printfQuda(" - number of post-smoother applications %d\n", nu_post);

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3)); 
  
  return ;
  
}

QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;

void setGaugeParam(QudaGaugeParam &gauge_param) {
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = anisotropy;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;

  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;

  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;

  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_precondition;

  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = 0;
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif
}

void setInvertParam(QudaInvertParam &inv_param) {

  inv_param.kappa = kappa;
  inv_param.mass = 0.5/kappa - 4.0;

  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;

  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_UKQCD_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = csw*inv_param.kappa;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;

  inv_param.dslash_type = dslash_type;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 
      2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = normalization;
  inv_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;

  inv_param.pipeline = 0;

  // do we want to use an even-odd preconditioned solve or not
  if(isEven) {
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    printf("### Running for the Even-Even Operator\n");
  }
  else {
    printf("### Running for the Odd-Odd Operator\n");
    inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
  }

  // do we want full solution or single-parity solution
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  
  inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;

  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.verbosity = QUDA_VERBOSE;
  inv_param.verbosity_precondition = QUDA_SILENT;

  inv_param.inv_type_precondition = QUDA_MG_INVERTER;

  inv_param.Nsteps = 20;
  inv_param.gcrNkrylov = 10;
  inv_param.tol = tol;
  inv_param.tol_restart = 1e-3;

  // require both L2 relative and heavy quark residual to determine 
  // convergence
  inv_param.residual_type = 
    static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
  // specify a tolerance for the residual for heavy quark residual
  inv_param.tol_hq = tol_hq; 
  
  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta = 1e-2;

  // domain decomposition preconditioner parameters
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 10;
  inv_param.omega = 1.0;


  //if(strcmp(verbosity_level,"verbose")==0) 
  //inv_param.verbosity = QUDA_VERBOSE;
  //else if(strcmp(verbosity_level,"summarize")==0) 
  //inv_param.verbosity = QUDA_SUMMARIZE;
  //else if(strcmp(verbosity_level,"silent")==0) 
  //inv_param.verbosity = QUDA_SILENT;
  //else{
  //warningQuda("Unknown verbosity level %s. Proceeding with QUDA_SUMMARIZE verbosity level\n",verbosity_level);
  //inv_param.verbosity = QUDA_SUMMARIZE;
  //}
}


void setMultigridParam(QudaMultigridParam &mg_param) {
  QudaInvertParam &inv_param = *mg_param.invert_param;

  inv_param.kappa = kappa;
  inv_param.mass = 0.5/kappa - 4.0;

  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = csw*inv_param.kappa;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;

  inv_param.dslash_type = dslash_type;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    mg_param.delta_muPR = delta_muPR;
    mg_param.delta_muCG = delta_muCG;
    mg_param.delta_kappaPR = delta_kappaPR;
    mg_param.delta_kappaCG = delta_kappaCG;

    //inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 
      2 : 1;
    
    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }
  
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = normalization;
  inv_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;

  // do we want to use an even-odd preconditioned solve or not
  if(isEven) inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  else inv_param.matpc_type = QUDA_MATPC_ODD_ODD;

  inv_param.solution_type = QUDA_MAT_SOLUTION;

  inv_param.solve_type = QUDA_DIRECT_SOLVE;

  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels;
  for (int i=0; i<mg_param.n_level; i++) {
    for (int j=0; j<QUDA_MAX_DIM; j++) {
	// if not defined use 4
      mg_param.geo_block_size[i][j] = geo_block_size[i][j] ? 
	geo_block_size[i][j] : 4;      
    }
    mg_param.spin_block_size[i] = 1;
    
    //QKXTM: DMH Develop branch code
    // default to 24 vectors if not set
    mg_param.n_vec[i] = nvec[i] == 0 ? 24 : nvec[i]; 
    mg_param.nu_pre[i] = nu_pre;
    mg_param.nu_post[i] = nu_post;
    
    mg_param.cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;
    
    mg_param.smoother[i] = smoother_type;

    mg_param.setup_maxiter = mg_setup_maxiter;
    mg_param.setup_tol = mg_setup_tol;

    // set the smoother / bottom solver tolerance 
    // (for MR smoothing this will be ignored)
    // repurpose heavy-quark tolerance for now

    mg_param.smoother_tol[i] = tol_hq; 
    mg_param.global_reduction[i] = QUDA_BOOLEAN_YES;

    // set to QUDA_DIRECT_PC_SOLVE for to enable even/odd 
    // preconditioning on the smoother
    mg_param.smoother_solve_type[i] = QUDA_DIRECT_PC_SOLVE; // EVEN-ODD
    
    // set to QUDA_MAT_SOLUTION to inject a full field into coarse grid
    // set to QUDA_MATPC_SOLUTION to inject single parity field into 
    // coarse grid

    // if we are using an outer even-odd preconditioned solve, then we
    // use single parity injection into the coarse grid
    mg_param.coarse_grid_solution_type[i] = solve_type == QUDA_DIRECT_PC_SOLVE ? QUDA_MATPC_SOLUTION : QUDA_MAT_SOLUTION;

    mg_param.omega[i] = 0.85; // over/under relaxation factor

    mg_param.location[i] = QUDA_CUDA_FIELD_LOCATION;
  }

  // only coarsen the spin on the first restriction
  mg_param.spin_block_size[0] = 2;

  // coarse grid solver is GCR
  mg_param.smoother[mg_levels-1] = QUDA_GCR_INVERTER;

  mg_param.compute_null_vector = generate_nullspace ? 
    QUDA_COMPUTE_NULL_VECTOR_YES : QUDA_COMPUTE_NULL_VECTOR_NO;
  mg_param.generate_all_levels = generate_all_levels ? 
    QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;

  mg_param.run_verify = QUDA_BOOLEAN_NO;

  // set file i/o parameters
  strcpy(mg_param.vec_infile, vec_infile);
  strcpy(mg_param.vec_outfile, vec_outfile);

  // these need to be set for now but are actually ignored by the MG setup
  // needed to make it pass the initialization test
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.tol = tol;
  inv_param.maxiter = niter;
  inv_param.reliable_delta = 1e-10;
  inv_param.gcrNkrylov = 10;
  //inv_param.max_res_increase = 4;

  inv_param.verbosity = QUDA_SUMMARIZE;
  inv_param.verbosity_precondition = QUDA_SUMMARIZE;
}

//=======================================================================//
//== C O N T A I N E R   A N D   Q U D A   I N I T I A L I S A T I O N  =//
//=======================================================================//

int main(int argc, char **argv)
{
  using namespace quda;

  for (int i = 1; i < argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    } 
    printfQuda("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

  // initialize QMP or MPI
#if defined(QMP_COMMS)
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);
#elif defined(MPI_COMMS)
  MPI_Init(&argc, &argv);
#endif

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  //QKXTM: DMH qkxtm specfic inputs
  qudaQKXTMinfo_Kepler info;  
  info.nsmearAPE = nsmearAPE;
  info.nsmearGauss = nsmearGauss;
  info.alphaAPE = alphaAPE;
  info.alphaGauss = alphaGauss;
  info.isEven = isEven;
  info.lL[0] = xdim;
  info.lL[1] = ydim;
  info.lL[2] = zdim;
  info.lL[3] = tdim;
  info.Nsources = numSourcePositions;
  info.Q_sq = Q_sq;
  info.traj = traj;

  if(strcmp(check_file_exist,"yes")==0 || 
     strcmp(check_file_exist,"YES")==0 )  info.check_files = true;
  else if(strcmp(check_file_exist,"no")==0 || 
	  strcmp(check_file_exist,"NO")==0 ) info.check_files = false;
  else errorQuda("Undefined input for option --check_corr_files");
 
  // Determine whether to write the correlation functions in ASCII or HDF5 
  // format
  if( strcmp(corr_file_format,"ASCII")==0 || 
      strcmp(corr_file_format,"ascii")==0 ) 
    info.CorrFileFormat = ASCII_FORM;   
  else if( strcmp(corr_file_format,"HDF5")==0 || 
	   strcmp(corr_file_format,"hdf5")==0 ) 
    info.CorrFileFormat = HDF5_FORM; 
  else fprintf(stderr,"Undefined option for --corr_file_format. Options are ASCII(ascii)/HDF5(hdf5)\n");
   
  //-C.K: Determine for which source-positions to run for the 3pt
  if(strcmp(run3pt,"all")==0 || 
     strcmp(run3pt,"ALL")==0) {
    for(int is = 0; is < numSourcePositions; is++) info.run3pt_src[is] = 1;
  }
  else if(strcmp(run3pt,"none")==0 || 
	  strcmp(run3pt,"NONE")==0) {
    for(int is = 0; is < numSourcePositions; is++) info.run3pt_src[is] = 0;
  }
  else if(strcmp(run3pt,"file")==0 || 
	  strcmp(run3pt,"FILE")==0) {
    printfQuda("Will read from file %s for which source-positions to perform the three-point function\n",pathListRun3pt);  
    FILE *ptr_run3pt;
    ptr_run3pt = fopen(pathListRun3pt,"r");
    if(ptr_run3pt == NULL) {
      fprintf(stderr,"Error opening file %s \n",pathListRun3pt);
      exit(-1);
    }
    for(int is = 0; is < numSourcePositions; is++) 
      fscanf(ptr_run3pt,"%d\n",&(info.run3pt_src[is]));
    fclose(ptr_run3pt);
  }
  else {
    printfQuda("Option --run3pt only accepts all/ALL and file/FILE parameters, or, if running for all source-positions, just disregard it.\n");
    exit(-1);
  }

  //-C.K: Get the list of source positions
  FILE *ptr_sources;
  ptr_sources = fopen(pathListSourcePositions,"r");
  if(ptr_sources == NULL){
    fprintf(stderr,"Error open file to read the source positions\n");
    exit(-1);
  }
  for(int is = 0 ; is < numSourcePositions ; is++)
    fscanf(ptr_sources,"%d %d %d %d",
	   &(info.sourcePosition[is][0]),
	   &(info.sourcePosition[is][1]), 
	   &(info.sourcePosition[is][2]), 
	   &(info.sourcePosition[is][3]));  
  fclose(ptr_sources);
  
  //-C.K: Read in the sink-source separations
  info.Ntsink = Ntsink;
  FILE *ptr_tsink;
  ptr_tsink = fopen(pathList_tsink,"r");
  if(ptr_tsink == NULL){
    fprintf(stderr,"Error opening file for sink-source separations\n");
    exit(-1);
  }
  for(int it = 0 ; it < Ntsink ; it++) {
    fscanf(ptr_tsink,"%d\n", &(info.tsinkSource[it]));
  }
  fclose(ptr_tsink);
  
  //-C.K: Determine for which projectors to run for the 3pt
  if(strcmp(proj_list_file,"default")==0){
    for(int i=0;i<Ntsink;i++){
      info.proj_list[i][0] = 0;   // Do only the G4 projector for all tsink's
      info.Nproj[i] = Nproj; // Nproj = 1 by default
    }
  }
  else{
    FILE *proj_ptr;
    char *proj_file;
    for(int it=0;it<Ntsink;it++){
      asprintf(&proj_file,"%s_tsink%d.txt",proj_list_file,
	       info.tsinkSource[it]);
      if( (proj_ptr = fopen(proj_file,"r")) == NULL ) {
	fprintf(stderr,"Cannot open projector file %s for reading.\n Hint: Make sure that 1: it ends as _tsink%d.txt and 2: the input passed is truncated up to this string.\n",proj_file,info.tsinkSource[it]);
	exit(-1);
      }
      
      fscanf(proj_ptr,"%d",&(info.Nproj[it]));
      for(int p=0;p<info.Nproj[it];p++) fscanf(proj_ptr,"%d\n",
					       &(info.proj_list[it][p]));
      
      fclose(proj_ptr);
    }
  }
  
  // Determine whether to write the correlation functions in position or 
  // momentum space
  if( strcmp(corr_write_space,"MOMENTUM")==0 || 
      strcmp(corr_write_space,"momentum")==0 ) 
    info.CorrSpace = MOMENTUM_SPACE;      
  else if( strcmp(corr_write_space,"POSITION")==0 || 
	   strcmp(corr_write_space,"position")==0 ) 
    info.CorrSpace = POSITION_SPACE; 
  else fprintf(stderr,"Undefined option for --corr_write_space. Options are MOMENTUM(momentum)/POSITION(position)\n");
  //--------------------------------------------------

  // QUDA parameters begin here.
  //-----------------------------------------------------------------

  if (dslash_type != QUDA_TWISTED_MASS_DSLASH && 
      dslash_type != QUDA_TWISTED_CLOVER_DSLASH && 
      dslash_type != QUDA_CLOVER_WILSON_DSLASH){
    printfQuda("This test is only for twisted mass or twisted clover operator\n");
    exit(-1);
  }

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);

  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  mg_param.invert_param = &mg_inv_param;
  setMultigridParam(mg_param);


  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);

  // declare the dimensions of the communication grid
  initCommsGridQuda(4, gridsize_from_cmdline, NULL, NULL);
  
  setDims(gauge_param.X);
  setSpinorSiteSize(24);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? 
    sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? 
    sizeof(double) : sizeof(float);
  
  void *gauge[4], *clover_inv=0, *clover=0;
  void *gauge_APE[4];
  void *gaugeContract[4];

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
    gauge_APE[dir] = malloc(V*gaugeSiteSize*gSize);
    gaugeContract[dir] = malloc(V*gaugeSiteSize*gSize);
    if( gauge[dir] == NULL || 
	gauge_APE[dir] == NULL || 
	gaugeContract[dir] == NULL ) 
      errorQuda("error allocate memory host gauge field\n"); 
  }
  
  // load in the command line supplied gauge field
  readLimeGauge(gauge, latfile, &gauge_param, &inv_param, 
		gridsize_from_cmdline);
  applyBoundaryCondition(gauge, V/2 ,&gauge_param);
  for(int mu = 0 ; mu < 4 ; mu++) {
    memcpy(gaugeContract[mu],gauge[mu],V*9*2*sizeof(double));
  }
  mapEvenOddToNormalGauge(gaugeContract,gauge_param,xdim,ydim,zdim,tdim);

  // load in the command line supplied smeared gauge field
  // first read gauge field without apply BC  
  readLimeGaugeSmeared(gauge_APE, latfile_smeared, &gauge_param, &inv_param,
		       gridsize_from_cmdline);
  mapEvenOddToNormalGauge(gauge_APE,gauge_param,xdim,ydim,zdim,tdim);
  
  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);
  //Print remaining info to stdout
  init_qudaQKXTM_Kepler(&info);
  printf_qudaQKXTM_Kepler();
  
  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  for(int i = 0 ; i < 4 ; i++){
    free(gauge[i]);
  }   

  printfQuda("Before clover term\n");
  // This line ensures that if we need to construct the clover inverse 
  // (in either the smoother or the solver) we do so
  if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || solve_type == QUDA_DIRECT_PC_SOLVE) {
    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
  }

  if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) 
    loadCloverQuda(NULL, NULL, &inv_param);
  printfQuda("After clover term\n");
  
  //QKXTM: DMH EXP
  // setup the multigrid solver for UP flavour
  mg_param.invert_param->twist_flavor = QUDA_TWIST_PLUS;
  void *mg_preconditionerUP = newMultigridQuda(&mg_param);
  inv_param.preconditionerUP = mg_preconditionerUP;

  // setup the multigrid solver for DN flavour
  mg_param.invert_param->twist_flavor = QUDA_TWIST_MINUS;
  void *mg_preconditionerDN = newMultigridQuda(&mg_param);
  inv_param.preconditionerDN = mg_preconditionerDN;  
  // reset twist flavour
  mg_param.invert_param->twist_flavor = twist_flavor;  


  calcMG_threepTwop_EvenOdd(gauge_APE, gaugeContract, &gauge_param,
			    &inv_param, info, twop_filename,
			    threep_filename, NEUTRON);

  // free the multigrid solvers
  destroyMultigridQuda(mg_preconditionerUP);
  destroyMultigridQuda(mg_preconditionerDN);
  
  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();
  
  for(int i = 0 ; i < 4 ; i++){
    free(gauge_APE[i]);
    free(gaugeContract[i]);
  }
  
  // finalize the QUDA library
  endQuda();
  
  // finalize the communications layer
#if defined(QMP_COMMS)
  QMP_finalize_msg_passing();
#elif defined(MPI_COMMS)
  MPI_Finalize();
#endif
  
  return 0;
}
