#include <dccrg.hpp>
#include <dccrg_cartesian_geometry.hpp>
#include "../grid.h"
#include "../spatial_cell.hpp"
#include "../definitions.h"
#include "../common.h"
#include "gridGlue.hpp"

/*
Calculate the number of cells on the maximum refinement level overlapping the list of dccrg cells in cells.
*/
int getNumberOfCellsOnMaxRefLvl(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid, const std::vector<CellID>& cells) {

   int nCells = 0;
   auto maxRefLvl = mpiGrid.mapping.get_maximum_refinement_level();
   
   for (auto cellid : cells) {
      auto refLvl = mpiGrid.get_refinement_level(cellid);
      nCells += pow(pow(2,maxRefLvl-refLvl),3);
   }

   return nCells;
   
}

/*
Map from dccrg cell id to fsgrid global cell ids when they aren't identical (ie. when dccrg has refinement).
*/

std::vector<CellID> mapDccrgIdToFsGridGlobalID(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
					       CellID dccrgID) {
   const auto maxRefLvl  = mpiGrid.get_maximum_refinement_level();
   const auto refLvl = mpiGrid.get_refinement_level(dccrgID);
   const auto cellLength = pow(2,maxRefLvl-refLvl);
   const auto topLeftIndices = mpiGrid.mapping.get_indices(dccrgID);
   std::array<int,3> fsgridDims;
   
   fsgridDims[0] = P::xcells_ini * pow(2,mpiGrid.get_maximum_refinement_level());
   fsgridDims[1] = P::ycells_ini * pow(2,mpiGrid.get_maximum_refinement_level());
   fsgridDims[2] = P::zcells_ini * pow(2,mpiGrid.get_maximum_refinement_level());

   std::vector<CellID> fsgridIDs(cellLength * cellLength * cellLength);
   for (uint k = 0; k < cellLength; ++k) {
      for (uint j = 0; j < cellLength; ++j) {
         for (uint i = 0; i < cellLength; ++i) {
	   const std::array<uint64_t,3> indices = {{topLeftIndices[0] + i,topLeftIndices[1] + j,topLeftIndices[2] + k}};
	   fsgridIDs[k*cellLength*cellLength + j*cellLength + i] = indices[0] + indices[1] * fsgridDims[0] + indices[2] * fsgridDims[1] * fsgridDims[0];
	 }
      }
   }
   return fsgridIDs;
}


/*Compute coupling DCCRG <=> FSGRID 

  onDccrgMapCells  maps fsgrid processes (key) => set of dccrg cellIDs owned by current rank that map to  the fsgrid cells owned by fsgrid process (val)
  onFsgridMapCells maps dccrg processes  (key) => set of dccrg cellIDs owned by dccrg-process that map to current rank fsgrid cells 
  onFsgridMapCells maps remote dccrg CellIDs to local fsgrid cells
*/

template <typename T, int stencil> void computeCoupling(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
							const std::vector<CellID>& cells,
							FsGrid< T, stencil>& momentsGrid,
							std::map<int, std::set<CellID> >& onDccrgMapRemoteProcess,
							std::map<int, std::set<CellID> >& onFsgridMapRemoteProcess,
							std::map<CellID, std::vector<int64_t> >& onFsgridMapCells
							) {
  
  
  //sorted list of dccrg cells. cells is typicall already sorted, but just to make sure....
  std::vector<CellID> dccrgCells = cells;
  std::sort(dccrgCells.begin(), dccrgCells.end());

  //make sure the datastructures are clean
  onDccrgMapRemoteProcess.clear();
  onFsgridMapRemoteProcess.clear();
  onFsgridMapCells.clear();
  
  
  //size of fsgrid local part
  const std::array<int, 3> gridDims(momentsGrid.getLocalSize());
  
 
  //Compute what we will receive, and where it should be stored
  for (int k=0; k<gridDims[2]; k++) {
    for (int j=0; j<gridDims[1]; j++) {
      for (int i=0; i<gridDims[0]; i++) {
	const std::array<int, 3> globalIndices = momentsGrid.getGlobalIndices(i,j,k);
	const dccrg::Types<3>::indices_t  indices = {{(uint64_t)globalIndices[0],
						      (uint64_t)globalIndices[1],
						      (uint64_t)globalIndices[2]}}; //cast to avoid warnings
	CellID dccrgCell = mpiGrid.get_existing_cell(indices, 0, mpiGrid.mapping.get_maximum_refinement_level());
	int process = mpiGrid.get_process(dccrgCell);
	int64_t  fsgridLid = momentsGrid.LocalIDForCoords(i,j,k);
	int64_t  fsgridGid = momentsGrid.GlobalIDForCoords(i,j,k);
	onFsgridMapRemoteProcess[process].insert(dccrgCell); //cells are ordered (sorted) in set
	onFsgridMapCells[dccrgCell].push_back(fsgridLid);
      }
    }
  }

  // Compute where to send data and what to send
  for(int i=0; i< dccrgCells.size(); i++) {
     //compute to which processes this cell maps
     std::vector<CellID> fsCells = mapDccrgIdToFsGridGlobalID(mpiGrid, dccrgCells[i]);

     //loop over fsgrid cells which this dccrg cell maps to
     for (auto const &fsCellID : fsCells) {
       int process = momentsGrid.getTaskForGlobalID(fsCellID).first; //process on fsgrid
       onDccrgMapRemoteProcess[process].insert(dccrgCells[i]); //add to map
     }    
  }
  

  //debug
  // int rank, nProcs;
  // int dRank=1;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
   
  // if(rank==dRank){
  //   for ( auto const &msg: onDccrgMapRemoteProcess)  {
  //     printf("SND %d => %d :\n", rank, msg.first);
  //     for ( auto const &id: msg.second)  {
  // 	printf(" %ld ", id);
  //     }
  //     printf("\n");
  //   }
  // }
  // MPI_Barrier(MPI_COMM_WORLD);
  // for(int r = 0; r < nProcs; r++){
  //   if(rank == r){
  //     for ( auto const &msg: onFsgridMapRemoteProcess)  {
  // 	if (msg.first == dRank) {
  // 	  printf("RCV %d => %d :\n", msg.first, rank);
  // 	  for ( auto const &id: msg.second)  {
  // 	    printf(" %ld ", id);
  // 	  }
  // 	  printf("\n");
  // 	}
  //     }
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

}

void feedMomentsIntoFsGrid(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                           const std::vector<CellID>& cells,
                           FsGrid< std::array<Real, fsgrids::moments::N_MOMENTS>, 2>& momentsGrid, bool dt2 /*=false*/) {


  int ii;
  //sorted list of dccrg cells. cells is typicall already sorted, but just to make sure....
  std::vector<CellID> dccrgCells = cells;
  std::sort(dccrgCells.begin(), dccrgCells.end());

  //Datastructure for coupling
  std::map<int, std::set<CellID> > onDccrgMapRemoteProcess; 
  std::map<int, std::set<CellID> > onFsgridMapRemoteProcess; 
  std::map<CellID, std::vector<int64_t> >  onFsgridMapCells;
    
  // map receive process => receive buffers 
  std::map<int, std::vector<Real> > receivedData; 

  // send buffers  to each process
  std::map<int, std::vector<Real> > sendData;

  //list of requests
  std::vector<MPI_Request> sendRequests;
  std::vector<MPI_Request> receiveRequests;


  //DEBUG IN
  // for(int i = 0;i < cells.size();i++){
  //   auto cellParams = mpiGrid[cells[i]]->get_cell_parameters();
  //   if(!dt2)
  //     printf("IN %ld: %g %g , %g %g %g \n", cells[i],cellParams[CellParams::RHOM],cellParams[CellParams::RHOQ],cellParams[CellParams::VX],cellParams[CellParams::VY], cellParams[CellParams::VZ]);
  //   else
  //     printf("IN %ld: %g %g , %g %g %g \n", cells[i],cellParams[CellParams::RHOM_DT2],cellParams[CellParams::RHOQ_DT2],cellParams[CellParams::VX_DT2],cellParams[CellParams::VY_DT2], cellParams[CellParams::VZ_DT2]);
  // }

  //computeCoupling
  computeCoupling(mpiGrid, cells, momentsGrid, onDccrgMapRemoteProcess, onFsgridMapRemoteProcess, onFsgridMapCells);
 
  // Post receives
  receiveRequests.resize(onFsgridMapRemoteProcess.size());  
  ii=0;
  for(auto const &receives: onFsgridMapRemoteProcess){
    int process = receives.first;
    int count = receives.second.size();
    receivedData[process].resize(count * fsgrids::moments::N_MOMENTS);
    MPI_Irecv(receivedData[process].data(), count * fsgrids::moments::N_MOMENTS * sizeof(Real),
	      MPI_BYTE, process, 1, MPI_COMM_WORLD,&(receiveRequests[ii++]));
  }
  
  // Launch sends
  ii=0;
  sendRequests.resize(onDccrgMapRemoteProcess.size());
  for (auto const &snd : onDccrgMapRemoteProcess){
    int targetProc = snd.first; 
    auto& sendBuffer=sendData[targetProc];
    for(CellID sendCell: snd.second){
      //Collect data to send for this dccrg cell
      auto cellParams = mpiGrid[sendCell]->get_cell_parameters();
      if(!dt2) {
        sendBuffer.push_back(cellParams[CellParams::RHOM]);
	sendBuffer.push_back(cellParams[CellParams::RHOQ]);
	sendBuffer.push_back(cellParams[CellParams::VX]);
	sendBuffer.push_back(cellParams[CellParams::VY]);
        sendBuffer.push_back(cellParams[CellParams::VZ]);
	sendBuffer.push_back(cellParams[CellParams::P_11]);
	sendBuffer.push_back(cellParams[CellParams::P_22]);
        sendBuffer.push_back(cellParams[CellParams::P_33]);
      } else {
        sendBuffer.push_back(cellParams[CellParams::RHOM_DT2]);
	sendBuffer.push_back(cellParams[CellParams::RHOQ_DT2]);
	sendBuffer.push_back(cellParams[CellParams::VX_DT2]);
	sendBuffer.push_back(cellParams[CellParams::VY_DT2]);
        sendBuffer.push_back(cellParams[CellParams::VZ_DT2]);
	sendBuffer.push_back(cellParams[CellParams::P_11_DT2]);
	sendBuffer.push_back(cellParams[CellParams::P_22_DT2]);
        sendBuffer.push_back(cellParams[CellParams::P_33_DT2]);
      }
    }
    int count = sendBuffer.size(); //note, compared to receive this includes all elements to be sent
    MPI_Isend(sendBuffer.data(), sendBuffer.size() * sizeof(Real),
	      MPI_BYTE, targetProc, 1, MPI_COMM_WORLD,&(sendRequests[ii]));
    ii++;
  }

  MPI_Waitall(receiveRequests.size(), receiveRequests.data(), MPI_STATUSES_IGNORE);
  for(auto const &receives: onFsgridMapRemoteProcess){
    int process = receives.first; //data received from this process
    Real* receiveBuffer = receivedData[process].data(); // data received from process
    for(auto const &cell: receives.second){ //loop over cellids (dccrg) for receive
      // this part heavily relies on both sender and receiver having cellids sorted!
      for(auto lid: onFsgridMapCells[cell]){
	std::array<Real, fsgrids::moments::N_MOMENTS> * fsgridData = momentsGrid.get(lid);
	for(int l = 0; l < fsgrids::moments::N_MOMENTS; l++)   {
	  fsgridData->at(l) = receiveBuffer[l];
	}
      }
      receiveBuffer+=fsgrids::moments::N_MOMENTS;
    }
  }

  MPI_Waitall(sendRequests.size(), sendRequests.data(), MPI_STATUSES_IGNORE);

  //DEBUG OUT
  //size of fsgrid local part
  //  const std::array<int, 3> gridDims(momentsGrid.getLocalSize());
  //Compute what we will receive, and where it should be stored
  // for (int k=0; k<gridDims[2]; k++) {
  //   for (int j=0; j<gridDims[1]; j++) {
  //     for (int i=0; i<gridDims[0]; i++) {
  // 	int64_t  fsgridGid = momentsGrid.GlobalIDForCoords(i,j,k);
  // 	int64_t  fsgridLid = momentsGrid.LocalIDForCoords(i,j,k);
  // 	std::array<Real, fsgrids::moments::N_MOMENTS> * fsgridData = momentsGrid.get(fsgridLid);
  // 	printf("OUT %ld (+1): %g %g , %g %g %g \n", fsgridGid + 1,
  // 	       fsgridData->at(0), fsgridData->at(1), fsgridData->at(2), fsgridData->at(3),fsgridData->at(4));	
  //     }
  //   }
  // }  
  // MPI_Barrier(MPI_COMM_WORLD);

}


void feedBgFieldsIntoFsGrid(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
    const std::vector<CellID>& cells, FsGrid< std::array<Real, fsgrids::bgbfield::N_BGB>, 2>& bgBGrid) {

   int nCells = getNumberOfCellsOnMaxRefLvl(mpiGrid, cells);
   bgBGrid.setupForTransferIn(nCells);

   // Setup transfer buffers
   std::vector< std::array<Real, fsgrids::bgbfield::N_BGB> > transferBuffer(cells.size());
   
   // Fill from cellParams
   // We only need to read data once per dccrg cell here
#pragma omp parallel for
   for(uint i = 0; i < cells.size(); ++i) {
      CellID dccrgId = cells[i];
      auto cellParams = mpiGrid[dccrgId]->get_cell_parameters();
      auto derivatives = mpiGrid[dccrgId]->derivatives;
      auto volumeDerivatives = mpiGrid[dccrgId]->derivativesBVOL;

      //      std::cout << "I am at line " << __LINE__ << " of " << __FILE__ << std::endl;
               
      std::array<Real, fsgrids::bgbfield::N_BGB>* thisCellData = &transferBuffer[i];
      
      thisCellData->at(fsgrids::bgbfield::BGBX) = cellParams[CellParams::BGBX];
      thisCellData->at(fsgrids::bgbfield::BGBY) = cellParams[CellParams::BGBY];
      thisCellData->at(fsgrids::bgbfield::BGBZ) = cellParams[CellParams::BGBZ];
      thisCellData->at(fsgrids::bgbfield::BGBXVOL) = cellParams[CellParams::BGBXVOL];
      thisCellData->at(fsgrids::bgbfield::BGBYVOL) = cellParams[CellParams::BGBYVOL];
      thisCellData->at(fsgrids::bgbfield::BGBZVOL) = cellParams[CellParams::BGBZVOL];
      
      thisCellData->at(fsgrids::bgbfield::dBGBxdy) = derivatives[fieldsolver::dBGBxdy];
      thisCellData->at(fsgrids::bgbfield::dBGBxdz) = derivatives[fieldsolver::dBGBxdz];
      thisCellData->at(fsgrids::bgbfield::dBGBydx) = derivatives[fieldsolver::dBGBydx];
      thisCellData->at(fsgrids::bgbfield::dBGBydz) = derivatives[fieldsolver::dBGBydz];
      thisCellData->at(fsgrids::bgbfield::dBGBzdx) = derivatives[fieldsolver::dBGBzdx];
      thisCellData->at(fsgrids::bgbfield::dBGBzdy) = derivatives[fieldsolver::dBGBzdy];
      
      thisCellData->at(fsgrids::bgbfield::dBGBXVOLdy) = volumeDerivatives[bvolderivatives::dBGBXVOLdy];
      thisCellData->at(fsgrids::bgbfield::dBGBXVOLdz) = volumeDerivatives[bvolderivatives::dBGBXVOLdz];
      thisCellData->at(fsgrids::bgbfield::dBGBYVOLdx) = volumeDerivatives[bvolderivatives::dBGBYVOLdx];
      thisCellData->at(fsgrids::bgbfield::dBGBYVOLdz) = volumeDerivatives[bvolderivatives::dBGBYVOLdz];
      thisCellData->at(fsgrids::bgbfield::dBGBZVOLdx) = volumeDerivatives[bvolderivatives::dBGBZVOLdx];
      thisCellData->at(fsgrids::bgbfield::dBGBZVOLdy) = volumeDerivatives[bvolderivatives::dBGBZVOLdy];
   }

   // Copy data into each fsgrid cell overlapping the dccrg cell
   for (uint i = 0; i < cells.size(); ++i) {
      CellID dccrgId = cells[i];
      const auto fsgridIds = mapDccrgIdToFsGridGlobalID(mpiGrid, dccrgId);
      for (auto fsgridId : fsgridIds) {
         bgBGrid.transferDataIn(fsgridId, &transferBuffer[i]);
      }
   }
   
   // Finish the actual transfer
   bgBGrid.finishTransfersIn();
}

void getVolumeFieldsFromFsGrid(FsGrid< std::array<Real, fsgrids::volfields::N_VOL>, 2>& volumeFieldsGrid,
                           dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                           const std::vector<CellID>& cells) {


   // Setup transfer buffers
   int nCells = getNumberOfCellsOnMaxRefLvl(mpiGrid, cells);
   std::vector< std::array<Real, fsgrids::volfields::N_VOL> > transferBuffer(nCells);
   std::vector< std::array<Real, fsgrids::volfields::N_VOL>*> transferBufferPointer;

   // Setup transfer pointers
   volumeFieldsGrid.setupForTransferOut(nCells);
   int k = 0;
   for(auto dccrgId : cells) {
      const auto fsgridIds = mapDccrgIdToFsGridGlobalID(mpiGrid, dccrgId);
      // Store a pointer to the first fsgrid cell that maps to each dccrg Id
      transferBufferPointer.push_back(&transferBuffer[k]);
      for (auto fsgridId : fsgridIds) {
         std::array<Real, fsgrids::volfields::N_VOL>* thisCellData = &transferBuffer[k++];
         volumeFieldsGrid.transferDataOut(fsgridId, thisCellData);
      }
   }
   // Do the transfer
   volumeFieldsGrid.finishTransfersOut();

   // Build a list of index pairs to cellparams and fsgrid
   std::vector<std::pair<int,int>> iCellParams;
   iCellParams.reserve(6);
   iCellParams.push_back(std::make_pair(CellParams::PERBXVOL, fsgrids::volfields::PERBXVOL));
   iCellParams.push_back(std::make_pair(CellParams::PERBYVOL, fsgrids::volfields::PERBYVOL));
   iCellParams.push_back(std::make_pair(CellParams::PERBZVOL, fsgrids::volfields::PERBZVOL));
   iCellParams.push_back(std::make_pair(CellParams::EXVOL,    fsgrids::volfields::EXVOL));
   iCellParams.push_back(std::make_pair(CellParams::EYVOL,    fsgrids::volfields::EYVOL));
   iCellParams.push_back(std::make_pair(CellParams::EZVOL,    fsgrids::volfields::EZVOL));

   // Build lists of index pairs to dccrg and fsgrid
   std::vector<std::pair<int,int>> iDerivativesBVOL;
   iDerivativesBVOL.reserve(6);
   iDerivativesBVOL.push_back(std::make_pair(bvolderivatives::dPERBXVOLdy, fsgrids::volfields::dPERBXVOLdy));
   iDerivativesBVOL.push_back(std::make_pair(bvolderivatives::dPERBXVOLdz, fsgrids::volfields::dPERBXVOLdz));
   iDerivativesBVOL.push_back(std::make_pair(bvolderivatives::dPERBYVOLdx, fsgrids::volfields::dPERBYVOLdx));
   iDerivativesBVOL.push_back(std::make_pair(bvolderivatives::dPERBYVOLdz, fsgrids::volfields::dPERBYVOLdz));
   iDerivativesBVOL.push_back(std::make_pair(bvolderivatives::dPERBZVOLdx, fsgrids::volfields::dPERBZVOLdx));
   iDerivativesBVOL.push_back(std::make_pair(bvolderivatives::dPERBZVOLdy, fsgrids::volfields::dPERBZVOLdy));
   
   // Distribute data from the transfer buffer back into the appropriate mpiGrid places
   #pragma omp parallel for

   for(int i=0; i< cells.size(); i++) {
     std::array<Real, fsgrids::volfields::N_VOL>* thisCellData = &transferBuffer[i];
      auto cellParams = mpiGrid[cells[i]]->get_cell_parameters();

      cellParams[CellParams::PERBXVOL]                          = thisCellData->at(fsgrids::volfields::PERBXVOL);
      cellParams[CellParams::PERBYVOL]                          = thisCellData->at(fsgrids::volfields::PERBYVOL);
      cellParams[CellParams::PERBZVOL]                          = thisCellData->at(fsgrids::volfields::PERBZVOL);
      cellParams[CellParams::EXVOL]                             = thisCellData->at(fsgrids::volfields::EXVOL);
      cellParams[CellParams::EYVOL]                             = thisCellData->at(fsgrids::volfields::EYVOL);
      cellParams[CellParams::EZVOL]                             = thisCellData->at(fsgrids::volfields::EZVOL);
      mpiGrid[cells[i]]->derivativesBVOL[bvolderivatives::dPERBXVOLdy] = thisCellData->at(fsgrids::volfields::dPERBXVOLdy);
      mpiGrid[cells[i]]->derivativesBVOL[bvolderivatives::dPERBXVOLdz] = thisCellData->at(fsgrids::volfields::dPERBXVOLdz);
      mpiGrid[cells[i]]->derivativesBVOL[bvolderivatives::dPERBYVOLdx] = thisCellData->at(fsgrids::volfields::dPERBYVOLdx);
      mpiGrid[cells[i]]->derivativesBVOL[bvolderivatives::dPERBYVOLdz] = thisCellData->at(fsgrids::volfields::dPERBYVOLdz);
      mpiGrid[cells[i]]->derivativesBVOL[bvolderivatives::dPERBZVOLdx] = thisCellData->at(fsgrids::volfields::dPERBZVOLdx);
      mpiGrid[cells[i]]->derivativesBVOL[bvolderivatives::dPERBZVOLdy] = thisCellData->at(fsgrids::volfields::dPERBZVOLdy);
   }
}

void getDerivativesFromFsGrid(FsGrid< std::array<Real, fsgrids::dperb::N_DPERB>, 2>& dperbGrid,
                          FsGrid< std::array<Real, fsgrids::dmoments::N_DMOMENTS>, 2>& dmomentsGrid,
                          FsGrid< std::array<Real, fsgrids::bgbfield::N_BGB>, 2>& bgbfieldGrid,
                          dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                          const std::vector<CellID>& cells) {

   // Setup transfer buffers
   int nCellsOnMaxRefLvl = getNumberOfCellsOnMaxRefLvl(mpiGrid, cells);
   std::vector< std::array<Real, fsgrids::dperb::N_DPERB> > dperbTransferBuffer(nCellsOnMaxRefLvl);
   std::vector< std::array<Real, fsgrids::dmoments::N_DMOMENTS> > dmomentsTransferBuffer(nCellsOnMaxRefLvl);
   std::vector< std::array<Real, fsgrids::bgbfield::N_BGB> > bgbfieldTransferBuffer(nCellsOnMaxRefLvl);
   
   std::vector< std::array<Real, fsgrids::dperb::N_DPERB>*> dperbTransferBufferPointer;
   std::vector< std::array<Real, fsgrids::dmoments::N_DMOMENTS>*> dmomentsTransferBufferPointer;
   std::vector< std::array<Real, fsgrids::bgbfield::N_BGB>*> bgbfieldTransferBufferPointer;
   
   dperbGrid.setupForTransferOut(nCellsOnMaxRefLvl);
   dmomentsGrid.setupForTransferOut(nCellsOnMaxRefLvl);
   bgbfieldGrid.setupForTransferOut(nCellsOnMaxRefLvl);
   
   int k = 0;
   for (auto dccrgId : cells) {

      // Assuming same local size in all fsgrids
      const auto fsgridIds = mapDccrgIdToFsGridGlobalID(mpiGrid, dccrgId);
      // Store a pointer to the first fsgrid cell that maps to each dccrg Id
      dperbTransferBufferPointer.push_back(&dperbTransferBuffer[k]);
      dmomentsTransferBufferPointer.push_back(&dmomentsTransferBuffer[k]);
      bgbfieldTransferBufferPointer.push_back(&bgbfieldTransferBuffer[k]);

      for (auto fsgridId : fsgridIds) {
      
         std::array<Real, fsgrids::dperb::N_DPERB>* dperbCellData = &dperbTransferBuffer[k];
         dperbGrid.transferDataOut(fsgridId, dperbCellData);
         std::array<Real, fsgrids::dmoments::N_DMOMENTS>* dmomentsCellData = &dmomentsTransferBuffer[k];
         dmomentsGrid.transferDataOut(fsgridId, dmomentsCellData);
         std::array<Real, fsgrids::bgbfield::N_BGB>* bgbfieldCellData = &bgbfieldTransferBuffer[k++];
         bgbfieldGrid.transferDataOut(fsgridId, bgbfieldCellData);
      }
   }
   
   // Do the transfer
   dperbGrid.finishTransfersOut();
   dmomentsGrid.finishTransfersOut();
   bgbfieldGrid.finishTransfersOut();

   std::vector<std::pair<int,int>> iDmoments;
   std::vector<std::pair<int,int>> iDperb;
   std::vector<std::pair<int,int>> iBgbfield;
   iDmoments.reserve(24);
   iDmoments.push_back(std::make_pair(fieldsolver::drhomdx, fsgrids::dmoments::drhomdx));
   iDmoments.push_back(std::make_pair(fieldsolver::drhomdy, fsgrids::dmoments::drhomdy));
   iDmoments.push_back(std::make_pair(fieldsolver::drhomdz, fsgrids::dmoments::drhomdz));
   iDmoments.push_back(std::make_pair(fieldsolver::drhoqdx, fsgrids::dmoments::drhoqdx));
   iDmoments.push_back(std::make_pair(fieldsolver::drhoqdy, fsgrids::dmoments::drhoqdy));
   iDmoments.push_back(std::make_pair(fieldsolver::drhoqdz, fsgrids::dmoments::drhoqdz));
   iDmoments.push_back(std::make_pair(fieldsolver::dp11dx , fsgrids::dmoments::dp11dx ));
   iDmoments.push_back(std::make_pair(fieldsolver::dp11dy , fsgrids::dmoments::dp11dy ));
   iDmoments.push_back(std::make_pair(fieldsolver::dp11dz , fsgrids::dmoments::dp11dz ));
   iDmoments.push_back(std::make_pair(fieldsolver::dp22dx , fsgrids::dmoments::dp22dx ));
   iDmoments.push_back(std::make_pair(fieldsolver::dp22dy , fsgrids::dmoments::dp22dy ));
   iDmoments.push_back(std::make_pair(fieldsolver::dp22dz , fsgrids::dmoments::dp22dz ));
   iDmoments.push_back(std::make_pair(fieldsolver::dp33dx , fsgrids::dmoments::dp33dx ));
   iDmoments.push_back(std::make_pair(fieldsolver::dp33dy , fsgrids::dmoments::dp33dy ));
   iDmoments.push_back(std::make_pair(fieldsolver::dp33dz , fsgrids::dmoments::dp33dz ));
   iDmoments.push_back(std::make_pair(fieldsolver::dVxdx  , fsgrids::dmoments::dVxdx  ));
   iDmoments.push_back(std::make_pair(fieldsolver::dVxdy  , fsgrids::dmoments::dVxdy  ));
   iDmoments.push_back(std::make_pair(fieldsolver::dVxdz  , fsgrids::dmoments::dVxdz  ));
   iDmoments.push_back(std::make_pair(fieldsolver::dVydx  , fsgrids::dmoments::dVydx  ));
   iDmoments.push_back(std::make_pair(fieldsolver::dVydy  , fsgrids::dmoments::dVydy  ));
   iDmoments.push_back(std::make_pair(fieldsolver::dVydz  , fsgrids::dmoments::dVydz  ));
   iDmoments.push_back(std::make_pair(fieldsolver::dVzdx  , fsgrids::dmoments::dVzdx  ));
   iDmoments.push_back(std::make_pair(fieldsolver::dVzdy  , fsgrids::dmoments::dVzdy  ));
   iDmoments.push_back(std::make_pair(fieldsolver::dVzdz  , fsgrids::dmoments::dVzdz  ));

   iDperb.reserve(15);
   iDperb.push_back(std::make_pair(fieldsolver::dPERBxdy , fsgrids::dperb::dPERBxdy ));
   iDperb.push_back(std::make_pair(fieldsolver::dPERBxdz , fsgrids::dperb::dPERBxdz ));
   iDperb.push_back(std::make_pair(fieldsolver::dPERBydx , fsgrids::dperb::dPERBydx ));
   iDperb.push_back(std::make_pair(fieldsolver::dPERBydz , fsgrids::dperb::dPERBydz ));
   iDperb.push_back(std::make_pair(fieldsolver::dPERBzdx , fsgrids::dperb::dPERBzdx ));
   iDperb.push_back(std::make_pair(fieldsolver::dPERBzdy , fsgrids::dperb::dPERBzdy ));
   iDperb.push_back(std::make_pair(fieldsolver::dPERBxdyy, fsgrids::dperb::dPERBxdyy));
   iDperb.push_back(std::make_pair(fieldsolver::dPERBxdzz, fsgrids::dperb::dPERBxdzz));
   iDperb.push_back(std::make_pair(fieldsolver::dPERBydxx, fsgrids::dperb::dPERBydxx));
   iDperb.push_back(std::make_pair(fieldsolver::dPERBydzz, fsgrids::dperb::dPERBydzz));
   iDperb.push_back(std::make_pair(fieldsolver::dPERBzdxx, fsgrids::dperb::dPERBzdxx));
   iDperb.push_back(std::make_pair(fieldsolver::dPERBzdyy, fsgrids::dperb::dPERBzdyy));
   iDperb.push_back(std::make_pair(fieldsolver::dPERBxdyz, fsgrids::dperb::dPERBxdyz));
   iDperb.push_back(std::make_pair(fieldsolver::dPERBydxz, fsgrids::dperb::dPERBydxz));
   iDperb.push_back(std::make_pair(fieldsolver::dPERBzdxy, fsgrids::dperb::dPERBzdxy));

   iBgbfield.reserve(6);
   iBgbfield.push_back(std::make_pair(fieldsolver::dBGBxdy, fsgrids::bgbfield::dBGBxdy));
   iBgbfield.push_back(std::make_pair(fieldsolver::dBGBxdz, fsgrids::bgbfield::dBGBxdz));
   iBgbfield.push_back(std::make_pair(fieldsolver::dBGBydx, fsgrids::bgbfield::dBGBydx));
   iBgbfield.push_back(std::make_pair(fieldsolver::dBGBydz, fsgrids::bgbfield::dBGBydz));
   iBgbfield.push_back(std::make_pair(fieldsolver::dBGBzdx, fsgrids::bgbfield::dBGBzdx));
   iBgbfield.push_back(std::make_pair(fieldsolver::dBGBzdy, fsgrids::bgbfield::dBGBzdy));
   
   // Distribute data from the transfer buffers back into the appropriate mpiGrid places
   #pragma omp parallel for
   for(uint i = 0; i < cells.size(); ++i) {

      int dccrgId = cells[i];

      // Calculate the number of fsgrid cells we need to average into the current dccrg cell
      auto refLvl = mpiGrid.mapping.get_refinement_level(dccrgId);
      int nCells = pow(pow(2,mpiGrid.mapping.get_maximum_refinement_level() - mpiGrid.mapping.get_refinement_level(dccrgId)),3);

      for (auto j : iDmoments) mpiGrid[dccrgId]->derivatives[j.first] = 0.0;
      for (auto j : iDperb   ) mpiGrid[dccrgId]->derivatives[j.first] = 0.0;
      for (auto j : iBgbfield) mpiGrid[dccrgId]->derivatives[j.first] = 0.0;
      
      for(int iCell = 0; iCell < nCells; ++iCell) {
         // The fsgrid cells that cover the i'th dccrg cell are pointed at by
         // transferBufferPointer[i] ... transferBufferPointer[i] + nCell. We want to average
         // over all of them to get the value for the dccrg cell
         
         std::array<Real, fsgrids::dperb::N_DPERB>* dperb    = dperbTransferBufferPointer[i] + iCell;
         std::array<Real, fsgrids::dmoments::N_DMOMENTS>* dmoments = dmomentsTransferBufferPointer[i] + iCell;
         std::array<Real, fsgrids::bgbfield::N_BGB>* bgbfield = bgbfieldTransferBufferPointer[i] + iCell;
      
         for (auto j : iDmoments) mpiGrid[dccrgId]->derivatives[j.first] += dmoments->at(j.second);
         for (auto j : iDperb   ) mpiGrid[dccrgId]->derivatives[j.first] += dperb   ->at(j.second);
         for (auto j : iBgbfield) mpiGrid[dccrgId]->derivatives[j.first] += bgbfield->at(j.second);
      }

      for (auto j : iDmoments) mpiGrid[dccrgId]->derivatives[j.first] /= nCells;
      for (auto j : iDperb   ) mpiGrid[dccrgId]->derivatives[j.first] /= nCells;
      for (auto j : iBgbfield) mpiGrid[dccrgId]->derivatives[j.first] /= nCells;

   }
}

bool belongsToLayer(const int layer, const int x, const int y, const int z,
                    FsGrid< fsgrids::technical, 2>& technicalGrid) {

   bool belongs = false;
   
   // loop through all neighbors (including diagonals)
   for (int ix = -1; ix <= 1; ++ix) {
      for (int iy = -1; iy <= 1; ++iy) {
         for (int iz = -1; iz <= 1; ++iz) {
            
            // not strictly necessary but logically we should not consider the cell itself
            // among its neighbors.
            if( ix == 0 && iy == 0 && iz == 0 || !technicalGrid.get(x+ix,y+iy,z+iz)) {
               continue;
            }
            
            if(layer == 1 && technicalGrid.get(x+ix,y+iy,z+iz)->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY) {
               // in the first layer, boundary cell belongs if it has a non-boundary neighbor
               belongs = true;
               return belongs;
               
            } else if (layer > 1 && technicalGrid.get(x+ix,y+iy,z+iz)->sysBoundaryLayer == layer - 1) {
               // in all other layers, boundary cell belongs if it has a neighbor in the previous layer
               belongs = true;
               return belongs;
            }
         }
      }
   }

   return belongs;
}

void setupTechnicalFsGrid(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const std::vector<CellID>& cells, FsGrid< fsgrids::technical, 2>& technicalGrid) {

   int nCells = getNumberOfCellsOnMaxRefLvl(mpiGrid, cells);   
   technicalGrid.setupForTransferIn(nCells);

   // Setup transfer buffers
   std::vector< fsgrids::technical > transferBuffer(cells.size());
   
#pragma omp parallel for
   for(uint i = 0; i < cells.size(); ++i) {

      fsgrids::technical* thisCellData = &transferBuffer[i];
      // Data needs to be collected from some different places for this grid.
      thisCellData->sysBoundaryFlag = mpiGrid[cells[i]]->sysBoundaryFlag;
      // Remove boundary layer copy here
      // thisCellData->sysBoundaryLayer = mpiGrid[cells[i]]->sysBoundaryLayer;
      thisCellData->maxFsDt = std::numeric_limits<Real>::max();        
   }

   for(uint i = 0; i < cells.size(); ++i) {
      
      const auto fsgridIds = mapDccrgIdToFsGridGlobalID(mpiGrid, cells[i]);
      
      for (auto fsgridId : fsgridIds) {
         // std::cout << "fsgridId: " << fsgridId << ", fsgrid Cell Coordinates:";
         // auto coords = technicalGrid.globalIDtoCellCoord(fsgridId);
         // for (auto coord : coords) std::cout << " " << coord;
         // std::cout << std::endl;
         technicalGrid.transferDataIn(fsgridId,&transferBuffer[i]);
      }
   }

   technicalGrid.finishTransfersIn();

   auto localSize = technicalGrid.getLocalSize();
   
   // Add layer calculation here. Include diagonals +-1.

   // Initialize boundary layer flags to 0.
   for (int x = 0; x < localSize[0]; ++x) {
      for (int y = 0; y < localSize[1]; ++y) {
         for (int z = 0; z < localSize[2]; ++z) {
            technicalGrid.get(x,y,z)->sysBoundaryLayer = 0;
         }
      }
   }   

   // In dccrg initialization the max number of boundary layers is set to 3.
   const int MAX_NUMBER_OF_BOUNDARY_LAYERS = 3 * (mpiGrid.get_maximum_refinement_level() + 1);

   // loop through max number of layers
   for(uint layer = 1; layer <= MAX_NUMBER_OF_BOUNDARY_LAYERS; ++layer) {
      
      // loop through all cells in grid
      for (int x = 0; x < localSize[0]; ++x) {
         for (int y = 0; y < localSize[1]; ++y) {
            for (int z = 0; z < localSize[2]; ++z) {
               
               // for the first layer, consider all cells that belong to a boundary, for other layers
               // consider all cells that have not yet been labeled.
               if((layer == 1 && technicalGrid.get(x,y,z)->sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY) ||
                  (layer > 1 && technicalGrid.get(x,y,z)->sysBoundaryLayer == 0)) {
                  
                  if (belongsToLayer(layer, x, y, z, technicalGrid)) {
                     
                     technicalGrid.get(x,y,z)->sysBoundaryLayer = layer;
                     
                     if (layer > 1) {
                        technicalGrid.get(x,y,z)->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE;
                     }
                  }
               }
            }
         }
      }
   }

   // for (int x = 0; x < localSize[0]; ++x) {
   //    for (int y = 0; y < localSize[1]; ++y) {
   //       for (int z = 0; z < localSize[2]; ++z) {
   //          std::cout << "boundary layer+flag at " << x << ", " << y << ", " << z << " = ";
   //          std::cout << technicalGrid.get(x,y,z)->sysBoundaryLayer;
   //          std::cout << " ";
   //          std::cout << technicalGrid.get(x,y,z)->sysBoundaryFlag;
   //       }
   //    }
   // }     
   //abort();
}

void getFsGridMaxDt(FsGrid< fsgrids::technical, 2>& technicalGrid,
      dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const std::vector<CellID>& cells) {

   int nCells = getNumberOfCellsOnMaxRefLvl(mpiGrid, cells);   
   technicalGrid.setupForTransferOut(nCells);

   // Buffer to store contents of the grid
   std::vector<fsgrids::technical> transferBuffer(nCells);
   std::vector<fsgrids::technical*> transferBufferPointer;

   int k = 0;
   for(int i=0; i< cells.size(); i++) {

      transferBufferPointer.push_back(&transferBuffer[k]);
      const auto fsgridIds = mapDccrgIdToFsGridGlobalID(mpiGrid, cells[i]);
      for (auto fsgridId : fsgridIds) {
         fsgrids::technical* thisCellData = &transferBuffer[k++];
         technicalGrid.transferDataOut(fsgridId, thisCellData);
      }
   }

   technicalGrid.finishTransfersOut();

   // After the transfer is completed, stuff the recieved maxFDt into the cells.
   #pragma omp parallel for
   for(int i=0; i< cells.size(); i++) {
      
      int dccrgId = cells[i];
      auto cellParams = mpiGrid[dccrgId]->get_cell_parameters();
      
      // Calculate the number of fsgrid cells we need to average into the current dccrg cell
      int nCells = pow(pow(2,mpiGrid.get_maximum_refinement_level() - mpiGrid.get_refinement_level(dccrgId)),3);

      cellParams[CellParams::MAXFDT] = std::numeric_limits<Real>::max();
      //cellParams[CellParams::FSGRID_RANK] = 0;
      //cellParams[CellParams::FSGRID_BOUNDARYTYPE] = 0;

      for (int iCell = 0; iCell < nCells; ++iCell) {
                 
         fsgrids::technical* thisCellData = transferBufferPointer[i] + iCell;

         if (thisCellData->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY || thisCellData->sysBoundaryLayer == 1) {
         
            cellParams[CellParams::MAXFDT] = std::min(cellParams[CellParams::MAXFDT],thisCellData->maxFsDt);
            
         }
         
         //TODO: Implement something for FSGRID_RANK and FSGRID_BOUNDARYTYPE
         //cellParams[CellParams::FSGRID_RANK] = thisCellData->fsGridRank;
         //cellParams[CellParams::FSGRID_BOUNDARYTYPE] = thisCellData->sysBoundaryFlag;
      }
   }
}

/*
Map from dccrg cell id to fsgrid global cell ids when they aren't identical (ie. when dccrg has refinement).
*/

std::vector<CellID> mapDccrgIdToFsGridGlobalID(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
					       CellID dccrgID) {

   const auto maxRefLvl  = mpiGrid.get_maximum_refinement_level();
   const auto refLvl = mpiGrid.get_refinement_level(dccrgID);
   const auto cellLength = pow(2,maxRefLvl-refLvl);
   const auto topLeftIndices = mpiGrid.mapping.get_indices(dccrgID);
   std::array<int,3> indices;
   std::vector<std::array<int,3>> allIndices;

   std::array<int,3> fsgridDims;
   fsgridDims[0] = P::xcells_ini * pow(2,mpiGrid.get_maximum_refinement_level());
   fsgridDims[1] = P::ycells_ini * pow(2,mpiGrid.get_maximum_refinement_level());
   fsgridDims[2] = P::zcells_ini * pow(2,mpiGrid.get_maximum_refinement_level());
   
   for (uint k = 0; k < cellLength; ++k) {
      for (uint j = 0; j < cellLength; ++j) {
         for (uint i = 0; i < cellLength; ++i) {
            indices[0] = topLeftIndices[0] + i;
            indices[1] = topLeftIndices[1] + j;
            indices[2] = topLeftIndices[2] + k;
            allIndices.push_back(indices);
         }
      }
   }

   std::vector<CellID> fsgridIDs;  


   for (auto cellCoord: allIndices) {
     
     fsgridIDs.push_back(cellCoord[0] 
			 + cellCoord[1] * fsgridDims[0] 
			 + cellCoord[2] * fsgridDims[1] * fsgridDims[0]);

   }

   return fsgridIDs;
}

