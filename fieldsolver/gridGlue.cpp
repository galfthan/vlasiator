#include <dccrg.hpp>
#include <dccrg_cartesian_geometry.hpp>
#include "../grid.h"
#include "../spatial_cell.hpp"
#include "../definitions.h"
#include "../common.h"
#include "gridGlue.hpp"


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


void feedMomentsIntoFsGrid(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                           const std::vector<CellID>& cells,
                           FsGrid< std::array<Real, fsgrids::moments::N_MOMENTS>, 2>& momentsGrid, bool dt2 /*=false*/) {

  int ii;
  //sorted list of dccrg cells. cells is typicall already sorted, but just to make sure....
  std::vector<CellID> dccrgCells = cells;
  std::sort(dccrgCells.begin(), dccrgCells.end());

  //size of fsgrid local part
  const std::array<int, 3> gridDims(momentsGrid.getLocalSize());

  // map receive process => dccrg cells we will receive from it
  std::map<int, std::set<CellID> > receivedCells; 
  // map receive process => receive buffers 
  std::map<int, std::vector<Real> > receivedData; 
  //list of receive requests
  std::vector<MPI_Request> receiveRequests;
  // map dccrg cellID => data received from it
  std::map<CellID, Real* > combinedReceivedData; 

  // map send process => send buffers  to each process
  std::map<int, std::vector<Real> > sendData; 
  //list of send requests
  std::vector<MPI_Request> sendRequests;

  
  
  //Compute what we will receive, so what dccrg cells and from who
  for (int k=0; k<gridDims[2]; k++) {
    for (int j=0; j<gridDims[1]; j++) {
      for (int i=0; i<gridDims[0]; i++) {
	const std::array<int, 3> globalIndices = momentsGrid.getGlobalIndices(i,j,k);
	const dccrg::Types<3>::indices_t  indices = {{(uint64_t)globalIndices[0],
						      (uint64_t)globalIndices[1],
						      (uint64_t)globalIndices[2]}}; //cast to avoid warnings
	CellID dccrgCell = mpiGrid.get_existing_cell(indices, 0, mpiGrid.mapping.get_maximum_refinement_level());
	int process = mpiGrid.get_process(dccrgCell);
	receivedCells[process].insert(dccrgCell); //cells are ordered (sorted) in set
      }
    }
  }

  // Post receives
  receiveRequests.resize(receivedCells.size());
  
  ii=0;
  for(auto const &receives: receivedCells){
    int process = receives.first;
    int count = receives.second.size();
    receivedData[process].resize(count * fsgrids::moments::N_MOMENTS);
    MPI_Irecv(receivedData[process].data(), count * fsgrids::moments::N_MOMENTS * sizeof(Real), MPI_BYTE, process, 1, MPI_COMM_WORLD,&(receiveRequests[ii]));
    ii++;
  }

  // Compute where to send data and what to send
  
  for(int i=0; i< dccrgCells.size(); i++) {
     //Collect data to send for this dccrg cell
     auto cellParams = mpiGrid[dccrgCells[i]]->get_cell_parameters();
     std::array<Real, fsgrids::moments::N_MOMENTS> sendBuffer; 
     if(!dt2) {
        sendBuffer[fsgrids::moments::RHOM] = cellParams[CellParams::RHOM];
	sendBuffer[fsgrids::moments::RHOQ] = cellParams[CellParams::RHOQ];
	sendBuffer[fsgrids::moments::VX] = cellParams[CellParams::VX];
	sendBuffer[fsgrids::moments::VY] = cellParams[CellParams::VY];
	sendBuffer[fsgrids::moments::VZ] = cellParams[CellParams::VZ];
	sendBuffer[fsgrids::moments::P_11] = cellParams[CellParams::P_11];
	sendBuffer[fsgrids::moments::P_22] = cellParams[CellParams::P_22];
	sendBuffer[fsgrids::moments::P_33] = cellParams[CellParams::P_33];
     } else {
        sendBuffer[fsgrids::moments::RHOM] = cellParams[CellParams::RHOM_DT2];
        sendBuffer[fsgrids::moments::RHOQ] = cellParams[CellParams::RHOQ_DT2];
        sendBuffer[fsgrids::moments::VX] = cellParams[CellParams::VX_DT2];
	sendBuffer[fsgrids::moments::VY] = cellParams[CellParams::VY_DT2];
	sendBuffer[fsgrids::moments::VZ] = cellParams[CellParams::VZ_DT2];
	sendBuffer[fsgrids::moments::P_11] = cellParams[CellParams::P_11_DT2];
	sendBuffer[fsgrids::moments::P_22] = cellParams[CellParams::P_22_DT2];
	sendBuffer[fsgrids::moments::P_33] = cellParams[CellParams::P_33_DT2];
     }

     //compute to which processes this cell maps
     std::set<int> mappedToProcesses;
     std::vector<CellID> fsCells = mapDccrgIdToFsGridGlobalID(mpiGrid, dccrgCells[i]);
     //loop over fsgrid cells which this dccrg cell maps to
     for (auto const &fsCellID : fsCells) {
       mappedToProcesses.insert(momentsGrid.getTaskForGlobalID(fsCellID).first);
     }

     //loop over all target processes, and add to sendData the data to send for this one dccrg cell
     for (auto process : mappedToProcesses){
       //       sendCellId[process].push_back(dccrgCells[i]);
       for( auto val: sendBuffer) {
	 //since dccrgCells is ordered, data sent to each process is also ordered according to dccrgCellid
	 sendData[process].push_back(val);  
       }
     }
  }
  
  // Post sends

  ii=0;
  sendRequests.resize(sendData.size());
  for(auto const &sends: sendData){
    int process = sends.first;
    int count = sends.second.size(); //note, compared to receive this includes all elements to be sent
    MPI_Isend(sends.second.data(), count * sizeof(Real), MPI_BYTE, process, 1, MPI_COMM_WORLD,&(sendRequests[ii]));
    ii++;
  }

  MPI_Waitall(receiveRequests.size(), receiveRequests.data(), MPI_STATUSES_IGNORE);
  
  for(auto const &receives: receivedCells){
    int process = receives.first; //data received from this process
    Real* receiveBuffer = receivedData[process].data(); // data received from process
    for(auto const &cell: receives.second){ //loop over cellids (dccrg) for receive
      // this part heavily relies on both sender and receiver having cellids sorted!
      combinedReceivedData[cell]=receiveBuffer;  //store pointer
      receiveBuffer+=fsgrids::moments::N_MOMENTS; //jump to next cell
    }
  }

  //Store the correct data into the fsGrid
  //loop over local fsgrid cells
  for (int k = 0; k < gridDims[2]; k++) {
    for (int j = 0; j < gridDims[1]; j++) {
      for (int i = 0; i < gridDims[0]; i++) {
	const std::array<int, 3> globalIndices = momentsGrid.getGlobalIndices(i,j,k);
	const dccrg::Types<3>::indices_t  indices = {{(uint64_t)globalIndices[0],
						      (uint64_t)globalIndices[1],
						      (uint64_t)globalIndices[2]}}; //cast to avoid warnings
	CellID dccrgCell = mpiGrid.get_existing_cell(indices, 0, mpiGrid.mapping.get_maximum_refinement_level());
	Real * dccrggridData = combinedReceivedData[dccrgCell];
	auto fsgridData = *momentsGrid.get(i,j,k);
	for(int l = 0; l < fsgrids::moments::N_MOMENTS; l++)
	  fsgridData[l] = dccrggridData[l];
      }
    }
  }
  MPI_Waitall(sendRequests.size(), sendRequests.data(), MPI_STATUSES_IGNORE);
}


void feedBgFieldsIntoFsGrid(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
    const std::vector<CellID>& cells, FsGrid< std::array<Real, fsgrids::bgbfield::N_BGB>, 2>& bgBGrid) {

  bgBGrid.setupForTransferIn(cells.size());

   // Setup transfer buffers
   std::vector< std::array<Real, fsgrids::bgbfield::N_BGB> > transferBuffer(cells.size());

   // Fill from cellParams
   #pragma omp parallel for
   for(int i=0; i< cells.size(); i++) {
      auto cellParams = mpiGrid[cells[i]]->get_cell_parameters();
      auto derivatives = mpiGrid[cells[i]]->derivatives;
      auto volumeDerivatives = mpiGrid[cells[i]]->derivativesBVOL;
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

   for(int i=0; i< cells.size(); i++) {
      bgBGrid.transferDataIn(cells[i] - 1, &transferBuffer[i]);
   }

   // Finish the actual transfer
   bgBGrid.finishTransfersIn();

}

void getVolumeFieldsFromFsGrid(FsGrid< std::array<Real, fsgrids::volfields::N_VOL>, 2>& volumeFieldsGrid,
                           dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                           const std::vector<CellID>& cells) {

   // Setup transfer buffers
   std::vector< std::array<Real, fsgrids::volfields::N_VOL> > transferBuffer(cells.size());

   // Setup transfer pointers
   volumeFieldsGrid.setupForTransferOut(cells.size());
   for(int i=0; i< cells.size(); i++) {
      std::array<Real, fsgrids::volfields::N_VOL>* thisCellData = &transferBuffer[i];
      volumeFieldsGrid.transferDataOut(cells[i] - 1, thisCellData);
   }
   // Do the transfer
   volumeFieldsGrid.finishTransfersOut();

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
   std::vector< std::array<Real, fsgrids::dperb::N_DPERB> > dperbTransferBuffer(cells.size());
   std::vector< std::array<Real, fsgrids::dmoments::N_DMOMENTS> > dmomentsTransferBuffer(cells.size());
   std::vector< std::array<Real, fsgrids::bgbfield::N_BGB> > bgbfieldTransferBuffer(cells.size());

   // Transfer dperbGrid data
   dperbGrid.setupForTransferOut(cells.size());
   for(int i=0; i< cells.size(); i++) {
      std::array<Real, fsgrids::dperb::N_DPERB>* thisCellData = &dperbTransferBuffer[i];
      dperbGrid.transferDataOut(cells[i] - 1, thisCellData);
   }
   // Do the transfer
   dperbGrid.finishTransfersOut();

   // Transfer dmomentsGrid data
   dmomentsGrid.setupForTransferOut(cells.size());
   for(int i=0; i< cells.size(); i++) {
      std::array<Real, fsgrids::dmoments::N_DMOMENTS>* thisCellData = &dmomentsTransferBuffer[i];
      dmomentsGrid.transferDataOut(cells[i] - 1, thisCellData);
   }
   // Do the transfer
   dmomentsGrid.finishTransfersOut();

   // Transfer bgbfieldGrid data
   bgbfieldGrid.setupForTransferOut(cells.size());
   for(int i=0; i< cells.size(); i++) {
      std::array<Real, fsgrids::bgbfield::N_BGB>* thisCellData = &bgbfieldTransferBuffer[i];
      bgbfieldGrid.transferDataOut(cells[i] - 1, thisCellData);
   }
   // Do the transfer
   bgbfieldGrid.finishTransfersOut();

   // Distribute data from the transfer buffers back into the appropriate mpiGrid places
   #pragma omp parallel for
   for(int i=0; i< cells.size(); i++) {
      std::array<Real, fsgrids::dperb::N_DPERB>* dperb = &dperbTransferBuffer[i];
      std::array<Real, fsgrids::dmoments::N_DMOMENTS>* dmoments = &dmomentsTransferBuffer[i];
      std::array<Real, fsgrids::bgbfield::N_BGB>* bgbfield = &bgbfieldTransferBuffer[i];
      auto cellParams = mpiGrid[cells[i]]->get_cell_parameters();

      mpiGrid[cells[i]]->derivatives[fieldsolver::drhomdx] = dmoments->at(fsgrids::dmoments::drhomdx);
      mpiGrid[cells[i]]->derivatives[fieldsolver::drhomdy] = dmoments->at(fsgrids::dmoments::drhomdy);
      mpiGrid[cells[i]]->derivatives[fieldsolver::drhomdz] = dmoments->at(fsgrids::dmoments::drhomdz);
      mpiGrid[cells[i]]->derivatives[fieldsolver::drhoqdx] = dmoments->at(fsgrids::dmoments::drhoqdx);
      mpiGrid[cells[i]]->derivatives[fieldsolver::drhoqdy] = dmoments->at(fsgrids::dmoments::drhoqdy);
      mpiGrid[cells[i]]->derivatives[fieldsolver::drhoqdz] = dmoments->at(fsgrids::dmoments::drhoqdz);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dp11dx] = dmoments->at(fsgrids::dmoments::dp11dx);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dp11dy] = dmoments->at(fsgrids::dmoments::dp11dy);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dp11dz] = dmoments->at(fsgrids::dmoments::dp11dz);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dp22dx] = dmoments->at(fsgrids::dmoments::dp22dx);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dp22dy] = dmoments->at(fsgrids::dmoments::dp22dy);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dp22dz] = dmoments->at(fsgrids::dmoments::dp22dz);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dp33dx] = dmoments->at(fsgrids::dmoments::dp33dx);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dp33dy] = dmoments->at(fsgrids::dmoments::dp33dy);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dp33dz] = dmoments->at(fsgrids::dmoments::dp33dz);

      mpiGrid[cells[i]]->derivatives[fieldsolver::dVxdx] = dmoments->at(fsgrids::dmoments::dVxdx);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dVxdy] = dmoments->at(fsgrids::dmoments::dVxdy);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dVxdz] = dmoments->at(fsgrids::dmoments::dVxdz);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dVydx] = dmoments->at(fsgrids::dmoments::dVydx);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dVydy] = dmoments->at(fsgrids::dmoments::dVydy);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dVydz] = dmoments->at(fsgrids::dmoments::dVydz);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dVzdx] = dmoments->at(fsgrids::dmoments::dVzdx);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dVzdy] = dmoments->at(fsgrids::dmoments::dVzdy);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dVzdz] = dmoments->at(fsgrids::dmoments::dVzdz);

      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBxdy] = dperb->at(fsgrids::dperb::dPERBxdy);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBxdz] = dperb->at(fsgrids::dperb::dPERBxdz);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBydx] = dperb->at(fsgrids::dperb::dPERBydx);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBydz] = dperb->at(fsgrids::dperb::dPERBydz);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBzdx] = dperb->at(fsgrids::dperb::dPERBzdx);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBzdy] = dperb->at(fsgrids::dperb::dPERBzdy);

      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBxdyy] = dperb->at(fsgrids::dperb::dPERBxdyy);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBxdzz] = dperb->at(fsgrids::dperb::dPERBxdzz);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBydxx] = dperb->at(fsgrids::dperb::dPERBydxx);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBydzz] = dperb->at(fsgrids::dperb::dPERBydzz);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBzdxx] = dperb->at(fsgrids::dperb::dPERBzdxx);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBzdyy] = dperb->at(fsgrids::dperb::dPERBzdyy);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBxdyz] = dperb->at(fsgrids::dperb::dPERBxdyz);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBydxz] = dperb->at(fsgrids::dperb::dPERBydxz);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dPERBzdxy] = dperb->at(fsgrids::dperb::dPERBzdxy);

      mpiGrid[cells[i]]->derivatives[fieldsolver::dBGBxdy] = bgbfield->at(fsgrids::bgbfield::dBGBxdy);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dBGBxdz] = bgbfield->at(fsgrids::bgbfield::dBGBxdz);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dBGBydx] = bgbfield->at(fsgrids::bgbfield::dBGBydx);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dBGBydz] = bgbfield->at(fsgrids::bgbfield::dBGBydz);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dBGBzdx] = bgbfield->at(fsgrids::bgbfield::dBGBzdx);
      mpiGrid[cells[i]]->derivatives[fieldsolver::dBGBzdy] = bgbfield->at(fsgrids::bgbfield::dBGBzdy);
   }

}
    

void setupTechnicalFsGrid(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const std::vector<CellID>& cells, FsGrid< fsgrids::technical, 2>& technicalGrid) {

   technicalGrid.setupForTransferIn(cells.size());

   // Fill the transfer buffers from the spatial cell structs
   std::vector<fsgrids::technical> transferBuffer(cells.size());

   #pragma omp parallel for
   for(int i=0; i< cells.size(); i++) {

      fsgrids::technical* thisCellData = &transferBuffer[i];
      // Data needs to be collected from some different places for this grid.
      thisCellData->sysBoundaryFlag = mpiGrid[cells[i]]->sysBoundaryFlag;
      thisCellData->sysBoundaryLayer = mpiGrid[cells[i]]->sysBoundaryLayer;
      //thisCellData->maxFsDt = mpiGrid[i]->get_cell_parameters()[CellParams::MAXFDT];
      thisCellData->maxFsDt = std::numeric_limits<Real>::max();
   }
   for(int i=0; i< cells.size(); i++) {
      technicalGrid.transferDataIn(cells[i] - 1,&transferBuffer[i]);
   }

   technicalGrid.finishTransfersIn();
}

void getFsGridMaxDt(FsGrid< fsgrids::technical, 2>& technicalGrid,
      dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const std::vector<CellID>& cells) {

   technicalGrid.setupForTransferOut(cells.size());

   // Buffer to store contents of the grid
   std::vector<fsgrids::technical> transferBuffer(cells.size());

   for(int i=0; i< cells.size(); i++) {
      fsgrids::technical* thisCellData = &transferBuffer[i];
      technicalGrid.transferDataOut(cells[i] - 1, thisCellData);
   }

   technicalGrid.finishTransfersOut();

   // After the transfer is completed, stuff the recieved maxFDt into the cells.
   #pragma omp parallel for
   for(int i=0; i< cells.size(); i++) {
      mpiGrid[cells[i]]->get_cell_parameters()[CellParams::MAXFDT] = transferBuffer[i].maxFsDt;
      mpiGrid[cells[i]]->get_cell_parameters()[CellParams::FSGRID_RANK] = transferBuffer[i].fsGridRank;
      mpiGrid[cells[i]]->get_cell_parameters()[CellParams::FSGRID_BOUNDARYTYPE] = transferBuffer[i].sysBoundaryFlag;
   }
}

