#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#ifndef CPU_SORT_IDS_HPP
#define CPU_SORT_IDS_HPP


// Comparator function for sorting vector of pairs
template<typename ID> inline bool paircomparator( const std::pair<ID, ID> & l, const std::pair<ID, ID> & r ) {
   return l.first < r.first;
}



template<typename ID, typename LENGTH> inline bool sortIds(const ID id, 
                                                           const LENGTH meshSize, 
                                                           std::vector<ID>& ids)
{
   
   std::vector< std::pair<ID, ID>> sortedIds(ids.size);
   
   //TODO conditionally parallel version?
#pragma omp parallel for
   for (uint i = 0; i < Ids.size() ; ++i ) {
      const ID id = ids[i];

      switch( dimension ) {
      case 0: {
         const ID idMapped = id; // Mapping the block id to different coordinate system if dimension is not zero:
         sortedIds[i] = std::make_pair( idMapped, id );
      }
         break;
       case 1: {
          // Do operation: 
          //   id = x + y*x_max + z*y_max*x_max 
          //=> id' = id - (x + y*x_max) + y + x*y_max = x + y*x_max + z*y_max*x_max - (x + y*x_max) + y + x*y_max
          //          = y + x*y_max + z*y_max*x_max
          const ID x_index = id % meshSize[0];
          const ID y_index = (id / meshSize[0]) % meshSize[1];
          
          // Mapping the block id to different coordinate system if dimension is not zero:
          const ID idMapped = id - (x_index + y_index*meshSize[0]) + y_index + x_index * meshSize[1];
          
          sortedIds[i] = std::make_pair( idMapped, block );
       }
         break;
       case 2: {
          // Do operation: 
          //   id  = x + y*x_max + z*y_max*x_max 
          //=> id' = z + y*z_max + x*z_max*y_max
          const ID x_index = id % meshSize[0];
          const ID y_index = (id / meshSize[0]) % meshSize[1];
          const ID z_index = (id / (meshSize[0] * meshSize[1]));

          // Mapping the id id to different coordinate system if dimension is not zero:
          //const uint idMapped 
          //  = z_indice 
          //  + y_indice * meshSize[2] 
          //  + x_indice*meshSize[1]*meshSize[2];
          const ID idMapped 
            = z_index 
            + y_index*meshSize[2]
            + x_index*meshSize[1]*meshSize[2];
          sortedIds[i] = std::make_pair( idMapped, id );
       }
          break;
      }
   }
   // Finally, sort the list and store the sorted blocks in ids
   std::sort( sortedIds.begin(), sortedIds.end(), paircomparator<ID> );
   for (uint i = 0; i < Ids.size() ; ++i ) {
      ids[i] = sortedIds[i].first();
   }
   
}

#endif
