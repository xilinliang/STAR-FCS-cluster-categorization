// StFcsTowerAssoc.h - recover the cluster <-> tower association on picoDst.
//
// WHY THIS EXISTS
//
// StEvent's StFcsCluster owns a list of StFcsHit pointers, so on a MuDst/StEvent
// chain the towers of a cluster are simply clu->hits(). StPicoFcsCluster stores
// no such list: picoDst keeps the cluster scalars (energy, x, y, nTowers,
// sigmaMin/Max, theta, chi2) in one branch and the tower hits (detectorId, id,
// energy) in another, with nothing linking them. That is why the first picoDst
// maker in this package could only use feature set 6 - the six variables that
// need no tower list.
//
// The link is recoverable geometrically, because both sides are in the same
// frame. StFcsCluster::x(),y() are the energy-weighted centroid in CELL units,
// and a tower's id maps to a cell through StFcsDb::getRowNumber/getColumnNumber,
// whose centre sits at (column - 0.5, row - 0.5). So: give every tower to the
// cluster whose centroid is nearest, then - since the picoDst does store the
// true nTowers per cluster - keep only that many, highest energy first.
//
// HOW WELL IT WORKS, measured on 167 ECal clusters of pi0.e30.vz0.run6.picoDst:
//
//   recovered nTowers == stored nTowers        92 %   (97 % within +-1)
//   recovered energy - stored energy           median 0.0000 GeV
//                                              5-95 % of dE/E: -0.9 % .. +9 %
//   recovered centroid - stored centroid       median 0.086 cells (~0.5 cm)
//                                              per-axis median bias 0.0000 cells
//
// The residual comes entirely from towers carrying a negligible share of the
// energy sitting between two clusters; there is no systematic shift, which is
// the check that the id -> cell mapping above is right.
//
// USE IT FOR SHAPES, NOT FOR SCALARS. Everything picoDst already stores -
// energy, x, y, nTowers, sigmaMin, sigmaMax, theta - should be read from the
// cluster, because those are exact. The recovered tower list is for the things
// picoDst does not store: the seed tower, the 3x3/5x5/11x11 grids, and the
// energy-weighted second moments. Feature sets 3, 13 and 34 all become
// computable from picoDst this way.
//
// Header-only, no ROOT and no STAR dependency, so it can be unit-tested with a
// plain compiler - see testAssoc.C. Include it from a .cxx, never from a
// dictionary header: rootcint does not need to see any of this.
//
// author: generated for Xilin Liang

#ifndef STAR_StFcsTowerAssoc_HH
#define STAR_StFcsTowerAssoc_HH

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace StFcsTowerAssoc {

// One tower hit. row and col are 1-based, exactly as StFcsDb::getRowNumber and
// getColumnNumber return them.
struct Tower {
   int id;
   int row;
   int col;
   float e;
};

// Cell-unit centre of a tower, in the same coordinates as StFcsCluster::x()/y().
inline float cellX(const Tower& t) { return t.col - 0.5f; }
inline float cellY(const Tower& t) { return t.row - 0.5f; }

// Assign each tower to one cluster.
//
//   tow       towers of ONE detector half, in any order
//   nClu      number of clusters in that same half
//   cluCol    cluster x() [cell units], nClu entries
//   cluRow    cluster y() [cell units], nClu entries
//   cluNTow   stored nTowers per cluster, or 0 to disable the truncation step
//   maxDist   ignore a tower further than this from every centroid [cells].
//             5.0 is the plateau: below ~3 real towers start being dropped,
//             above it nothing changes because the nTowers truncation takes over.
//   owner     output, one entry per tower: cluster index, or -1 if unassigned
//
// The truncation is what makes this accurate. Nearest-centroid alone gets the
// tower count right 50 % of the time; keeping only the stored nTowers highest
// towers per cluster takes that to 92 %, because the towers it discards are the
// low-energy stragglers that clustering did not include either.
inline void assign(const std::vector<Tower>& tow, int nClu, const float* cluCol,
                   const float* cluRow, const int* cluNTow, float maxDist,
                   std::vector<int>& owner) {
   const int nt = (int)tow.size();
   owner.assign(nt, -1);
   if (nClu <= 0 || nt == 0) return;

   const double maxD2 = (maxDist > 0) ? double(maxDist) * maxDist : 1e30;

   // step 1: nearest centroid
   for (int i = 0; i < nt; i++) {
      const double x = cellX(tow[i]);
      const double y = cellY(tow[i]);
      int best = -1;
      double bestD2 = maxD2;
      for (int c = 0; c < nClu; c++) {
         const double dx = x - cluCol[c];
         const double dy = y - cluRow[c];
         const double d2 = dx * dx + dy * dy;
         if (d2 <= bestD2) {
            bestD2 = d2;
            best = c;
         }
      }
      owner[i] = best;
   }

   // step 2: keep at most the stored nTowers per cluster, highest energy first
   if (!cluNTow) return;
   std::vector<std::pair<float, int> > mine;  // (-energy, tower index) so sort() is descending
   for (int c = 0; c < nClu; c++) {
      const int nKeep = cluNTow[c];
      if (nKeep <= 0) continue;  // unknown for this cluster: leave it alone
      mine.clear();
      for (int i = 0; i < nt; i++)
         if (owner[i] == c) mine.push_back(std::make_pair(-tow[i].e, i));
      if ((int)mine.size() <= nKeep) continue;
      std::sort(mine.begin(), mine.end());
      for (size_t k = (size_t)nKeep; k < mine.size(); k++) owner[mine[k].second] = -1;
   }
}

// Convenience: gather the towers of cluster c after assign().
inline void gather(const std::vector<Tower>& tow, const std::vector<int>& owner, int c,
                   std::vector<Tower>& out) {
   out.clear();
   for (size_t i = 0; i < tow.size() && i < owner.size(); i++)
      if (owner[i] == c) out.push_back(tow[i]);
}

// ---------------------------------------------------------------------------
// Neighbour clusters.
//
// StFcsClusterMaker calls two clusters neighbours when some hit is within
// mNeighborDistance_Ecal = 1.01 cells of a tower of each - that is, when they
// touch across the 4-neighbourhood of a tower. It records that by pushing a
// pointer into StFcsCluster::mNeighbor, and nNeighbor() is that vector's size.
//
// TWO THINGS TO KNOW BEFORE USING THIS.
//
// First, the vector is not de-duplicated: every hit that sees the same pair
// pushes again, so STAR's nNeighbor() counts LINKINGS, not distinct clusters,
// and its value depends on the order hits were consumed in. That order cannot
// be recovered from a picoDst. What is computed here is the number of DISTINCT
// neighbouring clusters, which is the same quantity with the ordering
// dependence taken out - and StFcsClusterFeatureMaker de-duplicates the STAR
// list the same way, so both tiers feed the model the same definition. The raw
// STAR count is still written to the tree, as nNeighborRaw, wherever it exists.
//
// Second, on picoDst this runs on the RECOVERED tower lists, so it inherits
// their accuracy. Adjacency is the robust part of that - a tower assigned to
// the wrong one of two touching clusters leaves them touching either way - but
// it is a reconstruction, not a reading.
//
//   tow/owner   as they come out of assign()
//   nRow,nCol   detector size, from StFcsDb
//   dist        1.01 reproduces StFcsClusterMaker's ECal setting; a larger
//               value widens the neighbourhood (1.42 would take the diagonals)
inline void neighborCounts(const std::vector<Tower>& tow, const std::vector<int>& owner, int nClu,
                           int nRow, int nCol, float dist, std::vector<int>& nNeighbor) {
   nNeighbor.assign(nClu > 0 ? nClu : 0, 0);
   if (nClu <= 1 || tow.empty() || nRow <= 0 || nCol <= 0) return;

   // grid -> tower index, so the neighbourhood of a tower is a lookup rather
   // than a scan over every other tower
   std::vector<int> cell((size_t)nRow * nCol, -1);
   for (size_t i = 0; i < tow.size(); i++) {
      if (tow[i].row < 1 || tow[i].row > nRow || tow[i].col < 1 || tow[i].col > nCol) continue;
      cell[(size_t)(tow[i].row - 1) * nCol + (tow[i].col - 1)] = (int)i;
   }

   // offsets within dist of a cell centre, diagonals included only if dist allows
   const int reach = (int)dist + 1;
   std::vector<int> adj((size_t)nClu * nClu, 0);
   std::vector<int> seen;

   for (size_t i = 0; i < tow.size(); i++) {
      // clusters owning this tower or any tower within dist of it. A tower that
      // assign() left unowned still acts as a bridge here, exactly as an
      // unclustered hit does in StFcsClusterMaker.
      seen.clear();
      for (int dr = -reach; dr <= reach; dr++) {
         for (int dc = -reach; dc <= reach; dc++) {
            if (double(dr) * dr + double(dc) * dc > double(dist) * dist) continue;
            const int r = tow[i].row + dr, c = tow[i].col + dc;
            if (r < 1 || r > nRow || c < 1 || c > nCol) continue;
            const int j = cell[(size_t)(r - 1) * nCol + (c - 1)];
            if (j < 0) continue;
            const int o = (j < (int)owner.size()) ? owner[j] : -1;
            if (o < 0 || o >= nClu) continue;
            bool have = false;
            for (size_t k = 0; k < seen.size(); k++)
               if (seen[k] == o) have = true;
            if (!have) seen.push_back(o);
         }
      }
      for (size_t a = 0; a + 1 < seen.size(); a++)
         for (size_t b = a + 1; b < seen.size(); b++) {
            adj[(size_t)seen[a] * nClu + seen[b]] = 1;
            adj[(size_t)seen[b] * nClu + seen[a]] = 1;
         }
   }

   for (int c = 0; c < nClu; c++) {
      int n = 0;
      for (int d = 0; d < nClu; d++)
         if (d != c && adj[(size_t)c * nClu + d]) n++;
      nNeighbor[c] = n;
   }
}

}  // namespace StFcsTowerAssoc

#endif
