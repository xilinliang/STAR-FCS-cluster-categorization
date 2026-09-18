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

}  // namespace StFcsTowerAssoc

#endif
