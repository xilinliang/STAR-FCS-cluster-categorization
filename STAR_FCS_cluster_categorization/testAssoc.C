// testAssoc.C - smoke test for StFcsTowerAssoc.h, the picoDst cluster <-> tower
// association.
//
//   root -b -q testAssoc.C                   (plain root is enough, no STAR needed)
//   g++ -DSTANDALONE -o testAssoc testAssoc.C && ./testAssoc
//
// Builds tower patterns whose true ownership is known by construction, runs the
// association over them, and checks it recovers that ownership. Run it after
// touching StFcsTowerAssoc.h.
//
// This is the part of the picoDst path that is a RECONSTRUCTION rather than a
// reading of the file, so it is the part worth testing.
//
// author: generated for Xilin Liang

#include <cmath>
#include <cstdio>
#include <vector>

#include "StRoot/StFcsMLCategoryMaker/StFcsTowerAssoc.h"

using namespace StFcsTowerAssoc;

namespace {

int nFail = 0;

void check(bool ok, const char* what) {
   printf("  %-62s %s\n", what, ok ? "OK" : "FAIL");
   if (!ok) nFail++;
}

// One gaussian shower dropped on the tower grid. Towers are 1-based in row and
// column; the centroid (cx, cy) is in cell units, so a tower's centre is at
// (col - 0.5, row - 0.5) - the convention StFcsCluster::x()/y() uses.
struct Shower {
   std::vector<Tower> tow;
   float e, cx, cy;
   int n;
};

Shower makeShower(float cx, float cy, float amp, float width, int idBase) {
   Shower s;
   s.e = 0;
   double sx = 0, sy = 0;
   const int col0 = (int)(cx + 0.5), row0 = (int)(cy + 0.5);
   for (int row = row0 - 4; row <= row0 + 4; row++) {
      for (int col = col0 - 4; col <= col0 + 4; col++) {
         if (row < 1 || col < 1) continue;
         const double dx = (col - 0.5) - cx;
         const double dy = (row - 0.5) - cy;
         const double v = amp * exp(-(dx * dx + dy * dy) / (2 * width * width));
         if (v < 0.01) continue;
         Tower t;
         t.id = idBase + (row - 1) * 22 + (col - 1);
         t.row = row;
         t.col = col;
         t.e = v;
         s.tow.push_back(t);
         s.e += v;
         sx += v * (col - 0.5);
         sy += v * (row - 0.5);
      }
   }
   s.n = (int)s.tow.size();
   s.cx = sx / s.e;
   s.cy = sy / s.e;
   return s;
}

}  // namespace

int testAssoc() {
   // ------------------------------------------------ two separated clusters
   {
      Shower a = makeShower(6.5, 17.5, 10.0, 0.9, 0);
      Shower b = makeShower(14.5, 17.5, 8.0, 0.9, 0);
      std::vector<Tower> tow = a.tow;
      const int nA = a.n;
      for (size_t i = 0; i < b.tow.size(); i++) tow.push_back(b.tow[i]);

      const float cx[2] = {a.cx, b.cx};
      const float cy[2] = {a.cy, b.cy};
      const int cn[2] = {a.n, b.n};
      std::vector<int> owner;
      assign(tow, 2, cx, cy, cn, 5.0, owner);

      bool right = true;
      for (int i = 0; i < (int)tow.size(); i++) {
         const int want = (i < nA) ? 0 : 1;
         if (owner[i] != want) right = false;
      }
      printf("two clusters, 8 cells apart:\n");
      check(right, "every tower goes back to the shower that made it");

      float eA = 0, eB = 0;
      int nRecA = 0, nRecB = 0;
      for (int i = 0; i < (int)tow.size(); i++) {
         if (owner[i] == 0) { eA += tow[i].e; nRecA++; }
         if (owner[i] == 1) { eB += tow[i].e; nRecB++; }
      }
      check(nRecA == a.n && nRecB == b.n, "recovered tower counts equal the true ones");
      check(fabs(eA - a.e) < 1e-4 && fabs(eB - b.e) < 1e-4, "recovered energies equal the true ones");
   }

   // ---------------------------------------- overlapping, truncation matters
   {
      // 2.5 cells apart: the tails interleave, so nearest-centroid alone hands
      // each cluster more towers than it has. Truncating to the stored nTowers
      // is what fixes the count.
      Shower a = makeShower(9.0, 17.5, 10.0, 1.0, 0);
      Shower b = makeShower(11.5, 17.5, 10.0, 1.0, 0);
      std::vector<Tower> tow;
      for (size_t i = 0; i < a.tow.size(); i++) tow.push_back(a.tow[i]);
      for (size_t i = 0; i < b.tow.size(); i++) tow.push_back(b.tow[i]);

      const float cx[2] = {a.cx, b.cx};
      const float cy[2] = {a.cy, b.cy};
      const int cn[2] = {a.n, b.n};
      const int noTrunc[2] = {0, 0};

      std::vector<int> loose, tight;
      assign(tow, 2, cx, cy, noTrunc, 5.0, loose);
      assign(tow, 2, cx, cy, cn, 5.0, tight);

      int nLoose0 = 0, nTight0 = 0, nTight1 = 0, nDropped = 0;
      for (size_t i = 0; i < tow.size(); i++) {
         if (loose[i] == 0) nLoose0++;
         if (tight[i] == 0) nTight0++;
         if (tight[i] == 1) nTight1++;
         if (tight[i] < 0) nDropped++;
      }
      printf("two clusters, 2.5 cells apart:\n");
      check(nTight0 <= a.n && nTight1 <= b.n, "truncation never gives a cluster more than its nTowers");
      check(nTight0 <= nLoose0, "truncation only removes towers, never adds any");
      check(nDropped > 0, "the surplus towers really were dropped");
      float eTight = 0, eAll = 0;
      for (size_t i = 0; i < tow.size(); i++) {
         eAll += tow[i].e;
         if (tight[i] >= 0) eTight += tow[i].e;
      }
      check(eTight > 0.98 * eAll, "what truncation drops carries under 2% of the energy");
   }

   // -------------------------------------------------- maxDist and edge cases
   {
      Shower a = makeShower(6.5, 17.5, 10.0, 0.9, 0);
      Tower far;
      far.id = 999;
      far.row = 30;
      far.col = 20;
      far.e = 0.5;
      std::vector<Tower> tow = a.tow;
      tow.push_back(far);
      const float cx[1] = {a.cx};
      const float cy[1] = {a.cy};
      const int cn[1] = {a.n + 1};  // deliberately loose, so only maxDist can reject the outlier
      std::vector<int> owner;
      assign(tow, 1, cx, cy, cn, 5.0, owner);
      printf("distance cut and edge cases:\n");
      check(owner[tow.size() - 1] == -1, "a tower beyond maxDist is left unassigned");

      std::vector<Tower> none;
      std::vector<int> o2;
      assign(none, 1, cx, cy, cn, 5.0, o2);
      check(o2.empty(), "no towers: no crash, empty result");
      assign(tow, 0, 0, 0, 0, 5.0, o2);
      check((int)o2.size() == (int)tow.size(), "no clusters: every tower unassigned");
      bool allMinus = true;
      for (size_t i = 0; i < o2.size(); i++)
         if (o2[i] != -1) allMinus = false;
      check(allMinus, "...and all of them are -1");

      // maxDist <= 0 must mean "no limit", not "reject everything"
      std::vector<int> o3;
      assign(tow, 1, cx, cy, 0, -1.0, o3);
      check(o3[tow.size() - 1] == 0, "maxDist <= 0 means no distance limit");
   }

   // ------------------------------------------------------ gather() agrees
   {
      Shower a = makeShower(6.5, 17.5, 10.0, 0.9, 0);
      Shower b = makeShower(14.5, 17.5, 8.0, 0.9, 0);
      std::vector<Tower> tow = a.tow;
      for (size_t i = 0; i < b.tow.size(); i++) tow.push_back(b.tow[i]);
      const float cx[2] = {a.cx, b.cx};
      const float cy[2] = {a.cy, b.cy};
      const int cn[2] = {a.n, b.n};
      std::vector<int> owner;
      assign(tow, 2, cx, cy, cn, 5.0, owner);
      std::vector<Tower> g0, g1;
      gather(tow, owner, 0, g0);
      gather(tow, owner, 1, g1);
      printf("gather():\n");
      check((int)g0.size() == a.n && (int)g1.size() == b.n, "returns the towers assign() gave each cluster");
      double sx = 0, sy = 0, w = 0;
      for (size_t i = 0; i < g0.size(); i++) {
         w += g0[i].e;
         sx += g0[i].e * cellX(g0[i]);
         sy += g0[i].e * cellY(g0[i]);
      }
      check(fabs(sx / w - a.cx) < 1e-3 && fabs(sy / w - a.cy) < 1e-3,
            "cellX/cellY reproduce the cluster centroid convention");
   }

   // ----------------------------------------------------- neighbour counting
   //
   // Hand-built tower sets, so the right answer is not in doubt. Towers are
   // (row, col), 1-based, and clusters touch when their towers are 4-adjacent.
   {
      std::vector<Tower> tow;
      std::vector<int> owner;
      // cluster 0 at columns 5-6, cluster 1 at columns 7-8 (touching),
      // cluster 2 far away at column 18
      const int col[6] = {5, 6, 7, 8, 18, 19};
      const int own[6] = {0, 0, 1, 1, 2, 2};
      for (int i = 0; i < 6; i++) {
         Tower t;
         t.id = i;
         t.row = 17;
         t.col = col[i];
         t.e = 1.0;
         tow.push_back(t);
         owner.push_back(own[i]);
      }
      std::vector<int> n;
      neighborCounts(tow, owner, 3, 34, 22, 1.01f, n);
      printf("neighbourCounts():\n");
      check(n.size() == 3, "one count per cluster");
      check(n[0] == 1 && n[1] == 1, "two touching clusters are each other's neighbour");
      check(n[2] == 0, "a cluster eight columns away has no neighbour");

      // Three single-tower clusters in a row, at columns 5, 6, 7. All THREE
      // come out as mutual neighbours, not a chain - and that is right rather
      // than sloppy. StFcsClusterMaker cross-links every pair of clusters that
      // a single hit touches, so the tower at column 6, which touches both of
      // its neighbours, makes the clusters at 5 and 7 neighbours of each other
      // as well, two cells apart though they are. Reproducing that is the point.
      std::vector<Tower> row3;
      std::vector<int> own3;
      for (int i = 0; i < 3; i++) {
         Tower t;
         t.id = i;
         t.row = 17;
         t.col = 5 + i;
         t.e = 1.0;
         row3.push_back(t);
         own3.push_back(i);
      }
      std::vector<int> n3;
      neighborCounts(row3, own3, 3, 34, 22, 1.01f, n3);
      check(n3[0] == 2 && n3[1] == 2 && n3[2] == 2,
            "one hit touching two clusters makes those two neighbours as well");

      // diagonal only: not a neighbour at 1.01, is one at 1.42
      std::vector<Tower> diag;
      std::vector<int> ownd;
      Tower a;
      a.id = 0; a.row = 17; a.col = 5; a.e = 1.0;
      Tower b;
      b.id = 1; b.row = 18; b.col = 6; b.e = 1.0;
      diag.push_back(a); ownd.push_back(0);
      diag.push_back(b); ownd.push_back(1);
      std::vector<int> nd;
      neighborCounts(diag, ownd, 2, 34, 22, 1.01f, nd);
      check(nd[0] == 0, "diagonal towers do not touch at 1.01 cells (StFcsClusterMaker's ECal rule)");
      neighborCounts(diag, ownd, 2, 34, 22, 1.42f, nd);
      check(nd[0] == 1, "...and do at 1.42");

      // an unassigned tower bridges two clusters, as an unclustered hit does
      // in StFcsClusterMaker
      std::vector<Tower> br;
      std::vector<int> ownb;
      const int bcol[3] = {5, 6, 7};
      const int bown[3] = {0, -1, 1};
      for (int i = 0; i < 3; i++) {
         Tower t;
         t.id = i; t.row = 17; t.col = bcol[i]; t.e = 1.0;
         br.push_back(t);
         ownb.push_back(bown[i]);
      }
      std::vector<int> nb;
      neighborCounts(br, ownb, 2, 34, 22, 1.01f, nb);
      check(nb[0] == 1 && nb[1] == 1, "an unassigned tower between two clusters links them");

      std::vector<int> none;
      neighborCounts(tow, owner, 1, 34, 22, 1.01f, none);
      check(none.size() == 1 && none[0] == 0, "a single cluster has no neighbours");
   }

   printf("\n%s\n", nFail ? "SOME CHECKS FAILED" : "all checks passed");
   return nFail;
}

#ifdef STANDALONE
int main() { return testAssoc(); }
#endif
