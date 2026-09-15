// testFeatures.C - smoke test for StFcsClusterFeatures.h
//
//   root -b -q testFeatures.C            (plain root is enough, no STAR needed)
//   g++ -DSTANDALONE -o testFeatures testFeatures.C && ./testFeatures
//
// Builds synthetic single-photon and two-photon tower patterns, runs both
// feature sets over them, and checks the invariants that must hold if the
// definitions are right. Run it after touching StFcsClusterFeatures.h.
//
// author: generated for Xilin Liang

#include <cstdio>
#include <cmath>
#include <cstring>

#include "StRoot/StFcsMLCategoryMaker/StFcsClusterFeatures.h"

using namespace StFcsClusterFeatures;

namespace {

const int NW = 11;
const float XW = 5.542;  // FCS ECal cell width [cm]; only the ratio matters here

int nFail = 0;

void check(bool ok, const char* what) {
   printf("  %-58s %s\n", what, ok ? "OK" : "FAIL");
   if (!ok) nFail++;
}

// Drop nPhoton gaussian showers on an 11x11 grid and pack the non-zero towers
// into the arrays ClusterInput wants. seedRow/seedCol place the cluster
// somewhere sensible in the detector.
struct Fake {
   float te[NW * NW];
   int trow[NW * NW];
   int tcol[NW * NW];
   int nTow;
   float e, x, y;
};

Fake makeCluster(int nPhoton, float sep, float amp = 10.0, float width = 0.9) {
   Fake f;
   f.nTow = 0;
   const int seedRow = 17, seedCol = 11;
   double sumE = 0, sumX = 0, sumY = 0;
   for (int dr = -5; dr <= 5; dr++) {
      for (int dc = -5; dc <= 5; dc++) {
         double v = 0;
         for (int p = 0; p < nPhoton; p++) {
            const double c0 = (p == 0) ? -sep / 2.0 : sep / 2.0;
            const double r0 = 0.0;
            v += amp * exp(-((dr - r0) * (dr - r0) + (dc - c0) * (dc - c0)) / (2 * width * width));
         }
         if (v < 0.01) continue;
         f.te[f.nTow] = v;
         f.trow[f.nTow] = seedRow + dr;
         f.tcol[f.nTow] = seedCol + dc;
         sumE += v;
         sumX += v * (seedCol + dc);
         sumY += v * (seedRow + dr);
         f.nTow++;
      }
   }
   f.e = sumE;
   f.x = sumX / sumE;
   f.y = sumY / sumE;
   return f;
}

ClusterInput toInput(const Fake& f) {
   ClusterInput c;
   c.e = f.e;
   c.x = f.x;
   c.y = f.y;
   c.sigmaMin = 0.8;  // stand-ins: StFcsClusterMaker fills these for real clusters
   c.sigmaMax = 1.5;
   c.theta = 0.0;
   c.nTowers = f.nTow;
   c.nNeighbor = 0;
   c.xw = XW;
   c.yw = XW;
   c.nTow = f.nTow;
   c.towerE = f.te;
   c.towerRow = f.trow;
   c.towerCol = f.tcol;
   return c;
}

void dump(int set, const char* label, const ClusterInput& c) {
   float v[kNVarMax];
   const int n = compute(set, c, v);
   const char** names = varNames(set);
   printf("%s, set %d (%d vars):\n", label, set, n);
   for (int i = 0; i < n; i++) {
      printf("  %-11s %10.4f", names[i], v[i]);
      if (i % 3 == 2) printf("\n");
   }
   printf("\n");
}

}  // namespace

int testFeatures() {
   Fake one = makeCluster(1, 0.0);
   Fake two = makeCluster(2, 2.6);
   ClusterInput c1 = toInput(one);
   ClusterInput c2 = toInput(two);

   dump(6, "single photon", c1);
   dump(6, "two photons", c2);
   dump(13, "single photon", c1);
   dump(13, "two photons", c2);
   dump(34, "two photons", c2);

   // set 6 must be exactly the first six of set 13 - that equality is what lets
   // a model trained on MuDst be applied to picoDst input
   {
      float a6[kNVarMax], a13[kNVarMax], b6[kNVarMax], b13[kNVarMax];
      compute(6, c1, a6);
      compute(13, c1, a13);
      compute(6, c2, b6);
      compute(13, c2, b13);
      bool same = true;
      for (int i = 0; i < 6; i++) {
         if (fabs(a6[i] - a13[i]) > 1e-6 || fabs(b6[i] - b13[i]) > 1e-6) same = false;
         if (strcmp(varNames(6)[i], varNames(13)[i]) != 0) same = false;
      }
      printf("checks, feature set 6:\n");
      check(same, "set 6 == first six of set 13, values and names");
      ClusterInput noTowers = c1;
      noTowers.nTow = 0;
      noTowers.towerE = 0;
      noTowers.towerRow = 0;
      noTowers.towerCol = 0;
      float t6[kNVarMax], t13[kNVarMax];
      check(compute(6, noTowers, t6) == 6, "set 6 works with no tower list (the picoDst case)");
      check(compute(13, noTowers, t13) == 0, "set 13 correctly refuses with no tower list");
   }

   float a[kNVarMax], b[kNVarMax];
   compute(13, c1, a);
   compute(13, c2, b);

   printf("checks, feature set 13:\n");
   check(fabs(a[0] - log(c1.e)) < 1e-4, "logE reproduces log(cluster energy)");
   check(a[6] > b[6], "seedFrac larger for one photon than for two");
   check(a[8] > b[8], "e1e2Asym larger for one photon than for two");
   check(b[9] > a[9], "sigX larger for the two-photon pattern (split along x)");
   check(fabs(a[9] - a[10]) < 0.05, "single photon is round: sigX ~ sigY");
   check(a[6] <= 1.0 && b[6] <= 1.0, "seed fraction never exceeds 1");

   compute(34, c1, a);
   compute(34, c2, b);
   printf("checks, feature set 34:\n");
   check(fabs(a[0] - c1.e) < 1e-3, "e is the cluster energy");
   check(b[4] > a[4], "radius larger for two photons");
   check(b[5] > a[5], "dispersion larger for two photons");
   double sum1 = 0;
   for (int i = 8; i < 33; i++) sum1 += a[i];
   check(sum1 + a[33] > 0.999 && sum1 + a[33] < 1.001,
         "5x5 fractions + eOut = 1 (energy is conserved)");
   check(a[33] >= 0.0, "energy outside the 5x5 is not negative");
   double sum2 = 0;
   for (int i = 8; i < 33; i++) sum2 += b[i];
   check(sum2 + b[33] > 0.999 && sum2 + b[33] < 1.001,
         "same for the two-photon cluster");
   check(a[8 + 12] > 0.1, "central tower t22 carries a sizeable fraction");

   printf("\n%s\n", nFail ? "SOME CHECKS FAILED" : "all checks passed");
   return nFail;
}

#ifdef STANDALONE
int main() { return testFeatures(); }
#endif
