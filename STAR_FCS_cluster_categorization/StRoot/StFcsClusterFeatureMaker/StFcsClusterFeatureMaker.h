// class StFcsClusterFeatureMaker
//
// Dumps one TTree entry per FCS ECal cluster with everything a cluster
// categorization model needs: shape scalars, an NxN tower image around the
// seed, and (in simulation) the GEANT truth that contributed to the cluster.
//
// Chain position:
//   ... StFcsClusterMaker -> [StFcsPointMaker] -> StFcsClusterFeatureMaker
// Put it AFTER StFcsPointMaker when building a training sample, so that the
// 1-photon / 2-photon chi2/ndf produced by the standard fitter are stored
// alongside the ML features (they are the baseline you have to beat).
//
// author: generated for Xilin Liang (STAR FCS ECal / ePIC cluster categorization)

#ifndef STAR_StFcsClusterFeatureMaker_HH
#define STAR_StFcsClusterFeatureMaker_HH

#include <string>

#include "StMaker.h"

class TFile;
class TTree;
class StFcsDb;
class StFcsCollection;

class StFcsClusterFeatureMaker : public StMaker {
  public:
   // NW must be odd. 11x11 covers a 61 x 61 cm^2 window in the ECal, which is
   // wide enough for two-photon showers up to the highest FCS energies.
   static const int kNW = 11;
   static const int kNPix = kNW * kNW;
   static const int kMaxTrk = 16;

   StFcsClusterFeatureMaker(const Char_t* name = "FcsClusFeat");
   ~StFcsClusterFeatureMaker();

   Int_t Init();
   Int_t Make();
   Int_t Finish();

   void setOutputFile(const char* f) { mFileName = f; }
   void setEnergyThreshold(float e) { mEmin = e; }        // min cluster energy to store
   void setTowerThreshold(float e) { mTowerEmin = e; }    // min tower energy into the image
   void setSaveTruth(int v) { mSaveTruth = v; }           // 1 = try to read g2t tables

  private:
   void resetBranches();
   void fillTruth(class StFcsCluster* clu);

   StFcsDb* mFcsDb = 0;
   StFcsCollection* mFcsColl = 0;

   std::string mFileName = "fcsEcalClusterFeatures.root";
   TFile* mFile = 0;
   TTree* mTree = 0;

   float mEmin = 0.5;
   float mTowerEmin = 0.0;
   int mSaveTruth = 1;

   // ---- branch buffers (one entry = one ECal cluster) ----
   Int_t bRun, bEvent, bDet, bClId, bNCluDet;
   Float_t bE, bX, bY;                      // x,y are in column/row units (STAR FCS convention)
   Float_t bStarX, bStarY, bStarZ;
   Float_t bEta, bPhi, bPt;
   Float_t bSigmaMax, bSigmaMin, bTheta;
   Int_t bNTowers, bNNeighbor, bNPoints;
   Int_t bCatStar;                          // category from StFcsClusterMaker (0/1/2)
   Float_t bChi2Ndf1, bChi2Ndf2;            // from StFcsPointMaker, 0 if it did not run
   Int_t bSeedId, bSeedRow, bSeedCol;
   Float_t bSeedE, bSeedFrac, bE2Frac, bE1e2Frac;
   Float_t bSigX, bSigY, bSigXY;            // energy-weighted second moments, cell units
   Float_t bImg[kNPix];                     // energies in the window, seed at the centre
   Float_t bMask[kNPix];                    // 1 if that tower belongs to THIS cluster

   // ---- truth (simulation only) ----
   Int_t bNTrk;
   Int_t bTrkId[kMaxTrk];
   Int_t bTrkPid[kMaxTrk];                  // GEANT pid
   Int_t bTrkParent[kMaxTrk];               // g2t next_parent_p
   Float_t bTrkE[kMaxTrk];                  // energy that track deposited in this cluster
   Float_t bTrkPtot[kMaxTrk];               // generated momentum of the track
   Int_t bTruthNPhoton;                     // photons above mTruthFrac of cluster energy
   Int_t bTruthSameParent;                  // 1 if the two leading photons share a parent
   Float_t bTruthPurity;                    // leading track energy / sum of track energies

   float mTruthFrac = 0.10;

#ifndef SKIPDefImp
   ClassDef(StFcsClusterFeatureMaker, 0)
#endif
};

#endif
