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
#ifndef __CINT__
#include <vector>
#endif

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
   static const int kMaxMc = 8;  // generator-level photons kept per cluster

   StFcsClusterFeatureMaker(const Char_t* name = "FcsClusFeat");
   ~StFcsClusterFeatureMaker();

   Int_t Init();
   Int_t Make();
   Int_t Finish();

   void setOutputFile(const char* f) { mFileName = f; }
   void setEnergyThreshold(float e) { mEmin = e; }        // min cluster energy to store
   void setTowerThreshold(float e) { mTowerEmin = e; }    // min tower energy into the image
   void setSaveTruth(int v) { mSaveTruth = v; }           // 1 = try to read g2t tables
   // Generator-level ("particle level") truth: collect the generated photons
   // from g2t_track/g2t_vertex, project them onto the ECal plane and match them
   // to clusters. Independent of how GEANT attributed the energy deposits, so
   // it gives a label even if hit-level attribution goes to shower secondaries.
   void setSaveMcTruth(int v) { mSaveMcTruth = v; }
   void setMcMatchRadius(float cm) { mMcMatchR = cm; }    // photon-to-cluster match, cm

   // ---- injecting generated photons from outside (the MuDst path) ----
   //
   // On a .fzd chain this maker reads g2t_track/g2t_vertex itself. On a MuDst
   // there are no g2t tables, but StMuMcTrack carries the same generator-level
   // information - it is built from g2t_track_st - so StFcsMuMcTruthMaker reads
   // it there and pushes the photons in through these two calls, once per
   // event, before this maker runs. That keeps StMuDSTMaker out of this
   // package's link dependencies, which matters: an unresolved StMuDst symbol
   // would make this library fail to dlopen in the .fzd chain, where the MuDst
   // libraries are not loaded.
   void clearMcPhotons();
   void addMcPhoton(int id, int parent, int parentPid, float e,
                    float px, float py, float pz, float vx, float vy, float vz);

  private:
   void resetBranches();
   void fillTruth(class StFcsCluster* clu);
   void collectMcPhotons();                               // once per event
   void matchMcPhotons(class StFcsCluster* clu, int det);  // per cluster

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
   Int_t bNNeighborRaw;                     // StFcsCluster::nNeighbor() as stored, with repeats
   Int_t bCatStar;                          // category from StFcsClusterMaker (0/1/2)
   Float_t bChi2Ndf1, bChi2Ndf2;            // from StFcsPointMaker, 0 if it did not run
   Int_t bSeedId, bSeedRow, bSeedCol;
   Float_t bSeedE, bSeedFrac, bE2Frac, bE1e2Frac;
   Float_t bXW, bYW;                        // cell width [cm], so nothing downstream hardcodes it
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

   // ---- generator-level truth (particle level), simulation only ----
   // Filled from g2t_track + g2t_vertex, independent of the hit attribution.
   // mcLabel is the one to train on when it is >= 0.
   Int_t bMcLabel;          // 0 = no photon, 1 = one photon, 2 = two or more; -1 = no MC
   Int_t bNMcPhoton;        // generated photons projecting inside mMcMatchR of this cluster
   Int_t bMcTrkId[kMaxMc];  // g2t track id of each matched photon
   Int_t bMcParent[kMaxMc]; // g2t id of its parent track (the pi0, for a pi0 gun)
   Int_t bMcParentPid;      // GEANT pid of the leading photon's parent, 0 if none
   Float_t bMcE[kMaxMc];    // generated energy of each matched photon [GeV]
   Float_t bMcDr[kMaxMc];   // its projected distance from the cluster centroid [cm]
   Float_t bMcX[kMaxMc];    // projected position on the ECal plane, STAR frame [cm]
   Float_t bMcY[kMaxMc];
   Float_t bMcSep;          // separation of the two leading matched photons [cm], -1 if <2
   Float_t bMcSepCell;      // the same in tower units - the merge-transition variable
   Float_t bMcZgg;          // |E1-E2|/(E1+E2) of the two leading matched photons, -1 if <2
   Int_t bNMcPhotonEvent;   // generated photons in the event reaching either ECal half

   int mSaveMcTruth = 1;
   float mMcMatchR = 11.0;  // cm, about two ECal towers
   int mMcExternal = 0;     // 1 when photons were injected for this event

   // hidden from CINT and kept last - rootcint cannot digest a nested struct
   // plus std::vector in a dictionary header
#ifndef __CINT__
   struct McPhoton {
      int id;
      int parent;
      int parentPid;
      float e;
      float p[3];   // generated momentum
      float v[3];   // start vertex, STAR frame [cm]
      float proj[2][2];  // [det 0/1][x,y] projection onto that ECal plane [cm]
      int projOk[2];
   };
   std::vector<McPhoton> mMcPhotons;
   void projectPhoton(McPhoton& ph);  // ray-plane onto both ECal halves
#endif

#ifndef SKIPDefImp
   ClassDef(StFcsClusterFeatureMaker, 0)
#endif
};

#endif
