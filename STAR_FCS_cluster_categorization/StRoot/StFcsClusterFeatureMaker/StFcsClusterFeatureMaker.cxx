// class StFcsClusterFeatureMaker
// see header for chain position and purpose

#include "StFcsClusterFeatureMaker.h"

#include <algorithm>
#include <map>
#include <vector>

#include "StEvent/StEnumerations.h"
#include "StEvent/StEvent.h"
#include "StEvent/StEventInfo.h"
#include "StEvent/StFcsCluster.h"
#include "StEvent/StFcsCollection.h"
#include "StEvent/StFcsHit.h"
#include "StEvent/StFcsPoint.h"
#include "StEventTypes.h"
#include "StFcsDbMaker/StFcsDb.h"
#include "StMessMgr.h"
#include "StThreeVectorF.hh"
#include "Stypes.h"
#include "TFile.h"
#include "TTree.h"
#include "tables/St_g2t_track_Table.h"

#ifndef SKIPDefImp
ClassImp(StFcsClusterFeatureMaker)
#endif

    StFcsClusterFeatureMaker::StFcsClusterFeatureMaker(const Char_t* name) : StMaker(name) {}

StFcsClusterFeatureMaker::~StFcsClusterFeatureMaker() {}

//-----------------------------------------------------------------------------
Int_t StFcsClusterFeatureMaker::Init() {
   mFcsDb = static_cast<StFcsDb*>(GetDataSet("fcsDb"));
   if (!mFcsDb) {
      LOG_ERROR << "StFcsClusterFeatureMaker::Init failed to get StFcsDb" << endm;
      return kStFatal;
   }

   mFile = new TFile(mFileName.c_str(), "RECREATE");
   mTree = new TTree("clusters", "FCS ECal cluster features");

   mTree->Branch("run", &bRun, "run/I");
   mTree->Branch("event", &bEvent, "event/I");
   mTree->Branch("det", &bDet, "det/I");  // 0 = ECal north, 1 = ECal south
   mTree->Branch("clid", &bClId, "clid/I");
   mTree->Branch("ncluDet", &bNCluDet, "ncluDet/I");

   mTree->Branch("e", &bE, "e/F");
   mTree->Branch("x", &bX, "x/F");  // column units
   mTree->Branch("y", &bY, "y/F");  // row units
   mTree->Branch("starx", &bStarX, "starx/F");
   mTree->Branch("stary", &bStarY, "stary/F");
   mTree->Branch("starz", &bStarZ, "starz/F");
   mTree->Branch("eta", &bEta, "eta/F");
   mTree->Branch("phi", &bPhi, "phi/F");
   mTree->Branch("pt", &bPt, "pt/F");

   mTree->Branch("sigmaMax", &bSigmaMax, "sigmaMax/F");
   mTree->Branch("sigmaMin", &bSigmaMin, "sigmaMin/F");
   mTree->Branch("theta", &bTheta, "theta/F");
   mTree->Branch("nTowers", &bNTowers, "nTowers/I");
   mTree->Branch("nNeighbor", &bNNeighbor, "nNeighbor/I");
   mTree->Branch("nPoints", &bNPoints, "nPoints/I");
   mTree->Branch("catStar", &bCatStar, "catStar/I");
   mTree->Branch("chi2ndf1", &bChi2Ndf1, "chi2ndf1/F");
   mTree->Branch("chi2ndf2", &bChi2Ndf2, "chi2ndf2/F");

   mTree->Branch("seedId", &bSeedId, "seedId/I");
   mTree->Branch("seedRow", &bSeedRow, "seedRow/I");
   mTree->Branch("seedCol", &bSeedCol, "seedCol/I");
   mTree->Branch("seedE", &bSeedE, "seedE/F");
   mTree->Branch("seedFrac", &bSeedFrac, "seedFrac/F");
   mTree->Branch("e2Frac", &bE2Frac, "e2Frac/F");
   mTree->Branch("e1e2Frac", &bE1e2Frac, "e1e2Frac/F");
   mTree->Branch("xw", &bXW, "xw/F");
   mTree->Branch("yw", &bYW, "yw/F");
   mTree->Branch("sigX", &bSigX, "sigX/F");
   mTree->Branch("sigY", &bSigY, "sigY/F");
   mTree->Branch("sigXY", &bSigXY, "sigXY/F");

   mTree->Branch("img", bImg, Form("img[%d]/F", kNPix));
   mTree->Branch("mask", bMask, Form("mask[%d]/F", kNPix));

   mTree->Branch("nTrk", &bNTrk, "nTrk/I");
   mTree->Branch("trkId", bTrkId, "trkId[nTrk]/I");
   mTree->Branch("trkPid", bTrkPid, "trkPid[nTrk]/I");
   mTree->Branch("trkParent", bTrkParent, "trkParent[nTrk]/I");
   mTree->Branch("trkE", bTrkE, "trkE[nTrk]/F");
   mTree->Branch("trkPtot", bTrkPtot, "trkPtot[nTrk]/F");
   mTree->Branch("truthNPhoton", &bTruthNPhoton, "truthNPhoton/I");
   mTree->Branch("truthSameParent", &bTruthSameParent, "truthSameParent/I");
   mTree->Branch("truthPurity", &bTruthPurity, "truthPurity/F");

   return kStOK;
}

//-----------------------------------------------------------------------------
Int_t StFcsClusterFeatureMaker::Finish() {
   if (!mFile) return kStOK;
   mFile->cd();
   mTree->Write();
   mFile->Close();
   LOG_INFO << "StFcsClusterFeatureMaker wrote " << mFileName << endm;
   return kStOK;
}

//-----------------------------------------------------------------------------
void StFcsClusterFeatureMaker::resetBranches() {
   for (int i = 0; i < kNPix; i++) {
      bImg[i] = 0.0;
      bMask[i] = 0.0;
   }
   bNTrk = 0;
   bTruthNPhoton = -1;
   bTruthSameParent = -1;
   bTruthPurity = -1.0;
   bChi2Ndf1 = 0.0;
   bChi2Ndf2 = 0.0;
}

//-----------------------------------------------------------------------------
Int_t StFcsClusterFeatureMaker::Make() {
   StEvent* event = (StEvent*)GetInputDS("StEvent");
   if (!event) {
      LOG_ERROR << "StFcsClusterFeatureMaker::Make did not find StEvent" << endm;
      return kStErr;
   }
   mFcsColl = event->fcsCollection();
   if (!mFcsColl) {
      LOG_ERROR << "StFcsClusterFeatureMaker::Make did not find StFcsCollection" << endm;
      return kStErr;
   }

   bRun = 0;
   bEvent = 0;
   if (event->info()) {
      bRun = event->info()->runId();
      bEvent = event->info()->id();
   }

   for (int det = 0; det < 2; det++) {  // ECal north(0) and south(1) only
      if (mFcsDb->ecalHcalPres(det) != 0) continue;

      const int nCol = mFcsDb->nColumn(det);
      const int nRow = mFcsDb->nRow(det);
      bXW = mFcsDb->getXWidth(det);
      bYW = mFcsDb->getYWidth(det);

      // full detector tower energy map for this event, so the image window can
      // include towers that clustering did not assign to this cluster
      std::vector<float> emap(nRow * nCol, 0.0);
      StSPtrVecFcsHit& hits = mFcsColl->hits(det);
      const int nh = mFcsColl->numberOfHits(det);
      for (int i = 0; i < nh; i++) {
         StFcsHit* hit = hits[i];
         if (hit->energy() < mTowerEmin) continue;
         const int row = mFcsDb->getRowNumber(det, hit->id());        // 1-based
         const int col = mFcsDb->getColumnNumber(det, hit->id());     // 1-based
         if (row < 1 || row > nRow || col < 1 || col > nCol) continue;
         emap[(row - 1) * nCol + (col - 1)] += hit->energy();
      }

      StSPtrVecFcsCluster& clusters = mFcsColl->clusters(det);
      const int nc = mFcsColl->numberOfClusters(det);
      bNCluDet = nc;
      bDet = det;

      for (int ic = 0; ic < nc; ic++) {
         StFcsCluster* clu = clusters[ic];
         if (clu->energy() < mEmin) continue;

         resetBranches();

         bClId = clu->id();
         bE = clu->energy();
         bX = clu->x();
         bY = clu->y();
         bSigmaMax = clu->sigmaMax();
         bSigmaMin = clu->sigmaMin();
         bTheta = clu->theta();
         bNTowers = clu->nTowers();
         bNNeighbor = clu->nNeighbor();
         bNPoints = clu->nPoints();
         bCatStar = clu->category();
         bChi2Ndf1 = clu->chi2Ndf1Photon();
         bChi2Ndf2 = clu->chi2Ndf2Photon();

         StThreeVectorD xyz = mFcsDb->getStarXYZfromColumnRow(det, bX, bY);
         bStarX = xyz.x();
         bStarY = xyz.y();
         bStarZ = xyz.z();
         StLorentzVectorD p4 = mFcsDb->getLorentzVector(xyz, bE, 0.0);
         bEta = p4.pseudoRapidity();
         bPhi = p4.phi();
         bPt = p4.perp();

         // ---- seed, energy ordering, second moments (cell units) ----
         StPtrVecFcsHit& chits = clu->hits();
         const int nch = (int)chits.size();
         float e1 = -1.0, e2 = -1.0;
         int seedRow = -1, seedCol = -1, seedId = -1;
         double wtot = 0, sx = 0, sy = 0, sxy = 0, mx = 0, my = 0;
         for (int k = 0; k < nch; k++) {
            StFcsHit* h = chits[k];
            const float he = h->energy();
            const int row = mFcsDb->getRowNumber(det, h->id());
            const int col = mFcsDb->getColumnNumber(det, h->id());
            if (he > e1) {
               e2 = e1;
               e1 = he;
               seedId = h->id();
               seedRow = row;
               seedCol = col;
            } else if (he > e2) {
               e2 = he;
            }
            wtot += he;
            mx += he * col;
            my += he * row;
            sx += he * col * col;
            sy += he * row * row;
            sxy += he * col * row;
         }
         if (wtot <= 0) continue;
         mx /= wtot;
         my /= wtot;
         bSigX = sqrt(fabs(sx / wtot - mx * mx));
         bSigY = sqrt(fabs(sy / wtot - my * my));
         bSigXY = sxy / wtot - mx * my;
         bSeedId = seedId;
         bSeedRow = seedRow;
         bSeedCol = seedCol;
         bSeedE = e1;
         bSeedFrac = (bE > 0) ? e1 / bE : 0;
         bE2Frac = (bE > 0 && e2 > 0) ? e2 / bE : 0;
         bE1e2Frac = (e1 > 0 && e2 > 0) ? (e1 - e2) / (e1 + e2) : 1.0;

         // ---- NxN image centred on the seed tower ----
         const int half = kNW / 2;
         for (int dr = -half; dr <= half; dr++) {
            for (int dc = -half; dc <= half; dc++) {
               const int row = seedRow + dr;
               const int col = seedCol + dc;
               const int pix = (dr + half) * kNW + (dc + half);
               if (row < 1 || row > nRow || col < 1 || col > nCol) {
                  bImg[pix] = -1.0;  // outside the detector: distinguish from an empty tower
                  continue;
               }
               bImg[pix] = emap[(row - 1) * nCol + (col - 1)];
            }
         }
         for (int k = 0; k < nch; k++) {
            StFcsHit* h = chits[k];
            const int dr = mFcsDb->getRowNumber(det, h->id()) - seedRow;
            const int dc = mFcsDb->getColumnNumber(det, h->id()) - seedCol;
            if (abs(dr) > half || abs(dc) > half) continue;
            bMask[(dr + half) * kNW + (dc + half)] = 1.0;
         }

         if (mSaveTruth) fillTruth(clu);

         mTree->Fill();
      }
   }
   return kStOK;
}

//-----------------------------------------------------------------------------
// GEANT truth. StFcsHit::getGeantTracks() is filled by StFcsFastSimulatorMaker,
// so this only does anything on simulation. Everything is guarded: on data the
// truth branches stay at their reset values.
void StFcsClusterFeatureMaker::fillTruth(StFcsCluster* clu) {
   std::map<unsigned int, float> dep;  // g2t track id -> energy deposited in this cluster
   StPtrVecFcsHit& chits = clu->hits();
   for (size_t k = 0; k < chits.size(); k++) {
      const std::vector<std::pair<unsigned int, float> >& gt = chits[k]->getGeantTracks();
      for (size_t j = 0; j < gt.size(); j++) dep[gt[j].first] += gt[j].second;
   }
   if (dep.empty()) return;

   std::vector<std::pair<float, unsigned int> > sorted;
   float sum = 0;
   for (std::map<unsigned int, float>::iterator it = dep.begin(); it != dep.end(); ++it) {
      sorted.push_back(std::make_pair(it->second, it->first));
      sum += it->second;
   }
   std::sort(sorted.rbegin(), sorted.rend());

   St_g2t_track* g2ttrk = (St_g2t_track*)GetDataSet("g2t_track");
   if (!g2ttrk) g2ttrk = (St_g2t_track*)GetDataSet("geant/g2t_track");
   g2t_track_st* trk = g2ttrk ? g2ttrk->GetTable() : 0;
   const int ntrk = g2ttrk ? g2ttrk->GetNRows() : 0;

   bNTrk = std::min((int)sorted.size(), kMaxTrk);
   int nPhoton = 0;
   int parent1 = -1, parent2 = -1;
   for (int i = 0; i < bNTrk; i++) {
      const unsigned int id = sorted[i].second;
      bTrkId[i] = (int)id;
      bTrkE[i] = sorted[i].first;
      bTrkPid[i] = -1;
      bTrkParent[i] = -1;
      bTrkPtot[i] = -1;
      if (trk) {
         for (int t = 0; t < ntrk; t++) {  // g2t id is 1-based and usually t+1, but do not assume
            if ((unsigned int)trk[t].id != id) continue;
            bTrkPid[i] = trk[t].ge_pid;
            bTrkParent[i] = trk[t].next_parent_p;
            bTrkPtot[i] = trk[t].ptot;
            break;
         }
      }
      if (bTrkPid[i] == 1 && sorted[i].first > mTruthFrac * sum) {  // GEANT3 pid 1 = gamma
         nPhoton++;
         if (parent1 < 0) {
            parent1 = bTrkParent[i];
         } else if (parent2 < 0) {
            parent2 = bTrkParent[i];
         }
      }
   }
   bTruthNPhoton = nPhoton;
   bTruthPurity = (sum > 0) ? sorted[0].first / sum : -1;
   bTruthSameParent = (parent1 > 0 && parent1 == parent2) ? 1 : 0;
}
