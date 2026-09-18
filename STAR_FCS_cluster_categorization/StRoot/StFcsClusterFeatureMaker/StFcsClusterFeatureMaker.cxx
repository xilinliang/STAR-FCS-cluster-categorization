// class StFcsClusterFeatureMaker
// see header for chain position and purpose

#include "StFcsClusterFeatureMaker.h"

#include <algorithm>
#include <cmath>
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
#include "tables/St_g2t_vertex_Table.h"

#ifndef SKIPDefImp
ClassImp(StFcsClusterFeatureMaker)
#endif

    // Out-of-class definitions for the static constants declared in the header.
    // An in-class initialiser is a declaration, not a definition: the moment one
    // of these is odr-used - std::min binds its arguments BY REFERENCE, so it is
    // - the linker wants a symbol, and .so loading fails with
    //   undefined symbol: _ZN24StFcsClusterFeatureMaker7kMaxTrkE
    // cons links the library without complaint, so this only shows up at
    // dlopen time in the macro. (C++17 would make these implicitly inline;
    // gcc 4.8 with -std=c++0x does not.)
    const int StFcsClusterFeatureMaker::kNW;
const int StFcsClusterFeatureMaker::kNPix;
const int StFcsClusterFeatureMaker::kMaxTrk;
const int StFcsClusterFeatureMaker::kMaxMc;

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

   // generator-level ("particle level") truth
   mTree->Branch("mcLabel", &bMcLabel, "mcLabel/I");
   mTree->Branch("nMcPhoton", &bNMcPhoton, "nMcPhoton/I");
   mTree->Branch("mcTrkId", bMcTrkId, "mcTrkId[nMcPhoton]/I");
   mTree->Branch("mcParent", bMcParent, "mcParent[nMcPhoton]/I");
   mTree->Branch("mcParentPid", &bMcParentPid, "mcParentPid/I");
   mTree->Branch("mcE", bMcE, "mcE[nMcPhoton]/F");
   mTree->Branch("mcDr", bMcDr, "mcDr[nMcPhoton]/F");
   mTree->Branch("mcX", bMcX, "mcX[nMcPhoton]/F");
   mTree->Branch("mcY", bMcY, "mcY[nMcPhoton]/F");
   mTree->Branch("mcSep", &bMcSep, "mcSep/F");
   mTree->Branch("mcSepCell", &bMcSepCell, "mcSepCell/F");
   mTree->Branch("mcZgg", &bMcZgg, "mcZgg/F");
   mTree->Branch("nMcPhotonEvent", &bNMcPhotonEvent, "nMcPhotonEvent/I");

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
   bMcLabel = -1;
   bNMcPhoton = 0;
   bMcParentPid = 0;
   bMcSep = -1.0;
   bMcSepCell = -1.0;
   bMcZgg = -1.0;
}

//-----------------------------------------------------------------------------
// Generator-level truth, step 1 of 2: collect the generated photons of the
// event and project each onto both ECal planes.
//
// "Generated" here means a GEANT track that is not a shower product: the
// photons from a pi0 gun are GEANT tracks (the pi0 decays in GEANT), so
// selecting eg_label > 0 alone would miss them. is_shower == 0 keeps the pi0
// daughters while rejecting the e+/e- of the electromagnetic cascade.
//
// The projection is a ray-plane intersection using StFcsDb's own description of
// each ECal half - getDetectorOffset() for a point on it, getNormal() for its
// orientation - so the detector tilt is handled rather than assumed away.
// Project one photon onto both ECal halves. The plane of each half comes from
// StFcsDb - getDetectorOffset() for a point on it, getNormal() for its
// orientation - so the tilt is taken from the database rather than assumed.
//
// The plane is taken at SHOWER MAX, not at the front face. StFcsDb puts the
// cluster centroid there too (getStarXYZfromColumnRow defaults FcsZ to
// getShowerMaxZ), so a projected photon and a cluster position are then
// measured on the same plane. Projecting to the front face instead shifts a
// photon inward by about depth * r / z - a centimetre or two at FCS radii - and
// shrinks mcSep by the same small factor. StFcsPicoFeatureMaker does the same,
// so the two feature files stay comparable.
void StFcsClusterFeatureMaker::projectPhoton(McPhoton& ph) {
   const StThreeVectorD v(ph.v[0], ph.v[1], ph.v[2]);
   const StThreeVectorD dir(ph.p[0], ph.p[1], ph.p[2]);
   int reaches = 0;
   for (int det = 0; det < 2; det++) {
      ph.projOk[det] = 0;
      ph.proj[det][0] = -9999.0;
      ph.proj[det][1] = -9999.0;
      const StThreeVectorD p0 = mFcsDb->getDetectorOffset(det, mFcsDb->getShowerMaxZ(det));
      const StThreeVectorD nrm = mFcsDb->getNormal(det);
      const double nd = nrm.dot(dir);
      if (fabs(nd) < 1e-9) continue;  // parallel to the plane
      const double t = nrm.dot(p0 - v) / nd;
      if (t <= 0) continue;           // plane is behind the photon
      const StThreeVectorD hit = v + dir * t;
      ph.proj[det][0] = hit.x();
      ph.proj[det][1] = hit.y();
      ph.projOk[det] = 1;
      reaches = 1;
   }
   if (reaches) bNMcPhotonEvent++;
}

//-----------------------------------------------------------------------------
// Injection API, used by StFcsMuMcTruthMaker on the MuDst path. Call
// clearMcPhotons() once per event, then addMcPhoton() per generated photon,
// before this maker's Make() runs.
void StFcsClusterFeatureMaker::clearMcPhotons() {
   mMcPhotons.clear();
   bNMcPhotonEvent = 0;
   mMcExternal = 1;
}

void StFcsClusterFeatureMaker::addMcPhoton(int id, int parent, int parentPid, float e,
                                           float px, float py, float pz,
                                           float vx, float vy, float vz) {
   if (!mFcsDb) return;
   McPhoton ph;
   ph.id = id;
   ph.parent = parent;
   ph.parentPid = parentPid;
   ph.e = e;
   ph.p[0] = px;
   ph.p[1] = py;
   ph.p[2] = pz;
   ph.v[0] = vx;
   ph.v[1] = vy;
   ph.v[2] = vz;
   projectPhoton(ph);
   mMcPhotons.push_back(ph);
   mMcExternal = 1;
}

//-----------------------------------------------------------------------------
void StFcsClusterFeatureMaker::collectMcPhotons() {
   if (mMcExternal) return;  // already supplied for this event, e.g. from MuDst
   mMcPhotons.clear();
   bNMcPhotonEvent = 0;
   if (!mSaveMcTruth) return;

   St_g2t_track* trkTable = (St_g2t_track*)GetDataSet("g2t_track");
   if (!trkTable) trkTable = (St_g2t_track*)GetDataSet("geant/g2t_track");
   St_g2t_vertex* vtxTable = (St_g2t_vertex*)GetDataSet("g2t_vertex");
   if (!vtxTable) vtxTable = (St_g2t_vertex*)GetDataSet("geant/g2t_vertex");
   if (!trkTable || !vtxTable) return;  // not a simulation chain: stays empty

   g2t_track_st* trk = trkTable->GetTable();
   g2t_vertex_st* vtx = vtxTable->GetTable();
   const int ntrk = trkTable->GetNRows();
   const int nvtx = vtxTable->GetNRows();
   if (!trk || !vtx) return;

   for (int i = 0; i < ntrk; i++) {
      if (trk[i].ge_pid != 1) continue;   // GEANT3 pid 1 = gamma
      if (trk[i].is_shower != 0) continue;  // drop cascade products
      if (trk[i].e <= 0) continue;

      // start vertex
      const int ivp = trk[i].start_vertex_p;
      int iv = -1;
      for (int v = 0; v < nvtx; v++) {
         if (vtx[v].id == ivp) {
            iv = v;
            break;
         }
      }
      if (iv < 0) continue;

      McPhoton ph;
      ph.id = trk[i].id;
      ph.parent = trk[i].next_parent_p;
      ph.parentPid = 0;
      for (int j = 0; j < ntrk; j++) {
         if (trk[j].id == ph.parent) {
            ph.parentPid = trk[j].ge_pid;
            break;
         }
      }
      ph.e = trk[i].e;
      for (int k = 0; k < 3; k++) {
         ph.p[k] = trk[i].p[k];
         ph.v[k] = vtx[iv].ge_x[k];
      }

      projectPhoton(ph);
      mMcPhotons.push_back(ph);
   }
}

//-----------------------------------------------------------------------------
// Generator-level truth, step 2 of 2: match the collected photons to one
// cluster and fill the per-cluster branches.
//
// mcLabel is what to train on: it says how many GENERATED photons point at this
// cluster, with no dependence on how GEANT shared the energy deposits out among
// tracks. For a pi0 gun, mcSep / mcSepCell is the variable that maps the merge
// transition - the separation at which two photons stop making two clusters.
void StFcsClusterFeatureMaker::matchMcPhotons(StFcsCluster* clu, int det) {
   if (!mSaveMcTruth || mMcPhotons.empty()) return;

   const StThreeVectorD cpos = mFcsDb->getStarXYZfromColumnRow(det, clu->x(), clu->y());

   // distance-ordered list of photons projecting within mMcMatchR
   std::vector<std::pair<float, int> > near;
   for (size_t i = 0; i < mMcPhotons.size(); i++) {
      if (!mMcPhotons[i].projOk[det]) continue;
      const double dx = mMcPhotons[i].proj[det][0] - cpos.x();
      const double dy = mMcPhotons[i].proj[det][1] - cpos.y();
      const double dr = sqrt(dx * dx + dy * dy);
      if (dr > mMcMatchR) continue;
      near.push_back(std::make_pair((float)dr, (int)i));
   }
   std::sort(near.begin(), near.end());

   bNMcPhoton = std::min((int)near.size(), (int)kMaxMc);
   for (int i = 0; i < bNMcPhoton; i++) {
      const McPhoton& ph = mMcPhotons[near[i].second];
      bMcTrkId[i] = ph.id;
      bMcParent[i] = ph.parent;
      bMcE[i] = ph.e;
      bMcDr[i] = near[i].first;
      bMcX[i] = ph.proj[det][0];
      bMcY[i] = ph.proj[det][1];
   }
   if (bNMcPhoton > 0) bMcParentPid = mMcPhotons[near[0].second].parentPid;

   if ((int)near.size() >= 2) {
      const McPhoton& a = mMcPhotons[near[0].second];
      const McPhoton& b = mMcPhotons[near[1].second];
      const double dx = a.proj[det][0] - b.proj[det][0];
      const double dy = a.proj[det][1] - b.proj[det][1];
      bMcSep = sqrt(dx * dx + dy * dy);
      const float xw = mFcsDb->getXWidth(det);
      bMcSepCell = (xw > 0) ? bMcSep / xw : -1.0;
      bMcZgg = (a.e + b.e > 0) ? fabs(a.e - b.e) / (a.e + b.e) : -1.0;
   }

   // 0 = no generated photon points here (hadronic, or junk)
   // 1 = one photon      -> single-photon cluster
   // 2 = two or more     -> merged
   bMcLabel = (int)near.size();
   if (bMcLabel > 2) bMcLabel = 2;
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

   // generator-level photons, once per event, before the cluster loop
   collectMcPhotons();

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
         matchMcPhotons(clu, det);

         mTree->Fill();
      }
   }

   // Photons injected from outside are good for this event only. Clearing the
   // flag means an event where the upstream maker found none falls back to the
   // g2t tables rather than silently reusing the previous event's photons.
   mMcExternal = 0;
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
