// class StFcsPicoFeatureMaker
// see the header for what picoDst gives and what is rebuilt here

#include "StFcsPicoFeatureMaker.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#include "StMessMgr.h"
#include "StThreeVectorF.hh"
#include "Stypes.h"
#include "TFile.h"
#include "TLorentzVector.h"
#include "TTree.h"
#include "TVector3.h"

#include "StFcsDbMaker/StFcsDb.h"
#include "StPicoDstMaker/StPicoDstMaker.h"
#include "StPicoEvent/StPicoDst.h"
#include "StPicoEvent/StPicoEvent.h"
#include "StPicoEvent/StPicoFcsCluster.h"
#include "StPicoEvent/StPicoFcsHit.h"
#include "StPicoEvent/StPicoMcTrack.h"
#include "StPicoEvent/StPicoMcVertex.h"

#include "StFcsMLCategoryMaker/StFcsTowerAssoc.h"

using namespace StFcsTowerAssoc;

#ifndef SKIPDefImp
ClassImp(StFcsPicoFeatureMaker)
#endif

    // Out-of-class definitions for the in-class-initialised statics. Without
    // them an odr-use (std::min binds by const reference) leaves an undefined
    // symbol that cons does not complain about and dlopen does - see the same
    // note in StFcsClusterFeatureMaker.cxx.
    const int StFcsPicoFeatureMaker::kNW;
const int StFcsPicoFeatureMaker::kNPix;
const int StFcsPicoFeatureMaker::kMaxMc;

StFcsPicoFeatureMaker::StFcsPicoFeatureMaker(StPicoDstMaker* picoMaker, const Char_t* name)
    : StMaker(name),
      mPicoDstMaker(picoMaker),
      mPicoDst(0),
      mFcsDb(0),
      mFileName("fcsEcalClusterFeaturesPico.root"),
      mFile(0),
      mTree(0),
      mEmin(0.5),
      mTowerEmin(0.0),
      mMaxDist(5.0),
      mUseNTow(1),
      mNeighborDist(1.01),
      mSaveMcTruth(1),
      mMcMatchR(11.0),
      mMcEmin(0.0),
      mHaveMc(0),
      mNEvents(0),
      mNCluster(0),
      mNNoMcArray(0) {}

StFcsPicoFeatureMaker::~StFcsPicoFeatureMaker() {}

//-----------------------------------------------------------------------------
Int_t StFcsPicoFeatureMaker::Init() {
   if (!mPicoDstMaker) {
      LOG_ERROR << "StFcsPicoFeatureMaker::Init no StPicoDstMaker given" << endm;
      return kStFatal;
   }

   // StFcsDb is used for geometry ONLY: id -> row/column, the cell widths, and
   // the ECal plane for projecting the generated photons. None of that needs a
   // database - run the StFcsDbMaker with setDbAccess(0) and StFcsDb falls back
   // to its built-in tower map and detector positions. That is why this chain
   // does not need St_db_Maker, which is just as well: a simulated picoDst
   // carries run number 1, for which no FCS calibration exists.
   mFcsDb = static_cast<StFcsDb*>(GetDataSet("fcsDb"));
   if (!mFcsDb) {
      LOG_ERROR << "StFcsPicoFeatureMaker::Init failed to get StFcsDb - add an "
                << "StFcsDbMaker to the chain (setDbAccess(0) is enough)" << endm;
      return kStFatal;
   }
   if (mFcsDb->nColumn(0) <= 0 || mFcsDb->nRow(0) <= 0) {
      LOG_ERROR << "StFcsPicoFeatureMaker::Init StFcsDb reports a "
                << mFcsDb->nRow(0) << " x " << mFcsDb->nColumn(0)
                << " ECal - the geometry is not initialised" << endm;
      return kStFatal;
   }
   LOG_INFO << Form("StFcsPicoFeatureMaker: ECal %d rows x %d columns, cell %.3f x %.3f cm; "
                    "tower association maxDist=%.1f cells, truncate to stored nTowers=%d",
                    mFcsDb->nRow(0), mFcsDb->nColumn(0), mFcsDb->getXWidth(0), mFcsDb->getYWidth(0),
                    mMaxDist, mUseNTow)
            << endm;

   mFile = new TFile(mFileName.c_str(), "RECREATE");
   mTree = new TTree("clusters", "FCS ECal cluster features from picoDst");

   // ---- the schema below must stay identical to StFcsClusterFeatureMaker ----
   mTree->Branch("run", &bRun, "run/I");
   mTree->Branch("event", &bEvent, "event/I");
   mTree->Branch("det", &bDet, "det/I");
   mTree->Branch("clid", &bClId, "clid/I");
   mTree->Branch("ncluDet", &bNCluDet, "ncluDet/I");

   mTree->Branch("e", &bE, "e/F");
   mTree->Branch("x", &bX, "x/F");
   mTree->Branch("y", &bY, "y/F");
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

   // Hit-level truth does not exist on picoDst. The branches are written anyway,
   // at their "no truth" values, so that the two trees have one schema and
   // trainTMVA.C does not have to care which produced the file.
   mTree->Branch("nTrk", &bNTrk, "nTrk/I");
   mTree->Branch("truthNPhoton", &bTruthNPhoton, "truthNPhoton/I");
   mTree->Branch("truthSameParent", &bTruthSameParent, "truthSameParent/I");
   mTree->Branch("truthPurity", &bTruthPurity, "truthPurity/F");

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
   mTree->Branch("genPid", &bGenPid, "genPid/I");
   mTree->Branch("genE", &bGenE, "genE/F");
   mTree->Branch("nGen", &bNGen, "nGen/I");

   // picoDst-only QA of the association
   mTree->Branch("vz", &bVz, "vz/F");
   mTree->Branch("nTowRec", &bNTowRec, "nTowRec/I");
   mTree->Branch("eRec", &bERec, "eRec/F");
   mTree->Branch("dxRec", &bDxRec, "dxRec/F");
   mTree->Branch("dyRec", &bDyRec, "dyRec/F");

   return kStOK;
}

//-----------------------------------------------------------------------------
Int_t StFcsPicoFeatureMaker::Finish() {
   LOG_INFO << Form("StFcsPicoFeatureMaker: %lld events, %lld ECal clusters written", mNEvents,
                    mNCluster)
            << endm;
   if (mNNoMcArray > 0)
      LOG_WARN << Form("StFcsPicoFeatureMaker: %lld events had no StPicoMcTrack array, so mcLabel "
                       "stayed at -1 there. Real data looks like this; a simulated picoDst "
                       "produced without the MC arrays does too, and then there is nothing to "
                       "train on.",
                       mNNoMcArray)
               << endm;
   if (!mFile) return kStOK;
   mFile->cd();
   mTree->Write();
   mFile->Close();
   LOG_INFO << "StFcsPicoFeatureMaker wrote " << mFileName << endm;
   return kStOK;
}

//-----------------------------------------------------------------------------
void StFcsPicoFeatureMaker::resetBranches() {
   for (int i = 0; i < kNPix; i++) {
      bImg[i] = 0.0;
      bMask[i] = 0.0;
   }
   bNTrk = 0;
   bTruthNPhoton = -1;
   bTruthSameParent = -1;
   bTruthPurity = -1.0;
   bNNeighbor = -1;  // recomputed below from the recovered tower adjacency
   bNPoints = -1;    // not stored in picoDst, and nothing here can rebuild it
   bMcLabel = -1;
   bNMcPhoton = 0;
   bMcParentPid = 0;
   bMcSep = -1.0;
   bMcSepCell = -1.0;
   bMcZgg = -1.0;
   bSeedId = -1;
   bSeedRow = -1;
   bSeedCol = -1;
   bSeedE = 0.0;
   bSeedFrac = 0.0;
   bE2Frac = 0.0;
   bE1e2Frac = 1.0;
   bSigX = 0.0;
   bSigY = 0.0;
   bSigXY = 0.0;
   bNTowRec = 0;
   bERec = 0.0;
   bDxRec = 0.0;
   bDyRec = 0.0;
}

//-----------------------------------------------------------------------------
// Project one generated photon onto both ECal halves, at SHOWER MAX depth
// rather than the front face - that is the plane the cluster centroid lives on,
// so a projected photon and a cluster position are then directly comparable.
void StFcsPicoFeatureMaker::projectPhoton(McPhoton& ph) {
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
      if (fabs(nd) < 1e-9) continue;
      const double t = nrm.dot(p0 - v) / nd;
      if (t <= 0) continue;
      const StThreeVectorD hit = v + dir * t;
      ph.proj[det][0] = hit.x();
      ph.proj[det][1] = hit.y();
      ph.projOk[det] = 1;
      reaches = 1;
   }
   if (reaches) bNMcPhotonEvent++;
}

//-----------------------------------------------------------------------------
// Generator-level photons of this event, from the picoDst MC arrays.
//
// StPicoMcTrack is written from the same g2t_track table as StMuMcTrack, so the
// selection is the same one StFcsMuMcTruthMaker uses on the MuDst side: GEANT
// pid 1 (gamma), not a shower product. The pi0 of a pi0 gun decays in GEANT, so
// its daughters ARE tracks here; isFromShower() is what removes the electron
// cascade inside the calorimeter.
//
// StPicoMcTrack has no parent pointer, only the id of its start vertex, so the
// parent is found the same way: the track that ENDS at that vertex.
// The generated ("gun") particle of the event.
//
// GEANT numbers the generator's particles first, so track id 1 is always one
// of them, and the primaries are the tracks that start at the same vertex as
// track 1. In a single-particle gun there is exactly one - checked on
// pi0.e30.vz0.run6: all 100 events have one primary, the pi0 (GEANT pid 7),
// at vertex 1, with its two photons starting at vertex 2. If there are several
// primaries (a multi-particle generator), the most energetic one is recorded
// and nGen says how many there were.
void StFcsPicoFeatureMaker::fillGenerated() {
   bGenPid = 0;
   bGenE = 0;
   bNGen = 0;
   const int ntrk = (int)mPicoDst->numberOfMcTracks();
   if (ntrk <= 0) return;

   StPicoMcTrack* first = 0;
   for (int i = 0; i < ntrk; i++) {
      StPicoMcTrack* t = mPicoDst->mcTrack(i);
      if (t && t->id() == 1) {
         first = t;
         break;
      }
   }
   if (!first) first = mPicoDst->mcTrack(0);
   if (!first) return;

   const int vtx0 = first->idVtxStart();
   for (int i = 0; i < ntrk; i++) {
      StPicoMcTrack* t = mPicoDst->mcTrack(i);
      if (!t || t->idVtxStart() != vtx0) continue;
      bNGen++;
      if (t->energy() > bGenE) {
         bGenE = t->energy();
         bGenPid = t->geantId();
      }
   }
}

//-----------------------------------------------------------------------------
void StFcsPicoFeatureMaker::collectMcPhotons() {
   mMcPhotons.clear();
   bNMcPhotonEvent = 0;
   mHaveMc = 0;
   if (!mSaveMcTruth) return;

   const int ntrk = (int)mPicoDst->numberOfMcTracks();
   const int nvtx = (int)mPicoDst->numberOfMcVertices();
   if (ntrk <= 0 || nvtx <= 0) {
      mNNoMcArray++;
      return;
   }
   // The MC record exists for this event, whether or not it contains a
   // generated photon. That distinction is what separates mcLabel = 0 (truth
   // says: no photon here) from mcLabel = -1 (no truth at all).
   mHaveMc = 1;

   for (int i = 0; i < ntrk; i++) {
      StPicoMcTrack* t = mPicoDst->mcTrack(i);
      if (!t) continue;
      if (t->geantId() != 1) continue;       // GEANT3 pid 1 = gamma
      if (t->isFromShower()) continue;       // drop electromagnetic cascade products
      if (t->energy() <= mMcEmin) continue;

      const int idVx = t->idVtxStart();
      float vx = 0, vy = 0, vz = 0;
      int found = 0;
      for (int j = 0; j < nvtx; j++) {
         StPicoMcVertex* v = mPicoDst->mcVertex(j);
         if (!v || v->id() != idVx) continue;
         const TVector3 pos = v->position();
         vx = pos.X();
         vy = pos.Y();
         vz = pos.Z();
         found = 1;
         break;
      }
      if (!found) continue;

      int parentId = 0, parentPid = 0;
      for (int j = 0; j < ntrk; j++) {
         StPicoMcTrack* p = mPicoDst->mcTrack(j);
         if (!p || p->id() == t->id()) continue;
         if (p->idVtxStop() != idVx) continue;
         parentId = p->id();
         parentPid = p->geantId();
         break;
      }

      McPhoton ph;
      ph.id = t->id();
      ph.parent = parentId;
      ph.parentPid = parentPid;
      ph.e = t->energy();
      const TVector3 p3 = t->p();
      ph.p[0] = p3.X();
      ph.p[1] = p3.Y();
      ph.p[2] = p3.Z();
      ph.v[0] = vx;
      ph.v[1] = vy;
      ph.v[2] = vz;
      projectPhoton(ph);
      mMcPhotons.push_back(ph);
   }
}

//-----------------------------------------------------------------------------
// Match the collected photons to one cluster. Same definition of mcLabel as on
// the MuDst side: how many GENERATED photons project within mMcMatchR of the
// cluster, capped at 2.
void StFcsPicoFeatureMaker::matchMcPhotons(float cluX, float cluY, int det) {
   // NO early return on an empty photon list. An event with MC truth but no
   // generated photon - every event of a pi- gun, for instance - is exactly
   // where class 0 comes from, and it must come out as mcLabel = 0. Returning
   // here used to leave those clusters at -1, i.e. "no truth", and trainTMVA.C
   // then threw the whole hadron sample away and trained class 0 on scraps.
   if (!mSaveMcTruth || !mHaveMc) return;

   const StThreeVectorD cpos = mFcsDb->getStarXYZfromColumnRow(det, cluX, cluY);

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

   bMcLabel = (int)near.size();
   if (bMcLabel > 2) bMcLabel = 2;
}

//-----------------------------------------------------------------------------
Int_t StFcsPicoFeatureMaker::Make() {
   mPicoDst = mPicoDstMaker->picoDst();
   if (!mPicoDst) {
      LOG_ERROR << "StFcsPicoFeatureMaker::Make no StPicoDst" << endm;
      return kStErr;
   }
   StPicoEvent* ev = mPicoDst->event();
   if (!ev) return kStErr;

   mNEvents++;
   bRun = ev->runId();
   bEvent = ev->eventId();
   bVz = ev->primaryVertex().z();

   collectMcPhotons();
   fillGenerated();

   const int nHitAll = (int)mPicoDst->numberOfFcsHits();
   const int nCluAll = (int)mPicoDst->numberOfFcsClusters();

   for (int det = 0; det < 2; det++) {  // ECal north(0) and south(1)
      const int nCol = mFcsDb->nColumn(det);
      const int nRow = mFcsDb->nRow(det);
      bXW = mFcsDb->getXWidth(det);
      bYW = mFcsDb->getYWidth(det);
      bDet = det;

      // ---- towers of this half ----
      std::vector<Tower> tow;
      std::vector<float> emap(nRow * nCol, 0.0);
      for (int i = 0; i < nHitAll; i++) {
         StPicoFcsHit* h = mPicoDst->fcsHit(i);
         if (!h || (int)h->detectorId() != det) continue;
         if (h->energy() < mTowerEmin) continue;
         const int row = mFcsDb->getRowNumber(det, h->id());     // 1-based
         const int col = mFcsDb->getColumnNumber(det, h->id());  // 1-based
         if (row < 1 || row > nRow || col < 1 || col > nCol) continue;
         Tower t;
         t.id = h->id();
         t.row = row;
         t.col = col;
         t.e = h->energy();
         tow.push_back(t);
         emap[(row - 1) * nCol + (col - 1)] += h->energy();
      }

      // ---- clusters of this half ----
      std::vector<int> cluIdx;
      std::vector<float> cluX, cluY;
      std::vector<int> cluNTow;
      for (int i = 0; i < nCluAll; i++) {
         StPicoFcsCluster* c = mPicoDst->fcsCluster(i);
         if (!c || (int)c->detectorId() != det) continue;
         cluIdx.push_back(i);
         cluX.push_back(c->x());
         cluY.push_back(c->y());
         cluNTow.push_back(mUseNTow ? c->nTowers() : 0);
      }
      const int nc = (int)cluIdx.size();
      bNCluDet = nc;
      if (nc == 0) continue;

      // ---- recover which tower belongs to which cluster ----
      std::vector<int> owner;
      assign(tow, nc, &cluX[0], &cluY[0], &cluNTow[0], mMaxDist, owner);

      // Neighbour clusters. StPicoFcsCluster does not store nNeighbor, and
      // leaving it at a constant -1 is not harmless: TMVA refuses a constant
      // input variable outright ("Variable nNeighbor is constant. Please remove
      // the variable." followed by abort), which takes feature set 13 out with
      // it. The adjacency it is built from survives into picoDst, so it is
      // recomputed here - see the long note in StFcsTowerAssoc.h for what the
      // number means and how it differs from StFcsCluster::nNeighbor().
      std::vector<int> nNbr;
      neighborCounts(tow, owner, nc, nRow, nCol, mNeighborDist, nNbr);

      std::vector<Tower> mine;
      for (int ic = 0; ic < nc; ic++) {
         StPicoFcsCluster* clu = mPicoDst->fcsCluster(cluIdx[ic]);
         if (!clu) continue;
         if (clu->energy() < mEmin) continue;

         resetBranches();

         // everything picoDst stores exactly
         bClId = clu->id();
         bE = clu->energy();
         bX = clu->x();
         bY = clu->y();
         bSigmaMax = clu->sigmaMax();
         bSigmaMin = clu->sigmaMin();
         bTheta = clu->theta();
         bNTowers = clu->nTowers();
         bNNeighbor = (ic < (int)nNbr.size()) ? nNbr[ic] : -1;
         bCatStar = clu->category();
         bChi2Ndf1 = clu->chi2Ndf1Photon();
         bChi2Ndf2 = clu->chi2Ndf2Photon();

         const StThreeVectorD xyz = mFcsDb->getStarXYZfromColumnRow(det, bX, bY);
         bStarX = xyz.x();
         bStarY = xyz.y();
         bStarZ = xyz.z();
         const TLorentzVector p4 = clu->fourMomentum();
         bPt = p4.Pt();
         bEta = p4.Eta();
         bPhi = p4.Phi();

         // ---- everything that needs the tower list ----
         gather(tow, owner, ic, mine);
         bNTowRec = (int)mine.size();
         if (bNTowRec > 0) {
            float e1 = -1.0, e2 = -1.0;
            int seedRow = -1, seedCol = -1, seedId = -1;
            double wtot = 0, sx = 0, sy = 0, sxy = 0, mx = 0, my = 0;
            for (int k = 0; k < bNTowRec; k++) {
               const float he = mine[k].e;
               if (he > e1) {
                  e2 = e1;
                  e1 = he;
                  seedId = mine[k].id;
                  seedRow = mine[k].row;
                  seedCol = mine[k].col;
               } else if (he > e2) {
                  e2 = he;
               }
               wtot += he;
               mx += he * mine[k].col;
               my += he * mine[k].row;
               sx += he * mine[k].col * mine[k].col;
               sy += he * mine[k].row * mine[k].row;
               sxy += he * mine[k].col * mine[k].row;
            }
            if (wtot > 0) {
               mx /= wtot;
               my /= wtot;
               bERec = wtot;
               // The recovered centroid is in cell CENTRES (col-0.5), while the
               // stored x,y are in cell units; the -0.5 is what makes the two
               // directly comparable, and dxRec/dyRec should scatter around 0.
               bDxRec = (mx - 0.5) - bX;
               bDyRec = (my - 0.5) - bY;
               bSigX = sqrt(fabs(sx / wtot - mx * mx));
               bSigY = sqrt(fabs(sy / wtot - my * my));
               bSigXY = sxy / wtot - mx * my;
               bSeedId = seedId;
               bSeedRow = seedRow;
               bSeedCol = seedCol;
               bSeedE = e1;
               // fractions are taken against the STORED cluster energy, which is
               // exact, not against the recovered sum
               bSeedFrac = (bE > 0) ? e1 / bE : 0;
               bE2Frac = (bE > 0 && e2 > 0) ? e2 / bE : 0;
               bE1e2Frac = (e1 > 0 && e2 > 0) ? (e1 - e2) / (e1 + e2) : 1.0;

               // 11x11 image around the seed, from the full tower map of the
               // half - so it contains the neighbours too, mask says which
               // towers this cluster owns
               const int half = kNW / 2;
               for (int dr = -half; dr <= half; dr++) {
                  for (int dc = -half; dc <= half; dc++) {
                     const int row = seedRow + dr;
                     const int col = seedCol + dc;
                     const int pix = (dr + half) * kNW + (dc + half);
                     if (row < 1 || row > nRow || col < 1 || col > nCol) {
                        bImg[pix] = -1.0;  // outside the detector
                        continue;
                     }
                     bImg[pix] = emap[(row - 1) * nCol + (col - 1)];
                  }
               }
               for (int k = 0; k < bNTowRec; k++) {
                  const int dr = mine[k].row - seedRow;
                  const int dc = mine[k].col - seedCol;
                  if (abs(dr) > half || abs(dc) > half) continue;
                  bMask[(dr + half) * kNW + (dc + half)] = 1.0;
               }
            }
         }

         matchMcPhotons(bX, bY, det);

         mNCluster++;
         mTree->Fill();
      }
   }

   return kStOK;
}
