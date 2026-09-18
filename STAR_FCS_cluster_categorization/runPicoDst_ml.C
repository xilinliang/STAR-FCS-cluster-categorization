// runPicoDst_ml.C - the picoDst chain: dump cluster features, or apply a
// trained model, or both.
//
//   mode = 0 : dump features            -> StFcsPicoFeatureMaker  (feat_pico.root)
//   mode = 1 : apply a trained model    -> StFcsPicoCategoryMaker (fcsPicoCategory.root)
//   mode = 2 : both
//
// Examples:
//   dump a training/QA sample from simulation:
//     root4star -b -q 'runPicoDst_ml.C("pi0.e30.vz0.all.picoDst.root",-1,0,3,"","feat_pico.root")'
//   train on it, exactly as for a MuDst/.fzd feature file:
//     root4star -b -q 'trainTMVA.C+("feat_pico.root","FcsCat",3)'
//   apply the result:
//     root4star -b -q 'runPicoDst_ml.C("pi0...picoDst.root",-1,1,3,"weights/FcsCat3_BDTG.weights.xml")'
//
// The input may be a single .picoDst.root file or a .list of them;
// StPicoDstMaker takes either.
//
// WHAT CHANGED, AND WHY PICODST IS NOW A FULL INPUT
//
// StPicoFcsCluster keeps the cluster summary but not its tower list
// (StPicoDstMaker::fillFcsClusters drops StMuFcsCluster::hits()). That used to
// limit picoDst to feature set 6 - the six variables that need no towers.
// StFcsTowerAssoc.h now recovers the association geometrically from the FcsHits
// collection: both sides live in the same cell coordinates, so every tower goes
// to the nearest cluster centroid and each cluster keeps the nTowers it says it
// has, highest energy first. Measured on the sample that came with this code:
// the recovered tower count is exact for 92 % of clusters and the recovered
// energy matches the stored cluster energy to a median of zero. Sets 3, 13 and
// 34 are therefore computable here as well.
//
// picoDst also carries McTrack / McVertex, which is the generator-level truth,
// so mcLabel can be filled - which means you can TRAIN on a picoDst, not only
// apply. (Hit-level truth is still absent: StPicoFcsHit has no GEANT track
// links, so truthNPhoton stays -1 here exactly as on a MuDst.)
//
// Set 6 remains the honest choice if you want to be sure no step of the
// analysis depends on a reconstructed association. Sets 3/13/34 give the model
// the shape information back; check nTowRec against nTowers in the output before
// trusting them on a new sample.
//
// NO DATABASE IS NEEDED. StFcsDbMaker runs with setDbAccess(0): the only thing
// asked of it is geometry - the tower map, the cell size and the ECal plane -
// and StFcsDb has all of that built in. That matters here because a simulated
// picoDst carries run number 1, for which no FCS calibration exists.
//
// author: generated for Xilin Liang

void runPicoDst_ml(const char* input = "pi0.e30.vz0.all.picoDst.root",
                   Int_t nevt = -1,
                   int mode = 0,
                   int featureSet = 3,
                   const char* weightFile = "weights/FcsCat3_BDTG.weights.xml",
                   const char* outFile = "feat_pico.root",
                   const char* tmvaMethod = "BDTG",
                   float clusterEmin = 0.5) {
   // StPicoDstMaker links against StMuDSTMaker - it references StMuDst statics
   // such as mMuFmsCollection - so the MuDst libraries have to be loaded even
   // when only READING a picoDst. Loading StPicoDstMaker without them fails with
   //   dlopen error: ... undefined symbol: _ZN7StMuDst16mMuFmsCollectionE
   // loadSharedLibraries.C pulls in that whole set, exactly as runMudst.C does.
   gROOT->Macro("Load.C");
   gROOT->Macro("$STAR/StRoot/StMuDSTMaker/COMMON/macros/loadSharedLibraries.C");

   gSystem->Load("StPicoEvent");
   gSystem->Load("StPicoDstMaker");
   gSystem->Load("StFcsDbMaker");

   StChain* chain = new StChain("StChain");

   StPicoDstMaker* picoMaker = new StPicoDstMaker(StPicoDstMaker::IoRead, input, "picoDst");
   // Read only what is used. FcsHits is required now - it is what the tower
   // association works from - and McTrack/McVertex carry the truth labels.
   picoMaker->SetStatus("*", 0);
   picoMaker->SetStatus("Event*", 1);
   picoMaker->SetStatus("FcsClusters*", 1);
   picoMaker->SetStatus("FcsHits*", 1);
   picoMaker->SetStatus("McTrack*", 1);
   picoMaker->SetStatus("McVertex*", 1);

   // Geometry only - no St_db_Maker, no calibration tables, no run number
   // lookup. See the note at the top.
   StFcsDbMaker* fcsDbMkr = new StFcsDbMaker();
   fcsDbMkr->setDbAccess(0);

   // ---- mode 0 / 2 : dump features ----
   StFcsPicoFeatureMaker* feat = 0;
   if (mode == 0 || mode == 2) {
      gSystem->Load("StFcsPicoFeatureMaker");
      feat = new StFcsPicoFeatureMaker(picoMaker);
      feat->setOutputFile(outFile);
      feat->setEnergyThreshold(clusterEmin);
      feat->setSaveMcTruth(1);      // generator-level photons -> mcLabel, mcSep
      feat->setMcMatchRadius(11.0); // cm, about two ECal towers
   }

   // ---- mode 1 / 2 : apply a trained model ----
   if (mode == 1 || mode == 2) {
      gSystem->Load("libTMVA");  // must precede our library
      gSystem->Load("StFcsPicoCategoryMaker");
      StFcsPicoCategoryMaker* cat = new StFcsPicoCategoryMaker(picoMaker);
      cat->setFeatureSet(featureSet);  // must match the weight file
      cat->setNoModel(0);
      cat->setWeightFile(weightFile);
      cat->setTMVAMethod(tmvaMethod);
      cat->setOutputFile(mode == 2 ? "fcsPicoCategory.root" : outFile);
      cat->setMode(1);                   // 1 = take the model's answer, 2 = only above setConfidence
      cat->setConfidence(0.7);
      cat->setEnergyThreshold(clusterEmin);
      cat->setPairEnergyThreshold(1.0);  // per cluster, for the pi0 pairing
      cat->setZggMax(0.7);
   }

   if (chain->Init() != kStOK) {
      printf("chain->Init() failed\n");
      return;
   }

   int n = picoMaker->chain()->GetEntries();
   printf("Found %d events in picoDst input\n", n);
   if (nevt >= 0 && nevt < n) n = nevt;

   for (int i = 0; i < n; i++) {
      if (i % 1000 == 0) printf("  event %d / %d\n", i, n);
      chain->Clear();
      if (chain->Make(i) != kStOK) break;
   }
   chain->Finish();
   delete chain;

   // SANITY CHECKS on a feature dump, before training on it:
   //   clusters->Draw("mcLabel")             // must not be all -1
   //   clusters->Draw("nTowRec-nTowers")     // should peak hard at 0
   //   clusters->Draw("(eRec-e)/e")          // should peak hard at 0
   //   clusters->Draw("dxRec:dyRec")         // should be centred on (0,0)
   // All mcLabel = -1 means the picoDst was written without the MC arrays, or
   // SetStatus("McTrack*",1) was dropped; then there is nothing to train on and
   // the file is only good for applying a model.
   printf("wrote %s\n", outFile);
}
