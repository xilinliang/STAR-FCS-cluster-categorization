// runMudst_ml.C - runMudst.C from FCS-ECal-pi0-reconstruction with the two new makers.
//
//   mode = 0 : dump cluster features for training  (StFcsClusterFeatureMaker after the point maker)
//   mode = 1 : run the ML category maker between cluster and point makers, then the pi0 finder
//   mode = 2 : both (ML categories applied, features dumped afterwards for QA)
//
// Example:
//   root4star -b -q 'runMudst_ml.C("st_physics_...MuDst.root",-1,5000,".",1,0,0,"","feat.root")'
//   root4star -b -q 'runMudst_ml.C("st_physics_...MuDst.root",-1,5000,".",1,0,1,"weights/FcsCat13_BDTG.weights.xml")'

void runMudst_ml(const char* file = "st_cosmic_adc_22326042_raw_0000005.MuDst.root",
                 int ifile = -1, Int_t nevt = 10, const char* outdir = ".", int readMuDst = 1,
                 int debug = 0, int mode = 0,
                 const char* modelFile = "weights/FcsCat13_BDTG.weights.xml",
                 const char* featFile = "fcsEcalClusterFeatures.root") {
   gROOT->Macro("Load.C");
   gROOT->Macro("$STAR/StRoot/StMuDSTMaker/COMMON/macros/loadSharedLibraries.C");
   gSystem->Load("StEventMaker");
   gSystem->Load("StFcsDbMaker");
   gSystem->Load("StFcsRawHitMaker");
   gSystem->Load("StFcsWaveformFitMaker");
   gSystem->Load("StFcsClusterMaker");
   gSystem->Load("libMinuit");
   gSystem->Load("StFcsPointMaker");

   StChain* chain = new StChain("StChain");
   chain->SetDEBUG(0);
   StMuDstMaker* muDstMaker = new StMuDstMaker(0, 0, "", file, ".", 1000, "MuDst");
   int n = muDstMaker->tree()->GetEntries();
   printf("Found %d entries in Mudst\n", n);
   int start = 0, stop = n;
   if (ifile >= 0) {
      start = ifile * nevt;
      stop = (ifile + 1) * nevt - 1;
      if (n < start) { printf(" No event left. Exiting\n"); return; }
      if (n < stop) { printf(" Overwriting end event# stop=%d\n", n); stop = n; }
   } else if (nevt >= 0 && nevt < n) {
      stop = nevt;
   } else if (nevt == -2) {
      stop = 2000000000;
   }
   printf("Doing Event=%d to %d\n", start, stop);

   St_db_Maker* dbMk = new St_db_Maker("db", "MySQL:StarDb", "$STAR/StarDb");
   if (dbMk) {
      const char* bl[] = {"tpc", "svt", "ssd", "ist", "pxl", "pp2pp", "ftpc",
                          "emc", "eemc", "mtd", "pmd", "tof", "etof", "rhicf"};
      for (int i = 0; i < 14; i++) dbMk->SetAttr("blacklist", bl[i]);
   }

   StFcsDbMaker* fcsDbMkr = new StFcsDbMaker();
   StFcsDb* fcsDb = (StFcsDb*)chain->GetDataSet("fcsDb");
   fcsDb->setReadGainCorrFromText();

   StEventMaker* eventMk = new StEventMaker();
   StFcsRawHitMaker* hit = new StFcsRawHitMaker();
   hit->setReadMuDst(readMuDst);
   StFcsWaveformFitMaker* wff = new StFcsWaveformFitMaker();
   wff->SetDebug(debug);

   StFcsClusterMaker* clu = new StFcsClusterMaker();
   clu->SetDebug(debug);

   // ---- ML category, strictly between clustering and point fitting ----
   if (mode == 1 || mode == 2) {
      gSystem->Load("libTMVA");  // must come before the maker's library
      gSystem->Load("StFcsMLCategoryMaker");
      StFcsMLCategoryMaker* mlcat = new StFcsMLCategoryMaker();
      mlcat->setFeatureSet(13);  // 13 (default) or 34; must match the weight file
      mlcat->setBackend(0);      // 0 = TMVA::Reader, 1 = plain-text MLP
      mlcat->setTMVAMethod("BDTG");
      mlcat->setWeightFile(modelFile);
      mlcat->setQaFile(Form("%s/fcsMLCategoryQa.root", outdir));
      mlcat->setMode(1);       // 1 = always override, 2 = override only above the confidence cut
      mlcat->setConfidence(0.7);
      mlcat->setEnergyThreshold(0.5);
   }

   StFcsPointMaker* poi = new StFcsPointMaker();
   poi->SetDebug(debug);

   // ---- feature dump, after the point maker so chi2ndf1/chi2ndf2 are filled ----
   if (mode == 0 || mode == 2) {
      gSystem->Load("StFcsClusterFeatureMaker");
      StFcsClusterFeatureMaker* feat = new StFcsClusterFeatureMaker();
      feat->setOutputFile(Form("%s/%s", outdir, featFile));
      feat->setEnergyThreshold(0.5);
      feat->setSaveTruth(1);   // no-op on data
   }

   gSystem->Load("StVpdCalibMaker");
   StVpdCalibMaker* vpdCalib = new StVpdCalibMaker();
   vpdCalib->setMuDstIn();

   gSystem->Load("StFcsPi0FinderForEcal");
   StFcsPi0FinderForEcal* fcsPi0Finder = new StFcsPi0FinderForEcal();
   fcsPi0Finder->st(1);

   chain->Init();
   chain->EventLoop(start, stop);
   chain->Finish();
   delete chain;
}
