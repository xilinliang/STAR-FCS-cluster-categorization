// runPicoDst_ml.C - apply the cluster categorization to StPicoDst input.
//
//   root4star -b -q 'runPicoDst_ml.C("pi0.e30.vz0.run3.picoDst.root",-1,1,"weights/FcsCat6_BDTG.weights.xml")'
//
// The input may be a single .picoDst.root file or a .list of them; StPicoDstMaker
// takes either.
//
// withModel = 0 dumps the features and the STAR category with no model at all.
// Run that first: it tells you what the input looks like before there is
// anything to evaluate, and it needs no weight file.
//
// FEATURE SET 6 IS NOT A CHOICE HERE, IT IS WHAT PICODST SUPPORTS.
// StPicoFcsCluster keeps the cluster summary but not its tower list
// (StPicoDstMaker::fillFcsClusters drops StMuFcsCluster::hits()), so the seven
// tower-level variables of set 13 cannot be computed from a picoDst. Train with
//     root4star -b -q 'trainTMVA.C("feat.root","FcsCat",6)'
// on a MuDst-produced feature tree, and the resulting weight file applies here
// unchanged - same six variables, same names, same definitions, one shared
// implementation in StFcsClusterFeatures.h.
//
// author: generated for Xilin Liang

void runPicoDst_ml(const char* input = "pi0.e30.vz0.run3.picoDst.root",
                   Int_t nevt = -1,
                   int withModel = 0,
                   const char* weightFile = "weights/FcsCat6_BDTG.weights.xml",
                   const char* outFile = "fcsPicoCategory.root",
                   const char* tmvaMethod = "BDTG") {
   gROOT->Macro("LoadLogger.C");
   gSystem->Load("St_base");
   gSystem->Load("StChain");
   gSystem->Load("StUtilities");
   gSystem->Load("StBFChain");
   gSystem->Load("StIOMaker");
   gSystem->Load("StarClassLibrary");
   gSystem->Load("StTreeMaker");
   gSystem->Load("StEvent");
   gSystem->Load("StPicoEvent");
   gSystem->Load("StPicoDstMaker");
   if (withModel) gSystem->Load("libTMVA");  // must precede our library
   gSystem->Load("StFcsPicoCategoryMaker");

   StChain* chain = new StChain("StChain");

   StPicoDstMaker* picoMaker = new StPicoDstMaker(StPicoDstMaker::IoRead, input, "picoDst");
   // Read only what we use. FCS clusters carry everything set 6 needs; FcsHits
   // is enabled too so you can look at the tower spectra, but nothing in this
   // chain requires it - drop it if you want the I/O to be leaner.
   picoMaker->SetStatus("*", 0);
   picoMaker->SetStatus("Event*", 1);
   picoMaker->SetStatus("FcsClusters*", 1);
   picoMaker->SetStatus("FcsHits*", 1);

   StFcsPicoCategoryMaker* cat = new StFcsPicoCategoryMaker(picoMaker);
   cat->setFeatureSet(6);
   cat->setNoModel(withModel ? 0 : 1);
   cat->setWeightFile(weightFile);
   cat->setTMVAMethod(tmvaMethod);
   cat->setOutputFile(outFile);
   cat->setMode(1);           // 1 = take the model's answer, 2 = only above setConfidence
   cat->setConfidence(0.7);
   cat->setEnergyThreshold(0.5);   // per cluster, for the tree
   cat->setPairEnergyThreshold(1.0);  // per cluster, for the pi0 pairing
   cat->setZggMax(0.7);

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
   printf("wrote %s\n", outFile);
}
