// runFzd_ml.C - produce the TRAINING sample, straight from the GEANT .fzd.
//
//   root4star -b -q 'runFzd_ml.C("pi0.e30.vz0.run1.fzd",-1,"feat_pi0.root")'
//
// WHY THIS MACRO EXISTS, AND WHY runMudst_ml.C CANNOT REPLACE IT
//
// The truth labels come from StFcsHit::getGeantTracks(), which is filled only
// by StFcsFastSimulatorMaker as it turns g2t_wca_hit into hits, plus the
// g2t_track table for pid and parentage. Neither survives into a MuDst:
// StMuFcsHit stores detectorId, id, adc and energy and nothing about which
// GEANT track deposited it (StPicoFcsHit likewise). So a feature dump made
// from a MuDst has features but no labels - truthNPhoton is -1 on every
// cluster and trainTMVA.C silently skips the lot.
//
// Training samples therefore come from the .fzd, through this macro. MuDst and
// picoDst are for APPLYING a trained model, not for training it.
//
// The chain: fzin reads the .fzd with St_geant_Maker and builds the g2t tables,
// fcsSim is StFcsFastSimulatorMaker, then the standard fcsCluster/fcsPoint
// reconstruction, then our feature dumper last so that chi2Ndf1/2Photon are
// already filled.
//
// GEOMETRY MUST MATCH THE SIMULATION. The .fzd carries its own geometry banks
// but bfc does not read them - it builds geometry from the tag in the chain
// options. Pass the same tag your starsim job used (look for the geometry line
// in the .kumac that made the .fzd). Getting this wrong does not crash: it
// silently mismaps volumes to detector ids, so the hits land on the wrong
// towers. If the clusters come out in implausible places, suspect this first.
//
// author: generated for Xilin Liang

void runFzd_ml(const char* fzd = "pi0.e30.vz0.run1.fzd",
               Int_t nevt = -1,
               const char* outFile = "feat.root",
               const char* geometry = "",        // REQUIRED: the tag starsim used
               const char* dbTime = "",          // e.g. sdt20230101
               float clusterEmin = 0.5,
               int saveTruth = 1) {
   // No geometry default on purpose. A wrong tag does not fail loudly, it
   // mismaps volumes to tower ids, and you would be training on scrambled
   // clusters without any error to tell you.
   if (strlen(geometry) == 0 || strlen(dbTime) == 0) {
      printf("\nPass the geometry tag and DB timestamp your starsim job used, e.g.\n");
      printf("  root4star -b -q 'runFzd_ml.C(\"%s\",-1,\"feat.root\",\"y2023a\",\"sdt20230101\")'\n\n", fzd);
      printf("Both are in the .kumac that produced the .fzd - look for the geometry\n");
      printf("line ('gexec' / 'detp geom y20..') and use the matching sdt<date>.\n");
      return;
   }

   // fzin  : St_geant_Maker reads the .fzd, g2t_* tables appear in the chain
   // fcsSim: StFcsFastSimulatorMaker - this is what fills the GEANT track links
   // NOTE fcsDat and fcsWFF are deliberately absent: those read real DAQ
   // waveforms. In simulation the fast simulator produces the hits directly.
   TString opts = Form("fzin,%s,%s,fcsSim,fcsCluster,fcsPoint,fcsDb,StEvent", geometry, dbTime);
   printf("chain options: %s\n", opts.Data());

   gROOT->LoadMacro("bfc.C");

   // Last == 0 tells bfc to build the chain and Init() it, then return without
   // looping - that is the documented hook for appending your own makers.
   bfc(0, opts, fzd);

   if (!chain) {
      printf("bfc did not create a chain - check the option string\n");
      return;
   }

   // Appending a maker attaches it to the current chain automatically, but
   // chain->Init() has already run, so this one has to be initialised by hand.
   gSystem->Load("StFcsClusterFeatureMaker");
   StFcsClusterFeatureMaker* feat = new StFcsClusterFeatureMaker();
   feat->setOutputFile(outFile);
   feat->setEnergyThreshold(clusterEmin);
   feat->setSaveTruth(saveTruth);
   feat->setSaveMcTruth(1);      // generator-level photons -> mcLabel, mcSep
   feat->setMcMatchRadius(11.0); // cm, about two ECal towers
   if (feat->Init() != kStOK) {
      printf("StFcsClusterFeatureMaker::Init failed\n");
      return;
   }

   int last = (nevt > 0) ? nevt : 1000000;
   chain->EventLoop(1, last);
   chain->Finish();
   printf("wrote %s\n", outFile);

   // SANITY CHECK BEFORE YOU TRAIN ON THIS. Open the tree and look at:
   //   clusters->Draw("truthNPhoton")            // must not be all -1
   //   clusters->Draw("trkPid[0]","e>1")         // 1 = gamma (GEANT3 pid)
   //   clusters->Draw("trkE[0]/e","e>1")         // leading-track energy share
   // All -1 means the truth never arrived: either fcsSim is not in the chain or
   // the .fzd has no FCS hits. A leading share near 1 with pid 1 means deposits
   // are attributed to the primary photon, which is what the labelling in
   // trainTMVA.C assumes. A spray of e+/e- (pid 2/3) instead means they are
   // attributed to shower secondaries, and the labeller has to walk
   // g2t_track::next_parent_p up to the primary first - tell me if you see that
   // and I will add the parent walk.
}
