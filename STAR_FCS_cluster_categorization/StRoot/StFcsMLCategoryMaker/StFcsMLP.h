// StFcsMLP - a tiny dependency-free feed-forward network evaluator.
//
// Why not ONNX Runtime: it is not part of the STAR library stack at RCF, and
// adding it to a StRoot library is a fight you do not need for a network this
// small. Export your trained model to the plain-text format below with
// python/export_weights.py and the forward pass is ~50 lines of C++.
//
// File format (whitespace separated, '#' starts a comment line):
//
//   STARFCSMLP 1
//   NIN   <nin>
//   MEAN  <nin floats>          # input standardization, applied before layer 0
//   STD   <nin floats>
//   NLAYER <L>
//   LAYER <in> <out> <relu|tanh|sigmoid|linear|softmax>
//   W <out*in floats, row major: W[o*in + i]>
//   B <out floats>
//   ... repeated L times ...
//
// author: generated for Xilin Liang

#ifndef STAR_StFcsMLP_HH
#define STAR_StFcsMLP_HH

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class StFcsMLP {
  public:
   StFcsMLP() : mNIn(0), mOK(false) {}

   bool load(const char* filename) {
      mOK = false;
      mLayers.clear();
      mMean.clear();
      mStd.clear();
      FILE* f = fopen(filename, "r");
      if (!f) {
         printf("StFcsMLP: cannot open %s\n", filename);
         return false;
      }
      std::vector<std::string> tok;
      char line[1 << 16];
      while (fgets(line, sizeof(line), f)) {
         char* p = strtok(line, " \t\n\r");
         while (p) {
            if (p[0] == '#') break;
            tok.push_back(p);
            p = strtok(0, " \t\n\r");
         }
      }
      fclose(f);

      size_t i = 0;
      int nlayer = 0;
      while (i < tok.size()) {
         const std::string& k = tok[i];
         if (k == "STARFCSMLP") {
            i += 2;
         } else if (k == "NIN") {
            mNIn = atoi(tok[i + 1].c_str());
            i += 2;
         } else if (k == "MEAN") {
            for (int j = 0; j < mNIn; j++) mMean.push_back(atof(tok[i + 1 + j].c_str()));
            i += 1 + mNIn;
         } else if (k == "STD") {
            for (int j = 0; j < mNIn; j++) mStd.push_back(atof(tok[i + 1 + j].c_str()));
            i += 1 + mNIn;
         } else if (k == "NLAYER") {
            nlayer = atoi(tok[i + 1].c_str());
            i += 2;
         } else if (k == "LAYER") {
            Layer L;
            L.nin = atoi(tok[i + 1].c_str());
            L.nout = atoi(tok[i + 2].c_str());
            L.act = tok[i + 3];
            i += 4;
            if (i >= tok.size() || tok[i] != "W") {
               printf("StFcsMLP: expected W after LAYER\n");
               return false;
            }
            i++;
            L.W.resize(L.nin * L.nout);
            for (int j = 0; j < L.nin * L.nout; j++) L.W[j] = atof(tok[i + j].c_str());
            i += L.nin * L.nout;
            if (i >= tok.size() || tok[i] != "B") {
               printf("StFcsMLP: expected B after W\n");
               return false;
            }
            i++;
            L.b.resize(L.nout);
            for (int j = 0; j < L.nout; j++) L.b[j] = atof(tok[i + j].c_str());
            i += L.nout;
            mLayers.push_back(L);
         } else {
            i++;  // unknown token, skip
         }
      }
      if ((int)mLayers.size() != nlayer || mLayers.empty()) {
         printf("StFcsMLP: read %d layers, header said %d\n", (int)mLayers.size(), nlayer);
         return false;
      }
      if ((int)mMean.size() != mNIn) mMean.assign(mNIn, 0.0);
      if ((int)mStd.size() != mNIn) mStd.assign(mNIn, 1.0);
      mOK = true;
      printf("StFcsMLP: loaded %s  nin=%d nlayer=%d nout=%d\n", filename, mNIn, (int)mLayers.size(),
             mLayers.back().nout);
      return true;
   }

   bool ok() const { return mOK; }
   int nInput() const { return mNIn; }
   int nOutput() const { return mLayers.empty() ? 0 : mLayers.back().nout; }

   // returns the output vector; empty if the model is not loaded or nin mismatches
   std::vector<float> eval(const std::vector<float>& in) const {
      std::vector<float> out;
      if (!mOK || (int)in.size() != mNIn) return out;
      std::vector<float> v(mNIn);
      for (int i = 0; i < mNIn; i++) v[i] = (in[i] - mMean[i]) / (mStd[i] != 0 ? mStd[i] : 1.0);
      for (size_t l = 0; l < mLayers.size(); l++) {
         const Layer& L = mLayers[l];
         std::vector<float> w(L.nout, 0.0);
         for (int o = 0; o < L.nout; o++) {
            double s = L.b[o];
            const float* Wr = &L.W[(size_t)o * L.nin];
            for (int i = 0; i < L.nin; i++) s += Wr[i] * v[i];
            w[o] = s;
         }
         if (L.act == "relu") {
            for (int o = 0; o < L.nout; o++)
               if (w[o] < 0) w[o] = 0;
         } else if (L.act == "tanh") {
            for (int o = 0; o < L.nout; o++) w[o] = tanh(w[o]);
         } else if (L.act == "sigmoid") {
            for (int o = 0; o < L.nout; o++) w[o] = 1.0 / (1.0 + exp(-w[o]));
         } else if (L.act == "softmax") {
            float m = w[0];
            for (int o = 1; o < L.nout; o++)
               if (w[o] > m) m = w[o];
            double sum = 0;
            for (int o = 0; o < L.nout; o++) {
               w[o] = exp(w[o] - m);
               sum += w[o];
            }
            for (int o = 0; o < L.nout; o++) w[o] /= sum;
         }
         v.swap(w);
      }
      out.swap(v);
      return out;
   }

  private:
   struct Layer {
      int nin;
      int nout;
      std::string act;
      std::vector<float> W;
      std::vector<float> b;
   };
   int mNIn;
   bool mOK;
   std::vector<float> mMean;
   std::vector<float> mStd;
   std::vector<Layer> mLayers;
};

#endif
