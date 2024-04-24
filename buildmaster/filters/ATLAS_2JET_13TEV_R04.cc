/*
Name_exp  : ATLAS_2JET_13TEV_R04
Reference : Measurement of dijet cross-sections in pp collisions at 13 TeV
            centre-of-mass energy using the ATLAS detector
ArXiv     : arXiv:1711.02692
Published : JHEP 05 (2018) 195
Hepdata   : https://www.hepdata.net/record/ins1634970

Inclusive jet and dijet cross-sections are measured in proton-proton collisions at a centre-of-mass energy of 13 TeV.
 The measurement uses a dataset with an integrated luminosity of 3.2 fb−1−1 recorded in 2015 with the ATLAS detector at the Large Hadron Collider. 
Jets are identified using the anti-ktt​ algorithm with a radius parameter value of R = 0.4. 
The inclusive jet cross-sections are measured double-differentially as a function of the jet transverse momentum, covering the range from 100 GeV to 3.5 TeV, 
and the absolute jet rapidity up to |y| = 3. The double-differential dijet production cross-sections are presented as a function of the dijet mass, 
covering the range from 300 GeV to 9 TeV, and the half absolute rapidity separation between the two leading jets within |y| < 3, y∗∗, up to y∗∗ = 3. 
Next-to-leading-order, and next-to-next-to-leading-order for the inclusive jet measurement, perturbative QCD calculations corrected for non-perturbative
 and electroweak effects are compared to the measured cross-sections.

The information on the experimental uncertainty is retrieved from hepdata.
Systematics uncertainties are fully corrrelated in dijet mass and rapidity.  

Total number of points        136
Total number of sys per point 336

bin 1:   0 < |y*| < 0.5
========================
points    28
real sys  336

bin 2:   0.5 < |y*| < 1.0
========================
points    28
real sys  336

bin 3:   1.0 < |y*| < 1.5
========================
points    27
real sys  336

bin 4:   1.5 < |y*| < 2.0
========================
points    24
real sys  336

bin 5:   2.0 < |y*| < 2.5
========================
points    21
real sys  336

bin 6:   2.5 < |y*| < 3.0
========================
points    8
real sys  336

Implemented by FL March 2024. 
*/

#include "ATLAS_2JET_13TEV_R04.h"

void ATLAS_2JET_13TEV_R04Filter::ReadData()
{

  //opening files
  fstream rS, rCorr;

  //bins specification
  int nbins = 6;
  std::vector<double> y = {0.25, 0.75, 1.25, 1.75, 2.25, 2.75}; // Y*=|Y_1 - Y_2| / 2
  std::vector<int> ndata  = {28, 28, 27, 24, 21, 8};    
  std::vector<double> stat(fNData);

  int n = 0;                                //count total number of datapoints

  for(int bin=0; bin < nbins; bin++ )
    {
    
      string data_file = "/bin_" + to_string(bin+1) + ".dat";
      stringstream DataFile("");
      DataFile << dataPath() << "rawdata/" << fSetName << data_file;
      rS.open(DataFile.str().c_str(), ios::in);
      if (rS.fail()) {
	cerr << "Error opening data file " << DataFile.str() << endl;
	exit(-1);
      }

    
      string line;
      double m12, dum;

      for (int i = n; i < n + ndata[bin]; i++)
	{
      
	  getline(rS,line);                     
	  istringstream lstream(line);          

	  lstream >> m12;
	  lstream >> dum >> dum;
      
	  fKin1[i] = y[bin];           // y, central value of the bin
	  fKin2[i] = m12;               // pt2, central value of the bin
	  fKin3[i] = 13000;            // sqrt(s)

	  lstream >> fData[i];         // cross section [fb/GeV]

	  //Read uncertainties
	  double sys1, sys2;

	  //1) Stat
	  double statsymm, dummy, delta;
	  lstream >> sys1 >> sys2;
	  symmetriseErrors(sys1,sys2,&statsymm,&dummy);
	  fStat[i] = statsymm;
      
	  //2) Systematics
      
	  double uncsymm;
	  for (int k=0; k<fNSys; k++)
	    {

	      lstream >> sys1 >> sys2;
	    
	      symmetriseErrors(sys1,sys2,&uncsymm,&delta);	  	  
	  
	      fSys[i][k].type = MULT;   
	      fSys[i][k].name = "CORR";
	      fSys[i][k].add = uncsymm;
	      fSys[i][k].mult = fSys[i][k].add*100/fData[i];

	      fData[i] += delta;
	    }
      
	  fSys[i][335].name = "ATLASLUMI";   
     
	}
          
      rS.close();
      n += ndata[bin];

    }
  
}
