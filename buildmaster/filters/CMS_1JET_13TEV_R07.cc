/*
Name_exp  : CMS_1JET_13TEV_R07
Reference : Measurement and QCD analysis of double-differential inclusive jet 
            cross-sections in pp collisions at s√= 13 TeV
ArXiv     : arXiv:2111.10431
Published : JHEP 02 (2022) 142, JHEP 12 (2022) 035 (addendum)
Hepdata   : https://www.hepdata.net/record/ins1972986

A measurement of the inclusive jet production in proton-proton collisions at the LHC at ss = 13 TeV is presented. 
The double-differential cross sections are measured as a function of the jet transverse momentum pTT​ and the absolute jet rapidity |y|. 
The anti-kTT​ clustering algorithm is used with distance parameter of 0.4 (0.7) in a phase space region with jet pTT​ from 97 GeV up to 3.1 TeV and |y| < 2.0.
Data collected with the CMS detector are used, corresponding to an integrated luminosity of 36.3 fb−1−1 (33.5 fb−1−1).
The measurement is used in a comprehensive QCD analysis at next-to-next-to-leading order, which results in significant improvement in the accuracy of the parton distributions in the proton. 
Simultaneously, the value of the strong coupling constant at the Z boson mass is extracted as αSS​(mZZ​) = 0.1170±0.0019. 
For the first time, these data are used  in a standard model effective field theory analysis at next-to-leading order,
where parton distributions and the QCD parameters are extracted simultaneously with imposed constraints on the Wilson coefficient c11​ of 4-quark contact interactions.


The information on experimental ucnertainties is retrieved from 
the hepdata entry (https://www.hepdata.net/record/ins1972986).
Correlations between statistical uncertainties are taken into account. 
Statistical uncertainties are correlated only between different pT bins of 
the same rapidity range due to unfolding. Different rapidities bins are
statistically correlated.
NData artificial systematic uncertainties are generated
to take into account such correlations.
There are 30 sources of uncertainty.


bin 1:   0 < |y| < 0.5
========================
points    22 
real sys  30

bin 2:   0.5 < |y| < 1.0
========================
points    21
real sys  30

bin 3:   1.5 < |y| < 2.0
========================
points    19 
real sys  30

bin 4:   2.0 < |y| < 2.5
========================
points    16 
real sys  30

tot points         78
tot sys per point  30

Implemented by FL March 2024.
*/

#include "CMS_1JET_13TEV_R07.h"
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

void CMS_1JET_13TEV_R07Filter::ReadData()
{

  //opening files
  fstream rS, rCorr, rSNP, rSEW;

  //bins specification
  int nbins = 4;
  int realsys=30;
  
  std::vector<double> y = {0.25, 0.75, 1.25, 1.75};
  std::vector<int> ndata  = {22, 21, 19, 16};
  std::vector<double> stat(fNData);  
  std::vector<double> jer(fNData);

  int n = 0;                                //count total number of datapoints

  for(int bin=0; bin < nbins; bin++ )
  {
    
    string data_file = "/CMS_13TeV_jets_Ybin" + to_string(bin+1) + ".dat";
    stringstream DataFile("");
    DataFile << dataPath() << "rawdata/" << fSetName << data_file;
    rS.open(DataFile.str().c_str(), ios::in);
    if (rS.fail()) {
      cerr << "Error opening data file " << DataFile.str() << endl;
      exit(-1);
    }

    
    string line;
    double pt, dum;    

    for (int i = n; i < n + ndata[bin]; i++)
    {
      
      getline(rS,line);                     
      istringstream lstream(line);          

      lstream >> pt;
      lstream >> dum >> dum;
      
      fKin1[i] = y[bin];           // y, central value of the bin
      fKin2[i] = pt;               // pt2, central value of the bin
      fKin3[i] = 13000;            // sqrt(s)

      lstream >> fData[i];         // cross section [fb/GeV]

      //Read uncertainties
      double sys1, sys2;
      
      // 2) STATISTICAL
      fStat[i]=0.;

      // 3) UNCORRELATED
      double symm, delta1;
      lstream >> sys1 >> sys2;
      symmetriseErrors(sys1,sys2,&symm,&delta1);
      fSys[i][0].type = MULT;   
      fSys[i][0].name = "UNCORR";
      fSys[i][0].mult = symm;
      fSys[i][0].add  = fSys[i][0].mult/(fData[i]+delta1)*100;

      // 4) other SYS UNCS
       double uncesymm1, delta2;
      for (int l=1; l<6; l++)
	{
	  lstream >> sys1 >> sys2;
	  symmetriseErrors(sys1,sys2,&uncesymm1,&delta2);
	  //sort out uncertainties
	  //double tmp1, tmp2;
	  //tmp1 = sys1;
	  //tmp2 = sys2; 

	  fSys[i][l].type = MULT;   
	  fSys[i][l].name = "CORR";
	  fSys[i][l].mult = uncesymm1;
	  fSys[i][l].add  = fSys[i][l].mult/fData[i]*100;

	  fData[i]+=delta2;

	}

      //5) JER 
      lstream >> jer[i];
      fSys[i][6].type = MULT;   
      fSys[i][6].name = "UNCORR";
      fSys[i][6].mult = 0.;
      fSys[i][6].add  = fSys[i][6].mult/fData[i]*100;
      
      //6) other SYS UNCS
       double uncersymm, delta3;
      for (int k=7; k<realsys; k++)
	{
	  lstream >> sys1 >> sys2;
	  symmetriseErrors(sys1,sys2,&uncersymm,&delta3);

	  fSys[i][k].type = MULT;   
	  fSys[i][k].name = "CORR";
	  fSys[i][k].mult = uncersymm;
	  fSys[i][k].add  = fSys[i][k].mult/fData[i]*100;

	  fData[i]+=delta1;

	  fSys[i][28].name = "CMSLUMI";
         
	}
    }

    n+=ndata[bin];
    rS.close();
  }

  
      
    /* NP Uncertainties to be implemented on theory predictions (FK)

      double np_corr, np_corr_erp, np_corr_erm;
      for(int bin=0; bin < nbins; bin++ )
  {

    string data_file = "/CMS_13TeV_NP_ybin" + to_string(bin+1) + ".dat";
    stringstream DataFile("");
    DataFile << dataPath() << "rawdata/" << fSetName << data_file;
    rSNP.open(DataFile.str().c_str(), ios::in);
    if (rSNP.fail()) {
      cerr << "Error opening data file " << DataFile.str() << endl;
      exit(-1);
    }

     
      string line;
      double np_corr, dum, n=0;
 

      for (int i = 0; i < 5; i++)  
      getline(rS,line);
      
      for(int i=n; i<n+ndata[bin]; i++){
	getline(rS,line);                     
        istringstream lstream(line);

      for(int k=0; k<6; k++){

	double np_corr, np_corr_erp, np_corr_erm, npsymm, npdelta;
      lstream >> dum >> dum >> dum >> np_corr >> np_corr_erp >> np_corr_erm;
      symmetriseErrors(np_corr_erm, np_corr_erp, &npsymm, &npdelta);
      fSys[i][k].mult = npsymm;
      fSys[i][k].add  = fSys[i][k].mult/fData[i]*100;
      fSys[i][k].type = MULT;   
      fSys[i][k].name = "SKIP";
      fData[i]+=npdelta;
      
	}
      }

    rSNP.close();
    n+=ndata[bin];
      
  }*/
 
      /*EW Uncertainties to be implemented on theory prediction (FK)

      double ew_corr;
      for(int bin=0; bin < nbins; bin++ )
  {

    string data_file = "/CMS_13TeV_EW_ybin" + to_string(bin+1) + ".dat";
    stringstream DataFile("");
    DataFile << dataPath() << "rawdata/" << fSetName << data_file;
    rSEW.open(DataFile.str().c_str(), ios::in);
    if (rSEW.fail()) {
      cerr << "Error opening data file " << DataFile.str() << endl;
      exit(-1);
    }                   
      
      for(int i=n; i<n+ndata[bin]; i++){
	istringstream lstream(line);
	getline(rSEW,line);                     

      for(int k=0; k<4; k++){

	lstream >> dum >> dum >> dum >> ew_corr;

      fSys[i][k].mult = ew_corr;
      fSys[i][k].add  = fSys[i][k].mult/fData[i]*100;
      fSys[i][k].type = MULT;   
      fSys[i][k].name = "SKIP";
      
	}
      }
  
 
    rSEW.close();
    n+=ndata[bin];
    
    } */

    
  //Defining covariance matrix for statistical uncertainties
  double** covmat = new double*[fNData];
  for (int i = 0; i < fNData; i++) 
    covmat[i] = new double[fNData];
    
  //Initialise covariance matrix
  for (int i = 0 ; i < fNData; i++)
    {
      for (int j = 0; j < fNData; j++)
	{
	  covmat[i][j] = jer[i]*jer[j];
	    
	}
       
    }
    
	  
    // int k = 0;
  string line;
    
     //Reading correlation coefficients
  for(int bin=0; bin < nbins; bin++ )
    {
      for(int bin1=0; bin1 < nbins; bin1++)
	{
	  
	 
	  string cov_file =  "/CMS_13TeV_jets_Ybin" + to_string(bin+1) + "__CMS_13TeV_jets_Ybin" + to_string(bin1+1) + ".dat";
	  stringstream DataFileCorr("");
	  DataFileCorr << dataPath() << "rawdata/" << fSetName << cov_file;
	  rCorr.open(DataFileCorr.str().c_str(), ios::in);
   
	  if (rCorr.fail())
	    {
	      cerr << "Error opening data file " << DataFileCorr.str() << endl;
	      exit(-1);
	    }  
	  
	  
	  for (int i = 0; i < 6; i++)
	    getline(rCorr,line);
	  
	  
	  int startRow = accumulate(ndata.begin(), ndata.begin() + bin, 0);
	  int startCol = accumulate(ndata.begin(), ndata.begin() + bin1, 0);

	  for (int i = 0; i < ndata[bin]; i++) {
	    for (int j = 0; j < ndata[bin1]; j++) {
	      double dum, rho;
	      getline(rCorr, line);
	      rCorr >> dum >> dum >> dum >> dum >> dum >> dum;
	      rCorr >> rho;
	      cout << "rho" << rho << endl;
	      for (int row = 0; row < ndata[bin]; ++row) {
		for (int col = 0; col < ndata[bin1]; ++col) {
		  covmat[startRow + row][startCol + col] *= rho;
		}
	      }
	    }
	  }

	  rCorr.close();
        }
    }
    

  //Generate artificial systematics
  double** syscor = new double*[fNData];
  for(int i = 0; i < fNData; i++)
    syscor[i] = new double[fNData];
  
  if(!genArtSys(fNData,covmat,syscor))
    {
      cerr << " in " << fSetName << " : cannot generate artificial systematics" << endl;
      exit(-1);
    }
  
  //Assign artificial systematics to data points consistently
  for (int i = 0; i < fNData; i++)
    {
      for (int l = realsys; l < fNSys; l++)
	{
	  fSys[i][l].add = syscor[i][l-realsys];
	  fSys[i][l].mult = fSys[i][l].add/fData[i]*1e2;
	  fSys[i][l].type = ADD;
	  fSys[i][l].name = "CORR";
	}
    }
  
   for (int i = 0; i < fNData; i++) {
    delete[] syscor[i];
    delete[] covmat[i];
}
delete[] syscor;
delete[] covmat;
  
}




