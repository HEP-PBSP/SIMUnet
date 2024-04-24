// $Id
//
// NNPDF++ 2013
//
// Authors: Nathan Hartland,  n.p.hartland@ed.ac.uk
//          Stefano Carrazza, stefano.carrazza@mi.infn.it
//          Luigi Del Debbio, luigi.del.debbio@ed.ac.uk

#pragma once

#include "buildmaster_utils.h"

// ********* Filters **************

class CMS_1JET_13TEV_R07Filter: public CommonData
{ public: CMS_1JET_13TEV_R07Filter():
  CommonData("CMS_1JET_13TEV_R07") { ReadData(); }

private:
  void ReadData();
};
