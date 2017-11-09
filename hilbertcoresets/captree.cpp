#include<thread>
#include<vector>


extern "C" {
  CapTree* CapTree_new() { /*TODO*/ }
  void CapTree_del() { /*TODO*/ }
  int CapTree_search() { /*TODO*/ }
  bool CapTree_check_build() { /*TODO*/ }
}

class CapTree {
  public:
    CapTree();
    ~CapTree();
    build();
    check_build();
    search();
  private:
    double *ys, *xis;
    double *rs;
    int *cRs, *cLs;
    bool build_done;
};

CapTree::CapTree(){
  this->build_done = false;
  this->cR = this->cL = this->y = this->xi = NULL;
  this->r  = -2.
}

CapTree::~CapTree(){ 
  for (auto blah){
    delete blah
  }
}

CapTree::build(){
}


int CapTree::search(){
  // if build not done yet, wait on build mutex
  //do search
}

bool CapTree::check_build(){
  //check whether build_done is true
}


