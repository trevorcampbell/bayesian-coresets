#include<thread>
#include<vector>
#include<iostream>


class CapTree {
  public:
    CapTree(double*, unsigned int, unsigned int);
    ~CapTree();
    bool check_build();
    unsigned int search(double*, double*);
  private:
    double *ys, *xis; 
    double *rs;
    unsigned int *cRs, *cLs;
    bool build_done;
    void build();
};

extern "C" {
  CapTree* CapTree_new(double *data, unsigned int N, unsigned int D) { return new CapTree(data, N, D); }
  void CapTree_del(CapTree *ptr) { if (ptr != NULL){delete ptr;} }
  int CapTree_search(CapTree *ptr, double *yw, double *y_yw) { return ptr->search(yw, y_yw); }
  bool CapTree_check_build(CapTree *ptr) { return ptr->check_build(); }
}

CapTree::CapTree(double *data, unsigned int N, unsigned int D){
  for (int i = 0; i < 3; i++){
    std::cout << data[D*i] << " " << std::endl;
  }
  this->build_done = false;
  //allocate size 2*N-1 taking advantage of complete binary tree
  this->ys = new double[(2*N-1)*D];
  this->xis = new double[(2*N-1)*D];
  this->rs = new double[2*N-1];
  this->cRs = new unsigned int[2*N-1];
  this->cLs = new unsigned int[2*N-1];
}

CapTree::~CapTree(){ 
  if (this->ys != NULL){ delete this->ys; this->ys = NULL; }
  if (this->xis != NULL){ delete this->xis; this->xis = NULL; }
  if (this->rs != NULL){ delete this->rs; this->rs = NULL; }
  if (this->cRs != NULL){ delete this->cRs; this->cRs = NULL; }
  if (this->cLs != NULL){ delete this->cLs; this->cLs = NULL; }
}

unsigned int CapTree::search(double *yw, double *y_yw){
  // if build not done yet, wait on build mutex
  //do search
  return 0;
}

bool CapTree::check_build(){
  //check whether build_done is true
  //w mutex
  return false;
}

void CapTree::build(){
  return;
}



