#include<thread>


extern "C" {
  CapTree* CapTree_new() { /*TODO*/ }
  void CapTree_del() { /*TODO*/ }
  int CapTree_search() { /*TODO*/ }
  bool CapTree_check_build() { /*TODO*/ }
}

class CapTree {
  public:
    CapTree();
    build();
    search();
  private:
    double r, *y, *xi;
    CapTree *cR, *cL;
    
};
