#include<thread>
#include<vector>
#include<iostream>
#include<mutex>
#include<condition_variable>


class CapTree {
  public:
    CapTree(double*, unsigned int, unsigned int);
    ~CapTree();
    bool check_build();
    void cancel_build();
    int search(double*, double*);
  private:
    void build(double*, unsigned int, unsigned int);
    double *ys, *xis; 
    double *rs;
    unsigned int *cRs, *cLs;
    bool build_done, build_cancelled;
    std::mutex build_mutex;
    std::condition_variable build_cv;
    std::thread *build_thread;
};

extern "C" {
  CapTree* CapTree_new(double *data, unsigned int N, unsigned int D) { return new CapTree(data, N, D); }
  void CapTree_del(CapTree *ptr) { if (ptr != NULL){delete ptr;} }
  int CapTree_search(CapTree *ptr, double *yw, double *y_yw) { return ptr->search(yw, y_yw); }
  bool CapTree_check_build(CapTree *ptr) { return ptr->check_build(); }
  void CapTree_cancel_build(CapTree *ptr) { return ptr->cancel_build(); }
}

CapTree::CapTree(double *data, unsigned int N, unsigned int D){
  //start a thread for building the tree
  this->build_done = this->build_cancelled = false;
  this->build_thread = new std::thread(&CapTree::build, this, data, N, D);
  return;
}

CapTree::~CapTree(){ 
  //clean up data ptrs if the build thread didn't already do it (if the build was cancelled early)
  if (this->ys != NULL){ delete this->ys; this->ys = NULL; }
  if (this->xis != NULL){ delete this->xis; this->xis = NULL; }
  if (this->rs != NULL){ delete this->rs; this->rs = NULL; }
  if (this->cRs != NULL){ delete this->cRs; this->cRs = NULL; }
  if (this->cLs != NULL){ delete this->cLs; this->cLs = NULL; }
  if (this->build_thread != NULL){ 
   this->cancel_build(); //if the build is already done, does nothing; if the build is in progress, cancels it
   this->build_thread.join(); //wait for the thread to terminate 
   delete this->build_thread;  //delete it
   this->build_thread = NULL; //set ptr to null
  }
}

int CapTree::search(double *yw, double *y_yw){
  {
    // if build not done yet & has not been cancelled already, wait on build mutex
    std::unique_lock<std::mutex> lk(this->build_mutex);
    if (!this->build_done && !this->build_cancelled){
      this->build_cv.wait(lk);
    }
    //we reacquire the lock here; check if the build was cancelled & did not finish in the mean time
    if (!this->build_done && this->build_cancelled){
      return -1;
    }
    //otherwise, build is done and we can search
  }

  //TODO search
  
  return 0;
}

bool CapTree::check_build(){
  //check whether build_done is true
  std::lock_guard<std::mutex> lk(this->build_mutex);
  return this->build_done; //return value initialized/copied before lock_guard falls out of scope
}

void CapTree::cancel_build(){
  //cancel the build (if build_done is already true, this will have no effect)
  std::unique_lock<std::mutex> lk(this->build_mutex);
  this->build_cancelled = true;
  lk.unlock();
  this->build_cv.notify_all();
  return;
}

void CapTree::build(double *data, unsigned int N, unsigned int D){
  this->ys = new double[(2*N-1)*D];
  this->xis = new double[(2*N-1)*D];
  this->rs = new double[2*N-1];
  this->cRs = new unsigned int[2*N-1];
  this->cLs = new unsigned int[2*N-1];

  //if build gets cancelled, quit early and clean up
  { 
    std::lock_guard<std::mutex> lk(this->build_mutex);
    if (this->build_cancelled){
      delete this->ys; this->ys = NULL;
      delete this->xis; this->xis = NULL;
      delete this->rs; this->rs = NULL;
      delete this->cRs; this->cLs = NULL;
      delete this->cRs; this->cLs = NULL;
      return;
    }
  }

  //acquire the lock on the build mutex, set build_done to true, unlock, and notify all waiting threads (possibly ::search)
  std::unique_lock<std::mutex> lk(this->build_mutex);
  this->build_done = true;
  lk.unlock();
  this->build_cv.notify_all();
  return;
}



