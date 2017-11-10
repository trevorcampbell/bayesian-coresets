#include<thread>
#include<vector>
#include<iostream>
#include<mutex>
#include<condition_variable>
#include<queue>
#include<algorithm>


class CapTree {
  public:
    CapTree(double*, unsigned int, unsigned int);
    ~CapTree();
    bool check_build();
    void cancel_build();
    int search(double*, double*, unsigned int D);
    double get_num_search_ops();
    double get_num_build_ops();
  private:
    void build(double*, unsigned int, unsigned int);
    double upper_bound(unsigned int, double*, double*, unsigned int);
    double lower_bound(unsigned int, double*, double*, unsigned int);
    double *ys, *xis; // ys = copy of data (less memory than storing ys for each node), xis = node centers 
    double *rs; //node dot min
    int *cRs, *cLs, *nys; //cR/cL are child idcs, nys = idx of lowr bound vector in the node
    double num_build_ops, num_search_ops;
    bool build_done, build_cancelled;
    std::mutex build_mutex;
    std::condition_variable build_cv;
    std::thread *build_thread;
};

extern "C" {
  CapTree* CapTree_new(double *data, unsigned int N, unsigned int D) { return new CapTree(data, N, D); }
  void CapTree_del(CapTree *ptr) { if (ptr != NULL){delete ptr;} }
  int CapTree_search(CapTree *ptr, double *yw, double *y_yw, unsigned int D) { return ptr->search(yw, y_yw, D); }
  bool CapTree_check_build(CapTree *ptr) { return ptr->check_build(); }
  void CapTree_cancel_build(CapTree *ptr) { return ptr->cancel_build(); }
  double CapTree_num_search_ops(CapTree *ptr) { return ptr->get_num_build_ops(); }
  double CapTree_num_build_ops(CapTree *ptr) { return ptr->get_num_search_ops(); }
}

CapTree::CapTree(double *data, unsigned int N, unsigned int D){
  //start a thread for building the tree
  this->num_build_ops = this->num_search_ops = 0.;
  this->ys = this->xis = this->rs = NULL;
  this->cRs = this->cLs = this->nys = NULL;
  this->build_done = this->build_cancelled = false;
  this->build_thread = new std::thread(&CapTree::build, this, data, N, D);
  return;
}

CapTree::~CapTree(){ 
  //clean up data ptrs if the build thread didn't already do it (if the build was cancelled early)
  if (this->build_thread != NULL){ 
   this->cancel_build(); //if the build is already done, does nothing; if the build is in progress, cancels it
   this->build_thread->join(); //wait for the thread to terminate 
   delete this->build_thread;  //delete it
   this->build_thread = NULL; //set ptr to null
  }
  if (this->ys != NULL){ delete this->ys; this->ys = NULL; }
  if (this->xis != NULL){ delete this->xis; this->xis = NULL; }
  if (this->rs != NULL){ delete this->rs; this->rs = NULL; }
  if (this->nys != NULL){ delete this->nys; this->nys = NULL; }
  if (this->cRs != NULL){ delete this->cRs; this->cRs = NULL; }
  if (this->cLs != NULL){ delete this->cLs; this->cLs = NULL; }
}

double CapTree::num_build_ops(){
  std::lock_guard<std::mutex> lk(this->build_mutex);
  return this->num_build_ops;
}

double CapTree::num_search_ops(){
  return this->num_search_ops;  
}

int CapTree::search(double *yw, double *y_yw, unsigned int D){
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

  //search
  double nf = 2.;
  double LB = -2.;
  int nopt = -1;
  auto cmp = [](std::tuple<unsigned int, double> left, std::tuple<unsigned int, double> right){ return std::get<1>(left) < std::get<1>(right); };
  std::priority_queue<std::tuple<unsigned int, double>, std::vector< std::tuple<unsigned int, double> >, decltype(cmp)> search_queue(cmp);
  search_queue.push(std::make_tuple(0, this->upper_bound(0, yw, y_yw, D) ));
  while (!search_queue.empty()){
    //get the next cap node to search
    auto tpl = search_queue.top();
    search_queue.pop();
    auto idx = std::get<0>(tpl);
    auto u = std::get<1>(tpl);

    //if its upper bound is greater than the current maximum LB
    if (u > LB){
      //compute the LB
      double ell = this->lower_bound(idx, yw, y_yw, D);
      nf += 2.;
      //if its lb is greater than the current best
      if (ell > LB){
        //update the max LB and store data idx that achieved it
        LB = ell;
        nopt = this->nys[idx];
      }
      //if this node has children, push them onto the search pq
      if (this->cRs[idx] > -0.5){ 
        search_queue.push(std::make_tuple(this->cRs[idx], this->upper_bound(this->cRs[idx], yw, y_yw, D)));
        search_queue.push(std::make_tuple(this->cLs[idx], this->upper_bound(this->cLs[idx], yw, y_yw, D)));
        nf += 4.;
      }
    }
  }
  this->num_search_ops += nf;
  return nopt;
}

double CapTree::upper_bound(unsigned int node_idx, double *yw, double *y_yw, unsigned int D){
    double bu, bv, b, rv, r1;
    bu = bv = b = rv = r1 = 0.;
    for (unsigned int d = 0; d < D; d++){
      bu += this->xis[node_idx*D+d]*y_yw[d];
      bv += this->xis[node_idx*D+d]*yw[d];
    }
    b = sqrt(std::max(0., 1. - bu*bu - bv*bv));
    rv = sqrt(std::max(0., this->rs[node_idx]*this->rs[node_idx] - bv*bv));
    r1 = sqrt(std::max(0., 1. - this->rs[node_idx]*this->rs[node_idx]));
    if (fabs(bv) > this->rs[node_idx] || bu >= rv){
      return 1.;
    } else {
      return (bu*rv+b*r1)/(b*b+bu*bu);
    }
}

double CapTree::lower_bound(unsigned int node_idx, double *yw, double *y_yw, unsigned int D){
    double bu, bv;
    bu = bv = 0.;
    for (unsigned int d = 0; d < D; d++){ 
      bu += this->ys[this->nys[node_idx]*D+d]*y_yw[d];
      bv += this->ys[this->nys[node_idx]*D+d]*yw[d];
    }
    if (1.-bv*bv <= 0. || bv <= -1.+1e-14){
      //the first condition can occur when y = +/- y_w, and here the direction is not well defined 
      //the second can happen when y is roughly =  -y_w, and here the direction is not numerically stable
      //in either case, we want to return a failure - output = -3 indicates this
      return -3.;
    } else {
      return bu/sqrt(1.-bv*bv);
    }
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
  //initialize memory -- since tree is full/complete, it must have 2*N-1 nodes
  this->ys = new double[N*D];
  this->xis = new double[(2*N-1)*D];
  this->rs = new double[2*N-1];
  this->nys = new int[2*N-1];
  this->cRs = new int[2*N-1];
  this->cLs = new int[2*N-1];


  for (unsigned int d = 0; d < N*D; d++){
    this->ys[d] = data[d];
  }

  double nbo = (double)N;
  
  //top node has all the data in it; initialize index list with all idcs from 1 to N
  std::vector< unsigned int > full_idcs(N);
  std::iota(full_idcs.begin(), full_idcs.end(), 0);

  //initialize queues for node construction
  unsigned int cur_idx = 0;
  std::queue< std::vector<unsigned int> > data_idcs_queue;
  std::queue< unsigned int > node_idx_queue;
  data_idcs_queue.push(full_idcs);
  node_idx_queue.push(cur_idx++);

  while (!data_idcs_queue.empty()){
    //get next node construction job
    auto data_idcs = data_idcs_queue.front();
    data_idcs_queue.pop();
    auto node_idx = node_idx_queue.front();
    node_idx_queue.pop();

    /////////////////////////
    //node construction code
    /////////////////////////


    /////////////////////////////////////
    //if only one idx, just store and quit
    //////////////////////////////////////
    if (data_idcs.size() == 1){
      auto didx = data_idcs[0];
      for (unsigned int d = 0; d < D; d++){
        this->xis[node_idx*D+d] = data[didx*D+d];
      }
      this->rs[node_idx] = 1.;
      this->nys[node_idx] = didx;
      this->cRs[node_idx] = this->cLs[node_idx] = -1;
      nbo += 1.;
      continue;
    }
    
    ////////////////////////////////
    //first get mean and store in xi
    ////////////////////////////////
    for (unsigned int d = 0; d < D; d++){
      this->xis[node_idx*D+d] = 0.;
    }
    for (auto& didx: data_idcs){
      for (unsigned int d = 0; d < D; d++){
        this->xis[node_idx*D+d] += data[didx*D+d]; 
      } 
    }
    double xinrmsq = 0.;
    for (unsigned int d = 0; d < D; d++){
      xinrmsq += this->xis[node_idx*D+d]*this->xis[node_idx*D+d];
    }
    //if xi has 0 norm, just set to 1st datapoint in list
    if (xinrmsq == 0.){
      auto& didx = data_idcs[0];
      for (unsigned int d = 0; d < D; d++){
        this->xis[node_idx*D+d] = data[didx*D+d]; 
      }
    } else {
      //otherwise, normalize the sum
      for (unsigned int d = 0; d < D; d++){
        this->xis[node_idx*D+d] /= sqrt(xinrmsq);
      }
    }

    //////////////////////////////////
    //find vec of min/max angle to xi, set r and y
    //////////////////////////////////
    double dotmin = 2.;
    double dotmax = -2.;
    unsigned int cR = 0, cY = 0;
    for (auto& didx: data_idcs){
      double dot = 0.;
      for (unsigned int d = 0; d < D; d++){
        dot += this->xis[node_idx*D+d]*data[didx*D+d];
      }
      if (dot < dotmin){
        dotmin = dot;
        cR = didx;
      }
      if (dot > dotmax){
        dotmax = dot; 
        cY = didx;
      }
    }
    //threshold the dotmin between -1 and 1 to avoid numerical issues
    dotmin = dotmin > 1. ? 1. : dotmin; dotmin = dotmin < -1. ? -1. : dotmin;
    this->rs[node_idx] = dotmin;
    this->nys[node_idx] = cY;
    
    ////////////////////////////////////////
    //find vec of max angle to cR
    ////////////////////////////////////////
    dotmin = 2.;
    unsigned int cL = 0;
    for (auto& didx: data_idcs){
      double dot = 0.;
      for (unsigned int d = 0; d < D; d++){
        dot += data[cR*D+d]*data[didx*D+d];
      }
      if (dot < dotmin){
        dotmin = dot;
        cL = didx;
      }
    }

    ////////////////////////////////////////
    //for each vector, allocate to either cL or cR
    ////////////////////////////////////////
    std::vector<unsigned int> r_idcs, l_idcs;
    for (auto& didx: data_idcs){
      double dotL = 0., dotR = 0.;
      for (unsigned int d = 0; d < D; d++){
        dotL += data[cL*D+d]*data[didx*D+d];
        dotR += data[cR*D+d]*data[didx*D+d];
      }
      if (dotL > dotR){
        l_idcs.push_back(didx);
      } else {
        r_idcs.push_back(didx);
      }
    }
    
    ///////////////////////////////////////////
    //in pathological cases, one might be empty; here just allocate evenly
    ///////////////////////////////////////////
    if (r_idcs.empty() || l_idcs.empty()){
      r_idcs.clear(); l_idcs.clear();
      for (unsigned int i = 0; i < data_idcs.size()/2; i++){
        r_idcs.push_back(data_idcs[i]);
      }
      for (unsigned int i = data_idcs.size()/2; i < data_idcs.size(); i++){
        l_idcs.push_back(data_idcs[i]);
      }
    }

    ///////////////////////////////////////////
    //add the two child construction jobs to the queue
    ///////////////////////////////////////////
    data_idcs_queue.push(r_idcs);
    this->cRs[node_idx] = cur_idx;
    node_idx_queue.push(cur_idx++);

    data_idcs_queue.push(l_idcs);
    this->cLs[node_idx] = cur_idx;
    node_idx_queue.push(cur_idx++);
    

    /////////////////////////////////////////////////
    //if build gets cancelled, clean up & quit early
    /////////////////////////////////////////////////
    { 
      std::lock_guard<std::mutex> lk(this->build_mutex);
      if (this->build_cancelled){
        delete this->ys; this->ys = NULL;
        delete this->xis; this->xis = NULL;
        delete this->rs; this->rs = NULL;
        delete this->nys; this->nys = NULL;
        delete this->cRs; this->cRs = NULL;
        delete this->cLs; this->cLs = NULL;
        return;
      }
    }
    nbo += 3. + 4.*data_idcs.size();
  }

  //acquire the lock on the build mutex, set build_done to true, unlock, and notify all waiting threads (possibly ::search)
  std::unique_lock<std::mutex> lk(this->build_mutex);
  this->build_done = true;
  this->num_build_ops = nbo;
  lk.unlock();
  this->build_cv.notify_all();
  return;
}



