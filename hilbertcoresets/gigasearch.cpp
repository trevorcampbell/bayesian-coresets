#include<thread>
#include<future>
#include<vector>
#include<iostream>
#include<mutex>
#include<queue>
#include<algorithm>
#include<random>


class GIGASearch{
  public:
    GIGASearch(double*, unsigned int, unsigned int);
    ~GIGASearch();
    void cancel_build();
    bool check_build();
    int search(double*, double*);
    double get_num_search_ops();
    double get_num_search_nodes();
    double get_num_build_ops();
  private:
    unsigned int N, D;
    double *data, *xis; // data = copy of data ptr, xis = node centers 
    double *rs; //node dot min
    int *cRs, *cLs, *nys; //cR/cL are child idcs, nys = idx of lowr bound vector in the node
    double num_build_ops, num_search_ops, num_search_nodes;
    bool build_done, build_cancelled, search_done;
    std::mutex build_mutex, search_mutex;
    std::thread *build_thread;

    void build();
    int search_tree(double*, double*);
    int search_linear(double*, double*);
    double upper_bound(unsigned int, double*, double*);
    double lower_bound(unsigned int, double*, double*);
};

extern "C" {
  GIGASearch* GIGASearch_new(double *data, unsigned int N, unsigned int D) { return new GIGASearch(data, N, D); }
  void GIGASearch_del(GIGASearch *ptr) { if (ptr != NULL){delete ptr;} }
  int GIGASearch_search(GIGASearch *ptr, double *yw, double *y_yw) { return ptr->search(yw, y_yw); }
  void GIGASearch_cancel_build(GIGASearch *ptr) { return ptr->cancel_build(); }
  bool GIGASearch_check_build(GIGASearch *ptr) { return ptr->check_build(); }
  double GIGASearch_num_search_ops(GIGASearch *ptr) { return ptr->get_num_search_ops(); }
  double GIGASearch_num_search_nodes(GIGASearch *ptr) { return ptr->get_num_search_nodes(); }
  double GIGASearch_num_build_ops(GIGASearch *ptr) { return ptr->get_num_build_ops(); }
}

GIGASearch::GIGASearch(double *data, unsigned int N, unsigned int D){
  //start a thread for building the tree
  this->N = N;
  this->D = D;
  this->data = data;
  num_build_ops = num_search_ops = num_search_nodes = 0.;
  xis = rs = NULL;
  cRs = cLs = nys = NULL;
  build_done = build_cancelled = search_done = false;
  build_thread = new std::thread(&GIGASearch::build, this);
  return;
}

GIGASearch::~GIGASearch(){ 
  //clean up data ptrs if the build thread didn't already do it (if the build was cancelled early)
  if (build_thread != NULL){ 
   cancel_build(); //if the build is already done, does nothing; if the build is in progress, cancels it
   build_thread->join(); //wait for the thread to terminate 
   delete build_thread;  //delete it
   build_thread = NULL; //set ptr to null
  }
  if (data != NULL){ data = NULL; } // don't delete ys, this is data ptr
  if (xis != NULL){ delete xis; xis = NULL; }
  if (rs != NULL){ delete  rs;  rs = NULL; }
  if (nys != NULL){ delete nys; nys = NULL; }
  if (cRs != NULL){ delete cRs; cRs = NULL; }
  if (cLs != NULL){ delete cLs; cLs = NULL; }
}

double GIGASearch::get_num_build_ops(){
  std::lock_guard<std::mutex> lk(build_mutex);
  return num_build_ops;
}

double GIGASearch::get_num_search_ops(){
  return num_search_ops;  
}

double GIGASearch::get_num_search_nodes(){
  return num_search_nodes;  
}


int GIGASearch::search(double *yw, double *y_yw){
  bool bdtmp = false;
  {
    std::lock_guard<std::mutex> lk(build_mutex);
    bdtmp = build_done;
  }
  if (bdtmp){
    search_done = false;
    auto tret = std::async(&GIGASearch::search_tree, this, yw, y_yw);
    auto lret = std::async(&GIGASearch::search_linear, this, yw, y_yw);
    auto nt = tret.get();
    auto nl = lret.get();
    return nt > nl ? nt : nl;
  } else {
    return search_linear(yw, y_yw);
  }
}

int GIGASearch::search_linear(double *yw, double *y_yw){
  double maxscore = -2;
  int nopt = -1;
  unsigned int check_interval = (unsigned int)log2(N);
  for (unsigned int n = 0; n < N; n++){
    double snum = 0., sdenom = 0.;
    for (unsigned int d = 0; d < D; d++){
      snum += data[n*D+d]*y_yw[d];
      sdenom += data[n*D+d]*yw[d];
    }
    if (sdenom > -1.+1e-14 && 1.-sdenom*sdenom > 0.){
      double score = snum/sqrt(1.-sdenom*sdenom);
      if (score > maxscore){
        maxscore = score;
        nopt = n;
      }
    }
    if (n % check_interval == 0){
      std::lock_guard<std::mutex> lk(search_mutex);
      if (search_done){
        return -1;
      }
    }
  }

  std::lock_guard<std::mutex> lk(search_mutex);
  search_done = true;
  num_search_ops += 2*N;
  num_search_nodes += N;
  return nopt;
}

int GIGASearch::search_tree(double *yw, double *y_yw){
  unsigned int nn = 1;
  double nf = 2.;
  double LB = -2.;
  int nopt = -1;
  int cur = -1;
  unsigned int check_interval = (unsigned int)log2(N);
  auto cmp = [](std::tuple<unsigned int, double> left, std::tuple<unsigned int, double> right){ return std::get<1>(left) < std::get<1>(right); };
  std::priority_queue<std::tuple<unsigned int, double>, std::vector< std::tuple<unsigned int, double> >, decltype(cmp)> search_queue(cmp);
  search_queue.push(std::make_tuple(0, upper_bound(0, yw, y_yw) ));
  while (!search_queue.empty()){
    if (nn % check_interval == 0){
      std::lock_guard<std::mutex> lk(search_mutex);
      if (search_done){
        return -1;
      }
    }
    //if we're done the depth traversal, pop a new node
    nn += 1;
    if (cur < 0){
      //pop next node
      auto tpl = search_queue.top();
      search_queue.pop();
      auto u = std::get<1>(tpl);
      if (u < LB){ //if upper bound is less than current best LB just toss this node out
        continue;
      }
      cur = std::get<0>(tpl); //otherwise start a new traversal from the node
    }
    //compute the LB
    double ell = lower_bound(cur, yw, y_yw);
    nf += 2.;
    //if its lb is greater than the current best, update the max LB and store data idx that achieved it
    if (ell > LB){
      LB = ell;
      nopt = nys[cur];
    }
    //if this node has children, push them onto the queue
    auto iR = cRs[cur];
    auto iL = cLs[cur];
    if (iR > -1){ 
      auto uR = upper_bound(iR, yw, y_yw);
      auto uL = upper_bound(iL, yw, y_yw);
      nf += 4.;
      //if either of the children is worth exploring (u > LB), set cur to that one and push the other onto the queue if it's worth exploring
      if (uR > uL && uR > LB){
        cur = iR;
        if (uL > LB){
          search_queue.push(std::make_tuple(iL, uL));
        }
      } else if (uL >= uR && uL > LB){
        cur = iL;
        if (uR > LB){
          search_queue.push(std::make_tuple(iR, uR));
        }
      } else { //if neither of the children are > LB, just pop a new node off the queue
        cur = -1;
      }
    } else {
      //this is a leaf node, pop a new one off the queue
      cur = -1;
    }
  }
  std::lock_guard<std::mutex> lk(search_mutex);
  search_done = true;
  num_search_ops += nf;
  num_search_nodes += nn;
  return nopt;
}

double GIGASearch::upper_bound(unsigned int node_idx, double *yw, double *y_yw){
    double bu, bv, b, rv, r1;
    bu = bv = b = rv = r1 = 0.;
    for (unsigned int d = 0; d < D; d++){
      bu += xis[node_idx*D+d]*y_yw[d];
      bv += xis[node_idx*D+d]*yw[d];
    }
    b = sqrt(std::max(0., 1. - bu*bu - bv*bv));
    rv = sqrt(std::max(0., rs[node_idx]*rs[node_idx] - bv*bv));
    r1 = sqrt(std::max(0., 1. - rs[node_idx]*rs[node_idx]));
    if (fabs(bv) > rs[node_idx] || bu >= rv){
      return 1.;
    } else {
      return (bu*rv+b*r1)/(b*b+bu*bu);
    }
}

double GIGASearch::lower_bound(unsigned int node_idx, double *yw, double *y_yw){
    double bu, bv;
    bu = bv = 0.;
    for (unsigned int d = 0; d < D; d++){ 
      bu += data[nys[node_idx]*D+d]*y_yw[d];
      bv += data[nys[node_idx]*D+d]*yw[d];
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

void GIGASearch::cancel_build(){
  //cancel the build (if build_done is already true, this will have no effect)
  std::lock_guard<std::mutex> lk(build_mutex);
  build_cancelled = true;
  return;
}

bool GIGASearch::check_build(){
  //return whether build is complete; lock_guard works here since return value is stored before unlock
  std::lock_guard<std::mutex> lk(build_mutex);
  return build_done;
}

void GIGASearch::build(){
  //initialize memory -- since tree is full/complete, it must have 2*N-1 nodes
  xis = new double[(2*N-1)*D];
  rs =  new double[2*N-1];
  nys = new int[2*N-1];
  cRs = new int[2*N-1];
  cLs = new int[2*N-1];

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
        xis[node_idx*D+d] = data[didx*D+d];
      }
      rs[node_idx] = 1.;
      nys[node_idx] = didx;
      cRs[node_idx] = cLs[node_idx] = -1;
      nbo += 1.;
      continue;
    }
    
    ////////////////////////////////
    //first get mean and store in xi
    ////////////////////////////////
    for (unsigned int d = 0; d < D; d++){
      xis[node_idx*D+d] = 0.;
    }
    for (auto& didx: data_idcs){
      for (unsigned int d = 0; d < D; d++){
        xis[node_idx*D+d] += data[didx*D+d]; 
      } 
    }
    double xinrmsq = 0.;
    for (unsigned int d = 0; d < D; d++){
      xinrmsq += xis[node_idx*D+d]*xis[node_idx*D+d];
    }
    //if xi has 0 norm, just set to 1st datapoint in list
    if (xinrmsq == 0.){
      auto& didx = data_idcs[0];
      for (unsigned int d = 0; d < D; d++){
        xis[node_idx*D+d] = data[didx*D+d]; 
      }
    } else {
      //otherwise, normalize the sum
      for (unsigned int d = 0; d < D; d++){
        xis[node_idx*D+d] /= sqrt(xinrmsq);
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
        dot += xis[node_idx*D+d]*data[didx*D+d];
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
    rs[node_idx] = dotmin;
    nys[node_idx] = cY;
    
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
    cRs[node_idx] = cur_idx;
    node_idx_queue.push(cur_idx++);

    data_idcs_queue.push(l_idcs);
    cLs[node_idx] = cur_idx;
    node_idx_queue.push(cur_idx++);
    

    /////////////////////////////////////////////////
    //if build gets cancelled, clean up & quit early
    /////////////////////////////////////////////////
    { 
      std::lock_guard<std::mutex> lk(build_mutex);
      if (build_cancelled){
        //dont set data to NULL in case build was cancelled but user wants to run linear search
        delete xis; xis = NULL;
        delete rs;   rs = NULL;
        delete nys; nys = NULL;
        delete cRs; cRs = NULL;
        delete cLs; cLs = NULL;
        return;
      }
    }
    nbo += 3. + 4.*data_idcs.size();
  }

  //acquire the lock on the build mutex, set build_done to true, unlock, and notify all waiting threads (possibly ::search)
  std::lock_guard<std::mutex> lk(build_mutex);
  build_done = true;
  num_build_ops = nbo;
  return;
}


///////////////////////
//old search/build code
///////////////////////
//void GIGASearch::build(double *data, unsigned int N, unsigned int D){
//  //initialize memory -- since tree is full/complete, it must have 2*N-1 nodes
//  ys = data;
//  xis = new double[(2*N-1)*D];
//  rs = new double[2*N-1];
//  nys = new int[2*N-1];
//  cRs = new int[2*N-1];
//  cLs = new int[2*N-1];
//  double * dir = new double[D];
//
//  double nbo = (double)N;
//
//  std::random_device rd;
//  std::mt19937 gen(rd());
//  
//  //top node has all the data in it; initialize index list with all idcs from 1 to N
//  std::vector< unsigned int > full_idcs(N);
//  std::iota(full_idcs.begin(), full_idcs.end(), 0);
//
//  //initialize queues for node construction
//  unsigned int cur_idx = 0;
//  std::queue< std::vector<unsigned int> > data_idcs_queue;
//  std::queue< unsigned int > node_idx_queue;
//  data_idcs_queue.push(full_idcs);
//  node_idx_queue.push(cur_idx++);
//
//  while (!data_idcs_queue.empty()){
//    //get next node construction job
//    auto data_idcs = data_idcs_queue.front();
//    data_idcs_queue.pop();
//    auto node_idx = node_idx_queue.front();
//    node_idx_queue.pop();
//
//    /////////////////////////
//    //node construction code
//    /////////////////////////
//
//
//    /////////////////////////////////////
//    //if only one idx, just store and quit
//    //////////////////////////////////////
//    if (data_idcs.size() == 1){
//      auto didx = data_idcs[0];
//      for (unsigned int d = 0; d < D; d++){
//        xis[node_idx*D+d] = data[didx*D+d];
//      }
//      rs[node_idx] = 1.;
//      nys[node_idx] = didx;
//      cRs[node_idx] = cLs[node_idx] = -1;
//      nbo += 1.;
//      continue;
//    }
//    
//    ////////////////////////////////
//    //first get mean and store in xi
//    ////////////////////////////////
//    for (unsigned int d = 0; d < D; d++){
//      xis[node_idx*D+d] = 0.;
//    }
//    for (auto& didx: data_idcs){
//      for (unsigned int d = 0; d < D; d++){
//        xis[node_idx*D+d] += data[didx*D+d]; 
//      } 
//    }
//    double xinrmsq = 0.;
//    for (unsigned int d = 0; d < D; d++){
//      xinrmsq += xis[node_idx*D+d]*xis[node_idx*D+d];
//    }
//    //if xi has 0 norm, just set to 1st datapoint in list
//    if (xinrmsq == 0.){
//      auto& didx = data_idcs[0];
//      for (unsigned int d = 0; d < D; d++){
//        xis[node_idx*D+d] = data[didx*D+d]; 
//      }
//    } else {
//      //otherwise, normalize the sum
//      for (unsigned int d = 0; d < D; d++){
//        xis[node_idx*D+d] /= sqrt(xinrmsq);
//      }
//    }
//
//    ////////////////////////////////////
//    //compute decision direction for children
//    //////////////////////////////////////
//    std::uniform_int_distribution<> dist(0, data_idcs.size()-1);
//    unsigned int nR, nL;
//    nR = dist(gen);
//    nL = nR;
//    while (nL == nR){
//      nL = dist(gen);
//    }
//    for (unsigned int d = 0; d < D; d++){
//      dir[d] = data[nR*D+d] - data[nL*D+d];
//    }
//
//    //////////////////////////////////
//    //find vec of min/max angle to xi, set r and y, and allocate idcs to children
//    //////////////////////////////////
//    double dotmin = 2.;
//    double dotmax = -2.;
//    unsigned int cY = 0;
//    std::vector<unsigned int> r_idcs, l_idcs;
//    for (auto& didx: data_idcs){
//      double dot = 0.;
//      double dotRL = 0.;
//      for (unsigned int d = 0; d < D; d++){
//        dot += xis[node_idx*D+d]*data[didx*D+d];
//        dotRL += dir[d]*data[didx*D+d];
//      }
//      if (dot < dotmin){
//        dotmin = dot;
//      }
//      if (dot > dotmax){
//        dotmax = dot; 
//        cY = didx;
//      }
//      if (dotRL > 0.){
//        r_idcs.push_back(didx);
//      } else {
//        l_idcs.push_back(didx);
//      }
//    }
//    //threshold the dotmin between -1 and 1 to avoid numerical issues
//    dotmin = dotmin > 1. ? 1. : dotmin; dotmin = dotmin < -1. ? -1. : dotmin;
//    rs[node_idx] = dotmin;
//    nys[node_idx] = cY;
//    
//    ///////////////////////////////////////////
//    //in pathological cases, one of r_idcs/l_idcs might be empty; here just allocate evenly
//    ///////////////////////////////////////////
//    if (r_idcs.empty() || l_idcs.empty()){
//      r_idcs.clear(); l_idcs.clear();
//      for (unsigned int i = 0; i < data_idcs.size()/2; i++){
//        r_idcs.push_back(data_idcs[i]);
//      }
//      for (unsigned int i = data_idcs.size()/2; i < data_idcs.size(); i++){
//        l_idcs.push_back(data_idcs[i]);
//      }
//    }
//
//    ///////////////////////////////////////////
//    //add the two child construction jobs to the queue
//    ///////////////////////////////////////////
//    data_idcs_queue.push(r_idcs);
//    cRs[node_idx] = cur_idx;
//    node_idx_queue.push(cur_idx++);
//
//    data_idcs_queue.push(l_idcs);
//    cLs[node_idx] = cur_idx;
//    node_idx_queue.push(cur_idx++);
//    
//
//    /////////////////////////////////////////////////
//    //if build gets cancelled, clean up & quit early
//    /////////////////////////////////////////////////
//    { 
//      std::lock_guard<std::mutex> lk(build_mutex);
//      if (build_cancelled){
//        ys = NULL; //don't delete ys, this is data ptr
//        delete xis; xis = NULL;
//        delete rs; rs = NULL;
//        delete nys; nys = NULL;
//        delete cRs; cRs = NULL;
//        delete cLs; cLs = NULL;
//        return;
//      }
//    }
//    nbo += 4. + 3.*data_idcs.size();
//  }
//
//  //acquire the lock on the build mutex, set build_done to true, unlock, and notify all waiting threads (possibly ::search)
//  std::unique_lock<std::mutex> lk(build_mutex);
//  build_done = true;
//  num_build_ops = nbo;
//  lk.unlock();
//  build_cv.notify_all();
//  return;
//}



//int GIGASearch::search(double *yw, double *y_yw, unsigned int D, unsigned int max_eval){
//  {
//    // if build not done yet & has not been cancelled already, wait on build mutex
//    std::unique_lock<std::mutex> lk(build_mutex);
//    if (!build_done && !build_cancelled){
//      build_cv.wait(lk);
//    }
//    //we reacquire the lock here; check if the build was cancelled & did not finish in the mean time
//    if (!build_done && build_cancelled){
//      return -1;
//    }
//    //otherwise, build is done and we can search
//  }
//
//  //search
//  double nn = 1.;
//  double nf = 2.;
//  double LB = -2.;
//  int nopt = -1;
//  auto cmp = [](std::tuple<unsigned int, double> left, std::tuple<unsigned int, double> right){ return std::get<1>(left) < std::get<1>(right); };
//  std::priority_queue<std::tuple<unsigned int, double>, std::vector< std::tuple<unsigned int, double> >, decltype(cmp)> search_queue(cmp);
//  search_queue.push(std::make_tuple(0, upper_bound(0, yw, y_yw, D) ));
//  while (!search_queue.empty()){
//    nn += 1.;
//    //pop next node
//    auto tpl = search_queue.top();
//    search_queue.pop();
//    auto u = std::get<1>(tpl);
//    if (u < LB){ //if upper bound is less than current best LB just toss this node out
//      continue;
//    }
//    auto cur = std::get<0>(tpl); //otherwise start a new traversal from the node
//
//    //compute the LB
//    double ell = lower_bound(cur, yw, y_yw, D);
//    nf += 2.;
//    //if its lb is greater than the current best, update the max LB and store data idx that achieved it
//    if (ell > LB){
//      LB = ell;
//      nopt = nys[cur];
//    }
//    //if this node has children, push them onto the queue if their upper bounds aren't too small
//    auto iR = cRs[cur];
//    auto iL = cLs[cur];
//    if (iR > -1){ 
//      auto uR = upper_bound(iR, yw, y_yw, D);
//      auto uL = upper_bound(iL, yw, y_yw, D);
//      nf += 4.;
//      if (uR > LB){
//        search_queue.push(std::make_tuple(iR, uR));
//      }
//      if (uL > LB){
//       search_queue.push(std::make_tuple(iL, uL));
//      }
//    }
//  }
//  num_search_nodes += nn;
//  num_search_ops += nf;
//  return nopt;
//}
