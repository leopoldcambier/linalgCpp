/** 
 * BUILD:
 * To build mfchol, making sure to 
 *  - Include mmio, eigen, scotch and openblas
 *  - Link with -lscotch -lscotcherr -lblas -llapack
 *      g++ -Wextra -Wall --std=c++11 -O3 -o mfchol mfchol.cpp -I(...) -lscotch -lscotcherr -lblas -llapack
 *  On Sherlock, something like this should be enough
 *      module load scotch eigen openblas
 *      g++ -Wextra -Wall --std=c++11 -O3 -o mfchol mfchol.cpp -I ../mmio/ -I $CPATH -L $LIBRARY_PATH  
 *         -lscotch -lscotcherr -lblas -llapack -lz
 *
 * RUN:
 *  ./mfchol input_file.mm nlevels output_ordering.ord output_neighbors.nbr
 * Where input_file.mm is a sparse SPD matrix in matrix-market format
 * 
 * Example for a 5x5 laplacians on a regular grid, we get
 * ./mfchol /path/to/neglapl_2_5.mm 2 ordering.ord nbr.nbr
 * cat ordering.ord
        2 3
        0 0 10 0 1 2 3 5 6 7 10 11 15
        0 1 10 9 13 14 17 18 19 21 22 23 24
        1 0 5 4 8 12 16 20
 * cat nbr.nbr
        2 3
        0 0 5 4 8 12 16 20
        0 1 5 4 8 12 16 20
        1 0 0
 * First line is 
 *      #levels #clusters(total)
 * Then each line is
 *      level sepid numberofentries n0 n1 n2...
 * where sepid is the separator id at the given levels
 * 
 * Example: each node is level|sepid
 *             20
 *        10        11
 *      00  01    02  03
**/

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SparseLU> 

#include <scotch.h>

#include <fstream>
#include <iostream>
#include <set>
#include <array>
#include <random>
#include <vector>
#include <algorithm>
#include <stdio.h>

#include <cblas.h>
#include <lapacke.h>

#include "mmio.hpp"

using namespace std;
using namespace Eigen;

typedef SparseMatrix<double> SpMat;

struct Front {
    int id;
    int start;
    int self_size;
    int nbr_size;
    vector<int> rows;
    vector<Front*> childrens;
    Front* parent;
    MatrixXd* front;
    Front(int id_, int start_, int size_) : id(id_), start(start_), self_size(size_) {
        nbr_size = 0;
        childrens = vector<Front*>(0);  
        parent = nullptr;      
        rows = vector<int>(0);
        front = nullptr;
    };
    void extend_add() {
        for(Front* c: childrens) {
            int cs = c->self_size;
            int cn = c->nbr_size;
            vector<int> subids(cn);
            int l = 0;
            for(int i = 0; i < cn; i++) {
                while (this->rows[l] != c->rows[cs + i]) { l++; }
                subids[i] = l;
            }
            for(int j = 0; j < c->nbr_size; j++) {
                int fj = subids[j];
                for(int i = 0; i < c->nbr_size; i++) {
                    int fi = subids[i];
                    (*this->front)(fi,fj) += (*c->front)(cs + i, cs + j);                    
                }
            }
        }
    };
    void factor() {
        int lda = front->rows();
        double* Ass = front->data();
        double* Ans = front->data() + self_size;
        double* Ann = front->data() + self_size * lda + self_size;                
        // POTRF
        int err = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', self_size, Ass, lda);
        assert(err == 0);
        // TRSM
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, nbr_size, self_size, 1.0, Ass, lda, Ans, lda);
        // SYRK
        cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, nbr_size, self_size, -1.0, Ans, lda, 1.0, Ann, lda);        
    }      
    void fwd(VectorXd& xglob) {     
        double* Lss = front->data();
        double* Lns = front->data() + self_size;
        auto xs = xglob.segment(start, self_size);
        VectorXd xn = VectorXd::Zero(nbr_size);
        // x[s] <- Lss^-1 x[s]
        cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, self_size, Lss, front->rows(), xs.data(), 1);
        // xn = -Lns x[s]
        cblas_dgemv(CblasColMajor, CblasNoTrans, nbr_size, self_size, -1.0, Lns, front->rows(), xs.data(), 1, 0.0, xn.data(), 1); // beta = 0.0     
        // x[n] = xn
        for(int i = 0; i < nbr_size; i++) {
            xglob[rows[self_size + i]] += xn[i]; // ~ beta = 1.0
        }
    }
    void bwd(VectorXd& xglob) {        
        double* Lss = front->data();        
        double* Lns = front->data() + self_size;
        auto xs = xglob.segment(start, self_size);
        VectorXd xn = VectorXd::Zero(nbr_size);
        // xn = x[n]
        for(int i = 0; i < nbr_size; i++) {
            xn[i] = xglob[rows[self_size + i]];
        }
        // x[s] -= Lns^T xn
        cblas_dgemv(CblasColMajor, CblasTrans, nbr_size, self_size, -1.0, Lns, front->rows(), xn.data(), 1, 1.0, xs.data(), 1); // beta = 1.0     
        // xs = Lss^-T xs
        cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, self_size, Lss, front->rows(), xs.data(), 1);
    }
    void extract(MatrixXd& A) {
        for(int j = 0; j < front->cols(); j++) {
            for(int i = 0; i < front->rows(); i++) {
                int Ai = rows[i];
                int Aj = rows[j];
                A(Ai,Aj) = (*front)(i,j);
            }
        }
    }
};

struct MF {
    vector<Front*> fronts;
    VectorXi perm;
    VectorXi invperm;
    MF(vector<Front*> fronts_, VectorXi perm_, VectorXi invperm_) : fronts(fronts_), perm(perm_), invperm(invperm_) {};
    void factorize() { 
        for(auto f: fronts) // In this code, the fronts are ordered properly, so we can just go through, but normally this has to be done bit more carefully
        {
            f->extend_add();
            f->factor();
        }
    }
    VectorXd solve(VectorXd& b) {
        VectorXd xglob = perm.asPermutation() * b;
        for(int b = 0; b < int(fronts.size()); b++) {
            fronts[b]->fwd(xglob);
        }
        for(int b = int(fronts.size()) - 1; b >= 0; b--) {            
            fronts[b]->bwd(xglob);        
        }
        return perm.asPermutation().transpose() * xglob;
    }
    void extract(MatrixXd& A) {
        for(auto f : fronts) {
            f->extract(A);
        }
    }
    void print_ordering(string ordering_file, string neighbors_file, string tree_file) {
        int nfronts = fronts.size();
        
        ofstream ord_file;
        ord_file.open(ordering_file);
        ord_file << nfronts << "\n";

        ofstream nbr_file;
        nbr_file.open(neighbors_file);
        nbr_file << nfronts << "\n";

        ofstream t_file;
        t_file.open(tree_file);
        t_file << nfronts << "\n";

        /** Prints:
         *
         * ordering file:
         * 1st line: 
         * <number of separators>
         * Each other line: 
         * <separator id> <number of node> <self 0> <self 1> ...
         *
         * neighbors file:
         * 1st line: 
         * <number of separators>
         * Each other line: 
         * <separator id> <number of neighbors> <neighbor 0> <neighbor 1> ...
         *
         * tree file:
         * 1st line:
         * <number of separators>
         * Each other line:
         * <separator id> <number of children> <parent id> <children 1 id> <children 2 id> ...
         * <parent id> is (-1) if the separator as no parent
         */


        for(int fi = 0; fi < nfronts; fi++) {            
            Front* f = fronts[fi];
            assert(fi == f->id);
            
            ord_file << fi << " " << f->self_size;
            for(int ii = 0; ii < f->self_size; ii++) {
                assert(f->rows[ii] == f->start + ii);
                ord_file << " " << invperm[f->rows[ii]]; // Self
            }
            ord_file << "\n";

            nbr_file << fi << " " << f->nbr_size;
            for(int ii = f->self_size; ii < f->self_size + f->nbr_size; ii++) {
                nbr_file << " " << invperm[f->rows[ii]]; // Nbrs
            } 
            nbr_file << "\n";

            t_file << fi << " " << (f->parent == nullptr ? -1 : f->parent->id) << " " << f->childrens.size();
            for(int c = 0 ; c < f->childrens.size(); c++) {
                t_file << " " << f->childrens[c]->id;
            }
            t_file << "\n";
        }
    }
};

MF initialize(SpMat& A, int nlevels) {
    int N = A.rows();
    int nnz = A.nonZeros();
    // Create rowval and colptr
    VectorXi rowval(nnz);
    VectorXi colptr(N+1);
    int k = 0;
    colptr[0] = 0;
    for(int j = 0; j < N; j++) {
        for (SpMat::InnerIterator it(A,j); it; ++it) {
            int i = it.row();
            if(i != j) {
                rowval[k] = i;
                k++;
            }
        }
        colptr[j+1] = k;
    }
    // Create SCOTCH graph
    SCOTCH_Graph* graph = SCOTCH_graphAlloc();    
    int err = SCOTCH_graphInit(graph);
    assert(err == 0);
    err = SCOTCH_graphBuild(graph, 0, N, colptr.data(), nullptr, nullptr, nullptr, k, rowval.data(), nullptr);
    assert(err == 0);
    err = SCOTCH_graphCheck(graph);
    assert(err == 0);
    cout << "Graph build OK" << endl;
    // Create strat
    SCOTCH_Strat* strat = SCOTCH_stratAlloc();
    err = SCOTCH_stratInit(strat);
    assert(err == 0);
    assert(nlevels > 0);
    string orderingstr = "n{sep=(/levl<" + to_string(nlevels-1) + "?g:z;)}";
    err = SCOTCH_stratGraphOrder(strat, orderingstr.c_str());
    assert(err == 0);
    cout << "Strat OK using " << orderingstr << endl;
    // Order with SCOTCH
    int nblk = 0;
    VectorXi permtab(N);
    VectorXi peritab(N);
    VectorXi rangtab(N+1);
    VectorXi treetab(N);
    err = SCOTCH_graphOrder(graph, strat, permtab.data(), peritab.data(), &nblk, rangtab.data(), treetab.data());
    assert(err == 0);
    cout << "Scotch ordering OK" << endl;
    auto P = permtab.asPermutation();
    // Permute matrix
    SpMat App = P * A * P.transpose();
    // Create all the fronts
    vector<Front*> fronts;
    for(int b = 0; b < nblk; b++) {
        int start = rangtab[b];;
        int self_size = rangtab[b+1] - rangtab[b];
        Front* f = new Front(b, start, self_size);
        fronts.push_back(f);
    }
    // Set the children & parent
    treetab.conservativeResize(nblk);
    cout << treetab.transpose() << endl;
    for(int b = 0; b < nblk; b++) {
        int p = treetab[b];
        assert(p == -1 || p > b); // Important for the following loop and for everything else. Leaves come first.
        if(p != -1) {
            fronts[p]->childrens.push_back(fronts[b]);
            fronts[b]->parent = fronts[p];
        }
    }
    // Fill and allocate everything
    for(int b = 0; b < nblk; b++) {        
        Front* f = fronts[b];
        int start = f->start;
        int self_size = f->self_size;
        int end = start + self_size;
        set<int> rows_;
        // Self rows
        for(int j = start; j < end; j++) {
            rows_.insert(j);
            for (SpMat::InnerIterator it(App,j); it; ++it) {
                int i = it.row();
                if(i >= start) rows_.insert(i);            
            }
        }
        // Add children
        for(auto c : f->childrens) {
            for(auto i : c->rows) {
                if(i >= start) rows_.insert(i);
            }
        }
        // Create vector and sort
        vector<int> rows(rows_.begin(), rows_.end());
        sort(rows.begin(), rows.end());
        // Allocate and fill front
        MatrixXd* front = new MatrixXd(rows.size(), rows.size());
        *front = MatrixXd::Zero(rows.size(), rows.size());
        for(int j = start; j < end; j++) {
            for (SpMat::InnerIterator it(App,j); it; ++it) {
                int i = it.row();              
                int j = it.col();  
                double v = it.value();
                if(i >= start && i >= j) {
                    auto row = lower_bound(rows.begin(), rows.end(), i);
                    assert(row != rows.end());
                    int fi = distance(rows.begin(), row);
                    int fj = j - start;
                    (*front)(fi,fj) = v;    
                }
            }
        }
        // Record        
        f->nbr_size = rows.size() - self_size;
        f->rows = rows;
        f->front = front;
    }
    // FIXME Need to free SCOTCH structures
    return MF(fronts, permtab, peritab);
}

int main(int argc, char** argv) {
    int nlevels = 5;
    string filename = "neglapl_2_128.mm";
    string ordering = "ordering.ord";
    string nbrs     = "nbrs.nbr";
    string tree     = "tree.tree";
    if(argc >= 2) {
        filename = argv[1];
    }
    if(argc >= 3) {
        nlevels = atoi(argv[2]);
    }
    if(argc >= 4) {
        ordering = argv[3];
    }
    if(argc >= 5) {
        nbrs = argv[4];
    }
    if(argc >= 6) {
        tree = argv[5];
    }
    // Read matrix
    cout << "Matrix file " << filename << endl;
    SpMat A = mmio::sp_mmread<double,int>(filename);
    VectorXd b = VectorXd::Random(A.rows());
    // Init
    cout << "Initializing with " << nlevels << " levels" << endl;
    MF mf = initialize(A, nlevels);
    cout << "Ordering file written to " << ordering << endl;
    cout << "Nbrs file written to " << nbrs << endl;
    cout << "Tree file written to " << tree << endl;
    mf.print_ordering(ordering, nbrs, tree);
    // Factor
    cout << "Factoring..." << endl;
    mf.factorize();
    // Solve
    cout << "Solve..." << endl;
    VectorXd x = mf.solve(b);
    // Check residual
    cout << "|Ax-b|/|b| " << (A*x - b).norm() / b.norm() << endl;
}