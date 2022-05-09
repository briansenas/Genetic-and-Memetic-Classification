#include "../tools/eigen-3.4.0/Eigen/Dense"
// From https://github.com/effolkronium/random
#include "../tools/random.hpp"

#include "../tools/mytools.h"
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <ctime>

using namespace std;
using namespace Eigen;
using namespace std::chrono;
using Random = effolkronium::random_static;

void cruceAritmetico(RowVectorXd parent1, RowVectorXd parent2, RowVectorXd& res1, RowVectorXd& res2){
    if(res1.cols() != parent1.cols())
        res1.resize(parent1.cols());
    if(res2.cols() != parent1.cols())
        res2.resize(parent2.cols());
    res1 = (parent1+parent2)*0.5;
    res2 = (parent1+parent2)*0.5;
}

void fillRange(vector<int>& toFill,unsigned int upperlimit){
    toFill.clear();
    for(unsigned int i=0;i<upperlimit;i++)
        toFill.push_back(i);
}

void shuffleFit(MatrixXd& mat,RowVectorXd& fitness, long int seed){
    Random::seed(seed);
    vector<int> indexes;
    fillRange(indexes,mat.rows());
    Random::shuffle(indexes);
    Transpositions<Dynamic, Dynamic> tr;
    tr.resize(mat.rows());
    float var = -1;

    for(unsigned int i=0;i<indexes.size();i++){
        var = fitness(indexes.at(i));
        fitness(indexes.at(i)) = fitness(i);
        fitness(i) = var ;
        tr[i] = indexes.at(i);
    }

    MatrixXd temp = tr * mat;
    mat = temp;
}
void getBest(RowVectorXd Fitness,vector<int>& indexGrid,unsigned int size){
    indexGrid.clear();
    vector<int> values, sorted;
    for(unsigned int i=0;i<Fitness.cols();i++){
        values.push_back(Fitness(i));
        sorted.push_back(Fitness(i));
    }
    sort(sorted.begin(),sorted.end());
    vector<int>::iterator index;
    for(unsigned int i=0;i<size;++i){
        index = find(values.begin(),values.end(),sorted[i]);
        indexGrid.push_back(int(index-values.begin()));
    }
}

int main(int argc, char** argv){
    int fila = 3, columna = 2;
    MatrixXd solutions(fila,columna);
    solutions << 1, 1,
                 2, 2,
                 3, 3;

    RowVectorXd fitness(fila);
    fitness << 10, 20, 30;

    for(unsigned int i=0,size=fila;i<size;i++){
        cout << solutions(i,0) << " - " << solutions(i,1) << " = " << fitness(i) << endl;
    }

    cout << " @@@@@@@@@@@@@@@@@ " << endl;

    shuffleFit(solutions, fitness, atoi(argv[1]));

    for(unsigned int i=0,size=fila;i<size;i++){
        cout << solutions(i,0) << " - " << solutions(i,1) << " = " << fitness(i) << endl;
    }
    vector<int> indexGrid;
    getBest(fitness, indexGrid,3);

    for(unsigned int i=0;i<indexGrid.size();i++){
        cout << indexGrid[i] << endl;
    }
    return 0;
}
