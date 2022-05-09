/**
 * @file Genetics.cpp
 * @version 1.0
 * @date 09/05/2022
 * @author Brian Sena Simons 3ºA-A2
 * @brief Funciones diferentes que pueda necesitar para el desarrollo de los
 * algoritmos geneticos
 * @code
 * [...]
    for(i=0,size=Cruzes;i<size;++i){
        BLXCross(Solutions.row(pair),Solutions.row(pair + 1),Cross1,Cross2);
        NewPoblation.row(nuevos) = Cross1;
        NewPoblation.row(nuevos+1) = Cross2;
        NewPoblation.row(nuevos+2) = (Fitness(pair)>Fitness(pair+1))? Solutions.row(pair) : Solutions.row(pair+1);
        pair+=2;
        nuevos+=3;
    }
 *[...]
 * @endcode
 **/
#include "../tools/Genetics.h"
#include "../inc/eigen-3.4.0/Eigen/Dense"
#include "../inc/random.hpp"
#include <fstream>
#include <math.h>
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace std::chrono;
using Random = effolkronium::random_static;

void ArithmeticCross(RowVectorXd parent1, RowVectorXd parent2, RowVectorXd& res1, RowVectorXd& res2, long int seed){
    if(seed!=-1)
        Random::seed(seed);

    if(res1.cols() != parent1.cols())
        res1.resize(parent1.cols());
    if(res2.cols() != parent1.cols())
        res2.resize(parent2.cols());
    res1 = (parent1+parent2)*Random::get(0,1);
    res2 = (parent1+parent2)*Random::get(0,1);
}

void BLXCross(RowVectorXd parent1, RowVectorXd parent2,RowVectorXd& res1, RowVectorXd& res2, float alpha, long int seed){
    if(seed!=-1)
        Random::seed(seed);

    float Cmin, Cmax, Distance, LO, HI;

    if(res1.cols() != parent1.cols())
        res1.resize(parent1.cols());
    if(res2.cols() != parent1.cols())
        res2.resize(parent2.cols());

    for(unsigned int i=0;i < parent1.cols(); i++){
        Cmin = min(parent1(i),parent2(i));
        Cmax = max(parent1(i),parent2(i));
        Distance = Cmax - Cmin;
        LO = Cmin - Distance * alpha;
        HI = Cmax + Distance * alpha;

        res1(i) = Random::get(LO,HI);
        res2(i) = Random::get(LO,HI);
    }
}

RowVectorXd getFit(MatrixXd data, vector<char> Tlabel, MatrixXd& Solutions,float alpha ){
    unsigned int ManualNeighbour, right=0,  reduct = 0;
    unsigned int i,j, size;
    RowVectorXd Weights;
    RowVectorXd results(Solutions.rows());

    for(j=0;j<Solutions.rows();j++){
        /// Truncamos los pesos a 0 y 1; Contamos las reducciones a 0.
        Weights = Solutions.row(j);
        reduct = 0;
        for(i=0,size=Weights.cols();i<size;i++){
            if(*(Weights.data() + i) < 0.1){
                *(Weights.data() + i) = 0;
                reduct++;
            }
            else if(*(Weights.data()+i)>1)
                *(Weights.data()+i) = 1;
        }
        /// Verificamos el porcentaje de acierto obtenido con la modificación.
        right = 0;
        for(i=0,size=data.rows();i<size;i++){
            ManualEuclideanDistance(Weights,data.row(i),data,i, ManualNeighbour);
            if(Tlabel[i] == Tlabel[ManualNeighbour])
                right++;
        }

        Solutions.row(j) = Weights;
        results(j) = alpha*(float(right)/float(data.rows())) + (1-alpha)*(float(reduct)/float(Weights.cols()));
    }
    return results;
}

RowVectorXd getFit(MatrixXd data, vector<char> Tlabel, MatrixXd& Solutions,MatrixXd& GenData, float alpha ){
    unsigned int ManualNeighbour, right=0,  reduct = 0;
    unsigned int i,j, size;
    RowVectorXd Weights;
    RowVectorXd results(Solutions.rows());
    if(GenData.rows()!=Solutions.rows() && GenData.cols()!=2)
        GenData.resize(Solutions.rows(),2);

    for(j=0;j<Solutions.rows();j++){
        /// Truncamos los pesos a 0 y 1; Contamos las reducciones a 0.
        Weights = Solutions.row(j);
        reduct = 0;
        for(i=0,size=Weights.cols();i<size;i++){
            if(*(Weights.data() + i) < 0.1){
                *(Weights.data() + i) = 0;
                reduct++;
            }
            else if(*(Weights.data()+i)>1)
                *(Weights.data()+i) = 1;
        }
        /// Verificamos el porcentaje de acierto obtenido con la modificación.
        right = 0;
        for(i=0,size=data.rows();i<size;i++){
            ManualEuclideanDistance(Weights,data.row(i),data,i, ManualNeighbour);
            if(Tlabel[i] == Tlabel[ManualNeighbour])
                right++;
        }

        Solutions.row(j) = Weights;
        GenData(j,0) = alpha*(float(right)/float(data.rows()));
        GenData(j,1) = (1-alpha)*(float(reduct)/float(Weights.cols()));
        results(j) = GenData(j,0) + GenData(j,1);
    }
    return results;
}
