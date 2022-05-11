/**
 * @file Util_Genetics.cpp
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

void getReductRight(MatrixXd data, vector<char> Tlabel, RowVectorXd& Weights, unsigned int &right, unsigned int &reduct){
        unsigned int i,size, ManualNeighbour;
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
}


RowVectorXd getFit(MatrixXd data, vector<char> Tlabel, MatrixXd& Solutions,float alpha ){
    unsigned int ManualNeighbour, right=0,  reduct = 0;
    unsigned int i,j, size;
    RowVectorXd Weights;
    RowVectorXd results(Solutions.rows());

    for(j=0;j<Solutions.rows();j++){
        /// Truncamos los pesos a 0 y 1; Contamos las reducciones a 0.
        Weights = Solutions.row(j);
        getReductRight(data,Tlabel,Weights,right,reduct);
        Solutions.row(j) = Weights;
        results(j) = alpha*(float(right)/float(data.rows())) + (1-alpha)*(float(reduct)/float(Weights.cols()));
    }
    return results;
}

RowVectorXd getFit(MatrixXd data, vector<char> Tlabel, MatrixXd& Solutions, MatrixXd& GenData, float alpha ){
    unsigned int right=0,  reduct = 0;
    unsigned int j;
    RowVectorXd Weights;
    RowVectorXd results(Solutions.rows());
    if(GenData.rows()!=Solutions.rows() && GenData.cols()!=2)
        GenData.resize(Solutions.rows(),2);

    for(j=0;j<Solutions.rows();j++){
        /// Truncamos los pesos a 0 y 1; Contamos las reducciones a 0.
        Weights = Solutions.row(j);
        getReductRight(data,Tlabel,Weights,right,reduct);
        Solutions.row(j) = Weights;
        GenData(j,0) = alpha*(float(right)/float(data.rows()));
        GenData(j,1) = (1-alpha)*(float(reduct)/float(Weights.cols()));
        results(j) = GenData(j,0) + GenData(j,1);
    }
    return results;
}

RowVectorXd get1Fit(MatrixXd data, vector<char> Tlabel, RowVectorXd& Weights, float alpha ){
    unsigned int right=0,  reduct = 0;
    RowVectorXd results(2);
    unsigned int i,size, ManualNeighbour;
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
    results(0) = alpha*(float(right)/float(data.rows()));
    results(1) = (1-alpha)*(float(reduct)/float(data.cols()));
    return results;
}


RowVectorXd LocalSearch(MatrixXd allData,vector<char> label, RowVectorXd Weights,
unsigned int& eval_num, unsigned int max_eval, unsigned int maxTilBetter, vector<float>& fitness, float alpha){
    eval_num = 0;
    vector<int> indexGrid, Evaluations;
    fillRange(indexGrid,allData.cols());
    std::normal_distribution<double> distribution(0.0, sqrt(0.3));
    unsigned int i=0,size=0, tilBetter = 0, ran_num = 0, reduct, right, ManualNeighbour;
    RowVectorXd Weights_before(allData.cols());
    float function_after=0, function_before=0, reduction_rate, right_rate;
    if(fitness.size()<=1){
        fitness.clear();
        fitness.push_back(0);
        fitness.push_back(0);
    }else{
        function_before = fitness[0] + fitness[1];
    }
    while(eval_num < max_eval and tilBetter<maxTilBetter ){
        /// Modificamos el indice barajado correspondiente para modificar los pesos.
        Weights[indexGrid[ran_num++]] += Random::get(distribution);
        /// Si hemos modificados volvemos a barajar.
        if(ran_num>=indexGrid.size()){
            Random::shuffle(indexGrid);
            ran_num = 0;
        }

        getReductRight(allData,label,Weights,right,reduct);

        /// Evaluamos la función.
        reduction_rate = (1-alpha)*(float(reduct)/float(Weights.cols()));
        right_rate = alpha*(float(right)/float(allData.rows()));
        //function_after = alpha*(right/allData.rows()) + (1-alpha)*(reduct/Weights.cols());
        function_after = right_rate + reduction_rate;
        eval_num++;

        /// Verificamos si hemos mejorado, actuamos acorde.
        if(function_after <= function_before){
            tilBetter++;
            Weights = Weights_before;
        }else{
            fitness.at(0) = right_rate;
            fitness.at(1) = reduction_rate;
            tilBetter = 0;
            Weights_before = Weights;
            function_before = function_after;
            Random::shuffle(indexGrid);
            ran_num = 0;
        }

    } // END WHILE
    return Weights;
}
