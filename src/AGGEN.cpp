#include "../inc/eigen-3.4.0/Eigen/Dense"
// From https://github.com/effolkronium/random
#include "../inc/random.hpp"

#include "../tools/mytools.h"
#include "../tools/Euclidean.h"
#include "../tools/ReadData.h"
#include "../tools/Genetics.h"
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

int main(int argc, char** argv){

    if(argc<=6){
        cerr << "[ERROR] Couldn't resolve file name;" << endl;
        cerr << "[SOLVE] Exec: ./main {filename} {seed}[-inf,inf] {No/Shuffle}[0,1]\n" <<
         "\t {BLX/ARITHMETIC}[0,1] {POP.SIZE}[0-30] {No/LocalSearch}[0,1] \n"  <<
         "\t [OPTIONAL:] {EveryWhen}[0,inf] {HowMany}[0,1] {Best/Worst}[0,1]" << endl;
        exit(-1);
    }

    /// Leemos los datos de entrada como es el archivo, barajar, tipo de cruce...
    string filename = argv[1];
    long int seed = atoi(argv[2]);
    int shuffle = atoi(argv[3]);
    int CrossType = atoi(argv[4]);
    int Chromo = atoi(argv[5]);
    int localSearch = atoi(argv[6]);
    // Default values:
    int Every = 10, perf = -1;
    float amount = 0.1;
    if(localSearch==1 && argc>7){
        Every = atoi(argv[7]);
        if(argv[8] != NULL)
            amount = atof(argv[8]);
        if(argv[9] != NULL)
            perf = atoi(argv[9]);
    }

    vector<char> label;
    MatrixXd allData = readValues(filename,label);

    /// Inicializamos todas las variables que vamos a necesitar para almacenar información
    if(shuffle==1){
        cout << "[WARNING]: Data has been shuffled; " << endl;
        shuffleData(allData,label,seed);
    }

    std::normal_distribution<double> distribution(0.0, sqrt(0.3));
    int cols = allData.cols(), pair;
    float P_c = 0.7, P_m = 0.1;
    vector<float> behaviour; behaviour.resize(2);
    int Cruzes = P_c * floor(Chromo/2.0);
    int Mutacion = P_m * (Chromo * cols), Rest;
    MatrixXd Solutions(Chromo, cols);
    MatrixXd NewPopulation(Chromo, cols), GenData(Chromo,2), NewGenData(Chromo,2);

    Solutions = (MatrixXd::Random(Chromo,cols) + MatrixXd::Constant(Chromo,cols,1))/2.0;
    // RowVectorXd Fitness(Chromo), NewFitness(Chromo);
    RowVectorXd Cross1(Chromo), Cross2(Chromo);
    long int evaluations = 0, max_evaluations = 15000;
    unsigned int i, size, Diversidad = 1,generation=0, nuevos;
    unsigned int maxTilBetter = 20*allData.cols(), eval_num=0,max_eval=15000, localsize=ceil(amount*Chromo);
    RowVectorXd::Index minIndex,maxIndex;
    vector<int> indexGrid;
    high_resolution_clock::time_point momentoInicio, momentoFin;

    momentoInicio = high_resolution_clock::now();
    milliseconds tiempo;

    // GET INITIAL FITNESS
    //Fitness = getFit(allData,label, Solutions,0.5);
    getFit(allData,label,Solutions,GenData,0.5);

    while(evaluations < max_evaluations) {
        //shuffleFit(Solutions, Fitness,-1);
        shuffleFit(Solutions,GenData,-1);

        // START CROSSING
        pair = 0;
        nuevos = 0;
        for(i=0,size=Cruzes;i<size;++i){
            if(CrossType == 0)
                BLXCross(Solutions.row(pair),Solutions.row(pair + 1),Cross1,Cross2);
            else
                ArithmeticCross(Solutions.row(pair),Solutions.row(pair + 1),Cross1,Cross2);
            NewPopulation.row(nuevos) = Cross1;
            NewPopulation.row(nuevos+1) = Cross2;
            //NewPopulation.row(nuevos+2) = (Fitness(pair)>Fitness(pair+1))? Solutions.row(pair) : Solutions.row(pair+1);
            NewPopulation.row(nuevos+2) =
                (GenData.row(pair).sum()>GenData.row(pair+1).sum())? Solutions.row(pair) : Solutions.row(pair+1);
            pair+=2;
            nuevos+=3;
        }

        Rest = Mutacion;
        while(Rest > 0){
            i = Random::get<unsigned>(0,Chromo-1);
            Diversidad = Random::get<unsigned>(1,allData.cols()-1);
            for(unsigned int j=0;j<Diversidad;++j){
                NewPopulation(i,Random::get<unsigned>(0,allData.cols()-1)) += Random::get(distribution);
            }
            Rest = Rest - Diversidad;
        }

        // LOCALSEARCH
        if(localSearch==1 && generation%Every==0){
            cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n";
            cout << "[LOCALSEARCH] GEN: " << generation << "\n";
            cout << "[PARAMETERS] SIZE: " << localsize << " - " << ((perf==1)?"SOLO MEJORES\n":"ALEATORIO\n");
            cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
            getFit(allData,label,NewPopulation, NewGenData, 0.5);
            if(perf!=1){
                //shuffleFit(Solutions, Fitness,-1);
                //shuffleFit(NewPopulation,GenData,-1);
                for(i=0,size=localsize;i<size;++i){
                    behaviour[0] = NewGenData(i,0);
                    behaviour[1] = NewGenData(i,1);
                    NewPopulation.row(i) = LocalSearch(allData,label, NewPopulation.row(i),
                            eval_num, max_eval,maxTilBetter,behaviour,0.5);
                    NewGenData(i,0) = behaviour[0];
                    NewGenData(i,1) = behaviour[1];
                    evaluations += eval_num;
                }
            }else{
                // Do over the best only
                getBest(NewGenData,indexGrid,localsize);
                for(i=0;i<localsize;++i){
                    behaviour[0] = NewGenData(indexGrid[i],0);
                    behaviour[1] = NewGenData(indexGrid[i],1);
                    NewPopulation.row(indexGrid[i]) =
                        LocalSearch(allData,label, NewPopulation.row(indexGrid[i]),
                                    eval_num, max_eval,maxTilBetter,behaviour,0.5);
                    NewGenData(indexGrid[i],0) = behaviour[0];
                    NewGenData(indexGrid[i],1) = behaviour[1];
                    evaluations += eval_num;
                }
            }
        }else{
            //NewFitness = getFit(allData,label,NewPopulation, NewGenData, 0.5);
            getFit(allData,label,NewPopulation, NewGenData, 0.5);
        }

        evaluations += Solutions.rows();
        ++generation;

        // MAKE SURE THE OLD BEST IS IN THE NEXT POPULATION
        //NewFitness.minCoeff(&minIndex);
        //NewFitness(minIndex) = Fitness.maxCoeff(&maxIndex);
        NewGenData.rowwise().sum().minCoeff(&minIndex);
        GenData.rowwise().sum().maxCoeff(&maxIndex);
        NewGenData.row(minIndex) = GenData.row(maxIndex);
        NewPopulation.row(minIndex) = Solutions.row(maxIndex);

        Solutions = NewPopulation;
        //Fitness = NewFitness;
        GenData = NewGenData;

        GenData.rowwise().sum().maxCoeff(&maxIndex);
        GenData.rowwise().sum().minCoeff(&minIndex);
        cout << "###################################\n" ;
        cout << "[GENERATION NUMBER]: " << generation << "\n";
        cout << "[BEST FITNESS]: " << GenData.row(maxIndex).sum() << " -> "
            << "Aciertos: " <<  GenData(maxIndex,0) << " + Reducción: " << GenData(maxIndex,1)  << "\n";
        cout << "[WORTS FITNESS]: " << GenData.row(minIndex).sum() << " -> "
            << "Aciertos: " << GenData(minIndex,0) << " + Reducción:" << GenData(minIndex,1)  << endl;

    } // END WHILE

    momentoFin = high_resolution_clock::now();
    tiempo = duration_cast<milliseconds>(momentoFin - momentoInicio);
    cout << "###################################\n" ;
    cout << "[GENERATION NUMBER]: " << generation << "\n";
    cout << "[BEST FITNESS]: " << GenData.row(maxIndex).sum() << " -> "
        << "Aciertos: " <<  GenData(maxIndex,0) << " + Reducción: " << GenData(maxIndex,1)  << "\n";
    cout << "[WORTS FITNESS]: " << GenData.row(minIndex).sum() << " -> "
        << "Aciertos: " << GenData(minIndex,0) << " + Reducción:" << GenData(minIndex,1)  << "\n";
    cout << "[TIME]: " << tiempo.count() << endl;
    return 0;
}
