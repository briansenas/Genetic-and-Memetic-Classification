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

    if(argc<=9){
        cerr << "[ERROR] Couldn't resolve file name;" << endl;
        cerr << "[EXECUTION:] ./main (filename_directory) (label1)[char] (label2)[char] (seed)[-∞,∞] \n"
        << "\t\t(0=No/1=Shuffle/2=Balanced) (0=BLX/1=ARITHMETIC) (POP.SIZE)[0,∞] (0=RandomKeepBestCross/1=TopKeepBestCross) (0=No/1=LocalSearch) \n"
        << "\t\t{LOCALSEARCH OPTIONAL: (HowOften)[0,inf] (POP.Percentage)[0.0,1.0] (0=RandomSearch/1=OnlyBestSearch)}" << endl;
        exit(-1);
    }
    bool debuggin = false;

    /// Leemos los datos de entrada como es el archivo, barajar, tipo de cruce...
    string filename = argv[1];
    char type1 = *argv[2];
    char type2 = *argv[3];
    long int seed = atoi(argv[4]);
    Random::seed(seed);
    srand(seed);
    int shuffle = atoi(argv[5]);
    int CrossType = atoi(argv[6]);
    int Chromo = atoi(argv[7]);
    if(Chromo < 0)
        Chromo *= -1;
    int bestOnly = atoi(argv[8]);
    int localSearch = atoi(argv[9]);
    // Default values:
    int Every = 10, perf = -1;
    float amount = 0.1;
    unsigned int localsize=ceil(amount*Chromo);
    if(localSearch==1 && argc>10){
        Every = atoi(argv[10]);
        if(argv[11] != NULL)
            amount = atof(argv[11]);
        if(argv[12] != NULL)
            perf = atoi(argv[12]);
    }
    if(localSearch==1){
        cout << "[WARNING] Using default values for localsearch \n";
        cout << "[LOCALSEARCH] Every:  " << Every << endl;
        cout << "[LOCALSEARCH]: SIZE: " << localsize << " - " << ((perf==1)?"SOLO MEJORES\n":"ALEATORIO\n");
    }

    vector<char> label;
    MatrixXd allData = readValues(filename,label);

    std::normal_distribution<double> distribution(0.0, sqrt(0.3));
    int cols = allData.cols();
    float P_c = 0.7, P_m = 0.1;
    vector<float> behaviour; behaviour.resize(2);
    int Cruzes = P_c * floor(Chromo/2.0);
    int Mutacion = P_m * (Chromo * cols);
    MatrixXd Solutions(Chromo, cols), FirstRandom(Chromo,cols);
    MatrixXd NewPopulation(Chromo, cols), GenData(Chromo,2), NewGenData(Chromo,2);

    FirstRandom = (MatrixXd::Random(Chromo,cols) + MatrixXd::Constant(Chromo,cols,1))/2.0;
    // RowVectorXd Fitness(Chromo), NewFitness(Chromo);
    RowVectorXd Cross1(Chromo), Cross2(Chromo), stored_fit(2);
    long int evaluations = 0, max_evaluations = 15000;
    unsigned int i, size, generation=0;
    unsigned int maxTilBetter = 20*allData.cols(), eval_num=0,max_eval=15000;
    RowVectorXd::Index minIndex,maxIndex;
    vector<int> indexGrid;
    high_resolution_clock::time_point momentoInicio, momentoFin;

    MatrixXd data, test, group1, group2;
    vector<char> Tlabel, Ttlabel, label_group1, label_group2;

    /// Inicializamos todas las variables que vamos a necesitar para almacenar información
    if(shuffle==1){
        cout << "[WARNING]: Data has been shuffled; " << endl;
        shuffleData(allData,label,seed);
    }
    if(shuffle==2){
        cout << "[WARNING]: Data has been balanced and shuffled (inevitably); " << endl;
        group1 = getClassLabelled(allData,label, label_group1, type1);
        group2 = getClassLabelled(allData,label, label_group2, type2);
    }

    MatrixXd *P1, *NP2, *swapper;
    milliseconds tiempo;

    for(int x=0,folds=5;x<folds;x++){
        momentoInicio = high_resolution_clock::now();
        if(shuffle!=2)
            getFold(allData,label,data,Tlabel,test,Ttlabel,x);
        else
            getBalancedFold(group1,label_group1,group2,label_group2,data,Tlabel, test, Ttlabel,x,seed);

        // GET INITIAL FITNESS
        //Solutions = FirstRandom;
        Solutions = (MatrixXd::Random(Chromo,cols) + MatrixXd::Constant(Chromo,cols,1))/2.0;
        P1 = &Solutions;
        NP2 = &NewPopulation;
        getFit(data,Tlabel,*P1,GenData,0.5);

        generation = evaluations = 0;
        while(evaluations < max_evaluations) {

            // GET BEST
            // START CROSSING
            if(bestOnly==1) {
                GenData.rowwise().sum().maxCoeff(&maxIndex);
                stored_fit = GenData.row(maxIndex);
                onlyBestCrossing(data,Tlabel,P1,NP2,GenData,CrossType,Cruzes,Mutacion);
            }else{
                shuffleFit(*P1,GenData,-1);
                GenData.rowwise().sum().maxCoeff(&maxIndex);
                stored_fit = GenData.row(maxIndex);
                randomCrossKeepBest(data,Tlabel,P1,NP2,GenData,CrossType,Cruzes,Mutacion);
            }

            // LOCALSEARCH
            if(localSearch==1 && generation%Every==0){
                if(perf!=1){
                    shuffleFit(*NP2,GenData,-1);
                    for(i=0,size=localsize;i<size;++i){
                        behaviour[0] = GenData(i,0);
                        behaviour[1] = GenData(i,1);
                        NP2->row(i) = LocalSearch(data,Tlabel, NP2->row(i),
                                eval_num, max_eval-evaluations,maxTilBetter,behaviour,0.5);
                        GenData(i,0) = behaviour[0];
                        GenData(i,1) = behaviour[1];
                        evaluations += eval_num;
                    }
                }else{
                    // Do over the best only
                    getBest(GenData,indexGrid,localsize);
                    for(i=0;i<localsize;++i){
                        behaviour[0] = GenData(indexGrid[i],0);
                        behaviour[1] = GenData(indexGrid[i],1);
                        NP2->row(indexGrid[i]) =
                            LocalSearch(data,Tlabel, NP2->row(indexGrid[i]),
                                    eval_num, max_eval-evaluations,maxTilBetter,behaviour,0.5);
                        GenData(indexGrid[i],0) = behaviour[0];
                        GenData(indexGrid[i],1) = behaviour[1];
                        evaluations += eval_num;
                    }
                }
            }

            evaluations += Solutions.rows();
            ++generation;

            // MAKE SURE THE OLD BEST IS IN THE NEXT POPULATION
            GenData.rowwise().sum().minCoeff(&minIndex);
            GenData.row(minIndex) = stored_fit;
            NP2->row(minIndex) = P1->row(maxIndex);

            swapper = P1;
            P1 = NP2;
            NP2 = swapper;

            if(debuggin){
                GenData.rowwise().sum().maxCoeff(&maxIndex);
                GenData.rowwise().sum().minCoeff(&minIndex);
                cout << "###################################\n" ;
                cout << "[GENERATION NUMBER]: " << generation << "\n";
                cout << "[BEST FITNESS]: " << GenData(maxIndex,0) << "\t" << GenData(maxIndex,1) << "\t" << GenData.row(maxIndex).sum() << "\n";
                 cout << "[WORST FITNESS]: " << GenData(minIndex,0) << "\t" << GenData(minIndex,1) << "\t" << GenData.row(minIndex).sum() << "\n";
            }

        } // END WHILE

        momentoFin = high_resolution_clock::now();
        tiempo = duration_cast<milliseconds>(momentoFin - momentoInicio);
        getFit(test,Ttlabel, *P1, GenData, 0.5);
        GenData.rowwise().sum().maxCoeff(&maxIndex);
        GenData.rowwise().sum().minCoeff(&minIndex);
        cout << "###################################\n" ;
        cout << "[GENERATION NUMBER]: " << generation << "\n";
        cout << "[BEST FITNESS]: " << GenData(maxIndex,0) << "\t" << GenData(maxIndex,1) << "\t" << GenData.row(maxIndex).sum() << "\n";
        // cout << "[WORST FITNESS]: " << GenData(minIndex,0) << "\t" << GenData(minIndex,1) << "\t" << GenData.row(minIndex).sum() << "\n";
        cout << "[TIME]: " << tiempo.count() << endl;

    } //END FOLD
    return 0;
}
