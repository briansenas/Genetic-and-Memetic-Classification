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

    if(argc<=8){
        cerr << "[ERROR] Couldn't resolve file name;" << endl;
        cerr << "[EXECUTION:] ./main (filename_directory) (label1)[char] (label2)[char] (seed)[-∞,∞] \n"
        << "\t\t(0=No/1=Shuffle/2=Balanced) (0=BLX/1=ARITHMETIC) (POP.SIZE)[0,∞] (0=No/1=LocalSearch) \n"
        << "\t\t{LOCALSEARCH OPTIONAL: (HowOften)[0,inf] (POP.Percentage)[0.0,1.0] (0=Random/1=OnlyBest)}" << endl;
        exit(-1);
    }
    bool debuggin = false;

    /// Leemos los datos de entrada como es el archivo, barajar, tipo de cruce...
    string filename = argv[1];
    char type1 = *argv[2];
    char type2 = *argv[3];
    long int seed = atoi(argv[4]);
    int shuffle = atoi(argv[5]);
    int CrossType = atoi(argv[6]);
    int Chromo = atoi(argv[7]);
    if(Chromo < 0)
        Chromo *= -1;
    int localSearch = atoi(argv[8]);
    // Default values:
    int Every = 10, perf = -1;
    float amount = 0.1;
    unsigned int localsize=ceil(amount*Chromo);
    if(localSearch==1 && argc>9){
        Every = atoi(argv[9]);
        if(argv[10] != NULL)
            amount = atof(argv[10]);
        if(argv[11] != NULL)
            perf = atoi(argv[11]);
    }
    if(localSearch==1){
        cout << "[WARNING] Using default values for localsearch \n";
        cout << "[LOCALSEARCH] Every:  " << Every << endl;
        cout << "[LOCALSEARCH]: SIZE: " << localsize << " - " << ((perf==1)?"SOLO MEJORES\n":"ALEATORIO\n");
    }

    vector<char> label;
    MatrixXd allData = readValues(filename,label);

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
    RowVectorXd Cross1(Chromo), Cross2(Chromo), stored_fit(2);
    long int evaluations = 0, max_evaluations = 15000;
    unsigned int i, size, Diversidad = 1,generation=0, nuevos;
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

    milliseconds tiempo;

    for(int x=0,folds=5;x<folds;x++){
        momentoInicio = high_resolution_clock::now();
        if(shuffle!=2)
            getFold(allData,label,data,Tlabel,test,Ttlabel,x);
        else
            getBalancedFold(group1,label_group1,group2,label_group2,data,Tlabel, test, Ttlabel,x,seed);

        // GET INITIAL FITNESS
        //Fitness = getFit(data,Tlabel, Solutions,0.5);
        getFit(data,Tlabel,Solutions,GenData,0.5);

        generation = evaluations = 0;
        while(evaluations < max_evaluations) {
            //shuffleFit(Solutions, Fitness,-1);
            shuffleFit(Solutions,GenData,-1);
            // GET BEST
            GenData.rowwise().sum().maxCoeff(&maxIndex);
            stored_fit = GenData.row(maxIndex);

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
                Diversidad = Random::get<unsigned>(1,data.cols()-1);
                for(unsigned int j=0;j<Diversidad;++j){
                    NewPopulation(i,Random::get<unsigned>(0,data.cols()-1)) += Random::get(distribution);
                }
                Rest = Rest - Diversidad;
            }

            // LOCALSEARCH
            if(localSearch==1 && generation%Every==0){
                getFit(data,Tlabel,NewPopulation, GenData, 0.5);
                if(perf!=1){
                    //shuffleFit(Solutions, Fitness,-1);
                    shuffleFit(NewPopulation,NewGenData,-1);
                    for(i=0,size=localsize;i<size;++i){
                        behaviour[0] = GenData(i,0);
                        behaviour[1] = GenData(i,1);
                        NewPopulation.row(i) = LocalSearch(data,Tlabel, NewPopulation.row(i),
                                eval_num, max_eval,maxTilBetter,behaviour,0.5);
                        GenData(i,0) = behaviour[0];
                        GenData(i,1) = behaviour[1];
                        evaluations += eval_num;
                    }
                }else{
                    // Do over the best only
                    getBest(NewGenData,indexGrid,localsize);
                    for(i=0;i<localsize;++i){
                        behaviour[0] = GenData(indexGrid[i],0);
                        behaviour[1] = GenData(indexGrid[i],1);
                        NewPopulation.row(indexGrid[i]) =
                            LocalSearch(data,Tlabel, NewPopulation.row(indexGrid[i]),
                                    eval_num, max_eval,maxTilBetter,behaviour,0.5);
                        GenData(indexGrid[i],0) = behaviour[0];
                        GenData(indexGrid[i],1) = behaviour[1];
                        evaluations += eval_num;
                    }
                }
            }else{
                //NewFitness = getFit(data,Tlabel,NewPopulation, NewGenData, 0.5);
                getFit(data,Tlabel,NewPopulation, GenData, 0.5);
            }

            evaluations += Solutions.rows();
            ++generation;

            // MAKE SURE THE OLD BEST IS IN THE NEXT POPULATION
            //NewFitness.minCoeff(&minIndex);
            //NewFitness(minIndex) = Fitness.maxCoeff(&maxIndex);
            GenData.rowwise().sum().minCoeff(&minIndex);
            GenData.row(minIndex) = stored_fit;
            NewPopulation.row(minIndex) = Solutions.row(maxIndex);

            Solutions = NewPopulation;
            //Fitness = NewFitness;
            //GenData = NewGenData;

            if(debuggin){
                GenData.rowwise().sum().maxCoeff(&maxIndex);
                GenData.rowwise().sum().minCoeff(&minIndex);
                cout << "###################################\n" ;
                cout << "[GENERATION NUMBER]: " << generation << "\n";
                cout << "[BEST FITNESS]: " << GenData(maxIndex,0) << "\t" << GenData(maxIndex,1) << "\t" << GenData.row(maxIndex).sum() << "\n";
                // cout << "[WORST FITNESS]: " << GenData(minIndex,0) << "\t" << GenData(minIndex,1) << "\t" << GenData.row(minIndex).sum() << "\n";
            }

        } // END WHILE

        momentoFin = high_resolution_clock::now();
        tiempo = duration_cast<milliseconds>(momentoFin - momentoInicio);
        getFit(test,Ttlabel, Solutions, GenData, 0.5);
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
