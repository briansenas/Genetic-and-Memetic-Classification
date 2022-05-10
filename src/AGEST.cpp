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

    if(argc<=4){
        cerr << "[ERROR] Couldn't resolve file name;" << endl;
        cerr << "[SOLVE] Exec: ./main {filename} {seed} {No-Shuffle}[0-1] {BLX-ARITHMETIC}[0-1]" << endl;
        exit(-1);
    }

    /// Leemos los datos de entrada como es el archivo, clase1, clase2 y la semilla.
    string filename = argv[1];
    long int seed = atoi(argv[2]);
    int shuffle = atoi(argv[3]);
    int CrossType = atoi(argv[4]);
    vector<char> label;
    MatrixXd allData = readValues(filename,label);

    /// Inicializamos todas las variables que vamos a necesitar para almacenar información
    if(shuffle==1){
        cout << "[WARNING]: Data has been shuffled; " << endl;
        shuffleData(allData,label,seed);
    }

    std::normal_distribution<double> distribution(0.0, sqrt(0.3));
    int Chromo= 30, cols = allData.cols(), pair;
    float P_m = 0.1, min, max;
    int Mutacion = P_m * (Chromo * cols), Rest;
    MatrixXd Solutions(Chromo, cols), Descendents(2,cols);
    MatrixXd NewPopulation(Chromo, cols), GenData(Chromo,2), NewGenData(Chromo,2), DFit(2,2);

    Solutions = (MatrixXd::Random(Chromo,cols) + MatrixXd::Constant(Chromo,cols,1))/2.0;
    // RowVectorXd Fitness(Chromo), NewFitness(Chromo);
    RowVectorXd Cross1(Chromo), Cross2(Chromo);
    long int evaluations = 0, max_evaluations = 15000;
    unsigned int i, size, Diversidad = 1,generation=0, pair1, pair2;
    RowVectorXd::Index minIndex,maxIndex;
    vector<int> indexGrid;
    high_resolution_clock::time_point momentoInicio, momentoFin;

    momentoInicio = high_resolution_clock::now();
    milliseconds tiempo;

    // GET INITIAL FITNESS
    //Fitness = getFit(allData,label, Solutions,0.5);
    getFit(allData,label,Solutions,GenData,0.5);

    while(evaluations < max_evaluations){
        //shuffleFit(Solutions, Fitness,-1);
        shuffleFit(Solutions,GenData,-1);

        // START CROSSING
        do{
            pair1 = Random::get<unsigned>(0,Solutions.rows()-1);
            pair2 = Random::get<unsigned>(0,Solutions.rows()-1);
        }while(pair1==pair2);

        if(CrossType == 0)
            BLXCross(Solutions.row(pair),Solutions.row(pair + 1),Cross1,Cross2);
        else
            ArithmeticCross(Solutions.row(pair),Solutions.row(pair + 1),Cross1,Cross2);

        pair1 = Random::get(0,1);
        if(pair1<0.1){
            Diversidad = Random::get<unsigned>(0,allData.cols()-1);
            for(i=0,size=Diversidad;i<size;++i){
                Cross1(Random::get<unsigned>(0,allData.cols()-1)) += Random::get(distribution);
            }
        }
        pair2 = Random::get(0,1);
        if(pair1<0.1){
            Diversidad = Random::get<unsigned>(0,allData.cols()-1);
            for(i=0,size=Diversidad;i<size;++i){
                Cross1(Random::get<unsigned>(0,allData.cols()-1)) += Random::get(distribution);
            }
        }
        Descendents.row(0) = Cross1;
        Descendents.row(1) = Cross1;

        getFit(allData,label,Descendents,DFit,0.5);


        // COMPETE TO GET BACK IN POPULATION
        min = GenData.rowwise().sum().minCoeff(&minIndex);
        max = DFit.rowwise().sum().maxCoeff(&maxIndex);
        // Interchange minimun with maximun new
        if(min < max) {
            GenData.row(minIndex) = DFit.row(maxIndex);  //Add new value
            Solutions.row(minIndex) = Descendents.row(maxIndex);  // Interchange parameters
        }
        min = GenData.rowwise().sum().minCoeff(&minIndex);  // New minimun?¿=?¿
        max = DFit.row((maxIndex+1)%2).sum();
        if(min < max){
            GenData.row(minIndex) = DFit.row(maxIndex);  //Add new value
            Solutions.row(minIndex) = Descendents.row(maxIndex);  // Interchange parameters
        }

        evaluations += 2;
        ++generation;

        GenData.rowwise().sum().maxCoeff(&maxIndex);
        GenData.rowwise().sum().minCoeff(&minIndex);

        cout << "###################################\n" ;
        cout << "[GENERATION NUMBER]: " << generation << "\n";
        cout << "[BEST FITNESS]: " << GenData.row(maxIndex).sum() << " -> "
            << "Aciertos: " <<  GenData(maxIndex,0) << " + Reducción: " << GenData(maxIndex,1)  << "\n";
        cout << "[WORTS FITNESS]: " << GenData.row(minIndex).sum() << " -> "
            << "Aciertos: " << GenData(minIndex,0) << " + Reducción:" << GenData(minIndex,1)  << "\n";

    }
    momentoFin = high_resolution_clock::now();
    tiempo = duration_cast<milliseconds>(momentoFin - momentoInicio);
    cout << "###################################\n" ;
    cout << "[GENERATION NUMBER]: " << generation << "\n";
    cout << "[BEST FITNESS]: " << GenData.row(maxIndex).sum() << " -> "
        << "Aciertos: " <<  GenData(maxIndex,0) << " + Reducción: " << GenData(maxIndex,1)  << "\n";
    cout << "[WORTS FITNESS]: " << GenData.row(minIndex).sum() << " -> "
        << "Aciertos: " << GenData(minIndex,0) << " + Reducción:" << GenData(minIndex,1)  << "\n";
    return 0;
}
