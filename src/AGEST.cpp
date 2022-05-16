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
#include <fstream>
#include <unistd.h>

using namespace std;
using namespace Eigen;
using namespace std::chrono;
using Random = effolkronium::random_static;

int main(int argc, char** argv){

    if(argc<=9){
        cerr << "[ERROR] Couldn't resolve file name;" << endl;
        cerr << "[EXECUTION:] ./main (filename_directory) (label1)[char] (label2)[char] (0=Print/1=WriteFile/2=Write+Plot_data) (seed)[-∞,∞] \n"
        << "\t\t(0=No/1=Shuffle/2=Balanced) (0=BLX/1=ARITHMETIC) (POP.SIZE)[0,∞] (0=No/1=LocalSearch) \n"
        << "\t\t{LOCALSEARCH OPTIONAL: (HowOften)[0,inf] (POP.Percentage)[0.0,1.0] (0=Random/1=OnlyBest)}" << endl;
        exit(-1);
    }
    bool debuggin = false;

    /// Leemos los datos de entrada como es el archivo, barajar, tipo de cruce...
    string filename = argv[1];
    char type1 = *argv[2];
    char type2 = *argv[3];
    int streambus = atoi(argv[4]);
    long long int seed = stoll(argv[5]);
    int shuffle = atoi(argv[6]);
    int CrossType = atoi(argv[7]);
    int Chromo = atoi(argv[8]);
    if(Chromo < 0)
        Chromo *= -1;
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
    bool printing = (streambus>=1)?false:true;
    bool plotting = false;
    ofstream plot,myfile;
    string writefile = "", plot_path = "", output;
    string path = get_selfpath();
    path = path.substr(0,path.find_last_of("/\\") + 1);
    if(streambus>=1) {
        //https://www.codegrepper.com/code-examples/cpp/c%2B%2B+get+filename+from+path
        // get filename
        std::string base_filename = filename.substr(filename.find_last_of("/\\") + 1);
        // remove extension from filename
        std::string::size_type const p(base_filename.find_last_of('.'));
        std::string file_without_extension = base_filename.substr(0, p);

        string datafilename = "AGGEN_"+file_without_extension+to_string(seed)+"-"
                            + ((CrossType==0)?"BLX":"ARI");
        writefile = path+"../results/"+datafilename;
        writefile += (localSearch==1)?"LS.txt":".txt";
        myfile.open(writefile,ios::out|ios::trunc);
        if(!myfile.is_open()){
            cerr << "[ERROR]: Couldn't open file, printing enabled" << endl;
            printing = true;
        }else{
            myfile << " ### Algoritmo Genetico Generacional " + to_string(Chromo) + " ### \n";
            myfile << "F\tclasific\treducir \tfitness \ttime\n";
        }

        if(streambus>1){
            plotting = true;
            if(plotting){
                plot_path = path+"../results/plots/"+datafilename+".txt";
                plot.open(plot_path,ios::out|ios::trunc);
                if(!plot.is_open()){
                    cerr << "[ERROR]: Couldn't open plot file, action dismissed\n";
                    plotting = false;
                }
            }
        }
    }
    filename = path+"../datos/"+filename;

    vector<char> label;
    MatrixXd allData = readValues(filename,label);

    std::normal_distribution<double> distribution(0.0, sqrt(0.3));
    int cols = allData.cols();
    float P_m = 0.1, min, max,alpha = 0.5;
    vector<float> behaviour; behaviour.resize(2);
    MatrixXd Solutions(Chromo, cols), Descendents(2,cols),FirstRandom(Chromo,cols);
    MatrixXd GenData(Chromo,2), DFit(2,2);

    FirstRandom = (MatrixXd::Random(Chromo,cols) + MatrixXd::Constant(Chromo,cols,1))/2.0;
    // RowVectorXd Fitness(Chromo), NewFitness(Chromo);
    RowVectorXd Cross1(cols), Cross2(cols);
    long int evaluations = 0, max_evaluations = 15000;
    unsigned int i, size, Diversidad = 1,generation=0, pair1=0, pair2=0;
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

        //Solutions = FirstRandom;
        Solutions = (MatrixXd::Random(Chromo,cols) + MatrixXd::Constant(Chromo,cols,1))/2.0;
        // GET INITIAL FITNESS
        //Fitness = getFit(data,Tlabel, Solutions,alpha);
        getFit(data,Tlabel,Solutions,GenData,alpha);
        generation = evaluations = 0;
        evaluations += Solutions.rows();
        while(evaluations < max_evaluations){
            //shuffleFit(Solutions, Fitness,-1);
            shuffleFit(Solutions,GenData,-1);

            // START CROSSING
            do{
                pair1 = Random::get<unsigned>(0,Solutions.rows()-1);
                pair2 = Random::get<unsigned>(0,Solutions.rows()-1);
            }while(pair1==pair2);

            if(CrossType == 0)
                BLXCross(Solutions.row(pair1),Solutions.row(pair2),Cross1,Cross2);
            else
                ArithmeticCross(Solutions.row(pair1),Solutions.row(pair2),Cross1,Cross2);

            pair1 = Random::get(0,1);
            if(pair1<=P_m){
                Diversidad = Random::get<unsigned>(0,data.cols()-1);
                for(i=0,size=Diversidad;i<size;++i){
                    Cross1(Random::get<unsigned>(0,data.cols()-1)) += Random::get(distribution);
                }
            }
            pair2 = Random::get(0,1);
            if(pair2<=P_m){
                Diversidad = Random::get<unsigned>(0,data.cols()-1);
                for(i=0,size=Diversidad;i<size;++i){
                    Cross2(Random::get<unsigned>(0,data.cols()-1)) += Random::get(distribution);
                }
            }
            Descendents.row(0) = Cross1;
            Descendents.row(1) = Cross2;

            getFit(data,Tlabel,Descendents,DFit,alpha);

            // LOCALSEARCH
            if(localSearch==1 && generation%Every==0){
                if(perf!=1){
                    shuffleFit(Descendents,DFit,-1);
                    for(i=0,size=localsize;i<size && i<Descendents.rows();++i){
                        behaviour[0] = DFit(i,0);
                        behaviour[1] = DFit(i,1);
                        Descendents.row(i) = LocalSearch(data,Tlabel, Descendents.row(i),
                                eval_num, max_eval-evaluations,maxTilBetter,behaviour,0.5);
                        DFit(i,0) = behaviour[0];
                        DFit(i,1) = behaviour[1];
                        evaluations += eval_num;
                    }
                }else{
                    // Do over the best only
                    getBest(DFit,indexGrid,localsize);
                    for(i=0;i<localsize && i<Descendents.rows();++i){
                        behaviour[0] = DFit(i,0);
                        behaviour[1] = DFit(i,1);
                        Descendents.row(indexGrid[i]) =
                            LocalSearch(data,Tlabel, Descendents.row(indexGrid[i]),
                                    eval_num, max_eval-evaluations,maxTilBetter,behaviour,0.5);
                        DFit(i,0) = behaviour[0];
                        DFit(i,1) = behaviour[1];
                        evaluations += eval_num;
                    }
                }
            }

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


            if(debuggin){
                GenData.rowwise().sum().maxCoeff(&maxIndex);
                GenData.rowwise().sum().minCoeff(&minIndex);
                cout << "###################################\n" ;
                cout << "[GENERATION NUMBER]: " << generation << "\n";
                cout << "[BEST FITNESS]: " << GenData(maxIndex,0) << "\t" << GenData(maxIndex,1) << "\t" << GenData.row(maxIndex).sum() << "\n";
                //cout << "[WORST FITNESS]: " << GenData(minIndex,0) << "\t" << GenData(minIndex,1) << "\t" << GenData.row(minIndex).sum() << "\n";
            }
            if(plotting){
                GenData.rowwise().sum().maxCoeff(&maxIndex);
                GenData.rowwise().sum().minCoeff(&minIndex);
                output = to_string(generation) + "\t" + to_string(GenData(maxIndex,0)/alpha) +
                    "\t" + to_string(GenData(maxIndex,1)/(1-alpha)) + "\t" + to_string(GenData.row(maxIndex).sum());
                plot << std::setw(31) << output;
                output = "\t" + to_string(GenData(minIndex,0)/alpha) + "\t" + to_string(GenData(minIndex,1)/(1-alpha)) +
                        "\t" + to_string(GenData.row(minIndex).sum()) + "\n";
                plot << std::setw(30) << output;
            }
            progress_bar(float(x*max_eval + evaluations)/float(max_eval*folds));

        } // END WHILE

        if(plotting) plot << "\n\n";
        momentoFin = high_resolution_clock::now();
        tiempo = duration_cast<milliseconds>(momentoFin - momentoInicio);
        getFit(test,Ttlabel, Solutions, GenData, 0.5);
        GenData.rowwise().sum().maxCoeff(&maxIndex);
        GenData.rowwise().sum().minCoeff(&minIndex);
        if(printing){
            cout << "\n###################################\n" ;
            cout << "[GENERATION NUMBER]: " << generation << "\n";
            cout << "[BEST FITNESS]: " << GenData(maxIndex,0)/alpha << "\t" << GenData(maxIndex,1)/(1-alpha) << "\t" << GenData.row(maxIndex).sum() << "\t";
            cout << tiempo.count() << endl;
            // cout << "[WORST FITNESS]: " << GenData(minIndex,0)/alpha << "\t" << GenData(minIndex,1)/(1-alpha) << "\t" << GenData.row(minIndex).sum() << "\n";
        }else{
            output = to_string(x) + "\t" + to_string(GenData(maxIndex,0)/alpha)
                    + "\t" +to_string(GenData(maxIndex,1)/(1-alpha)) + "\t" +
                    to_string(GenData.row(maxIndex).sum()) + "\t" + to_string(tiempo.count()) + "\n";
            myfile << std::setw(30) << output;
        }

    } //END FOLD
    if(!printing) {
        myfile.close();
        cout << "\n[FINISHED] Results saved in " + writefile << endl;
        if(plotting)
            cout << "[PLOT_DATA] Saved plot data in " + plot_path << endl;
    }else
        cout << "\n [FINISHED] " << endl;
    return 0;
}
