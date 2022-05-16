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

    if(argc<=10){
        cerr << "[ERROR] Couldn't resolve file name;" << endl;
        cerr << "[EXECUTION:] ./main (filename_directory) (label1)[char] (label2)[char] (0=Print/1=WriteFile/2=Write+Plot_data) (seed)[-∞,∞] \n"
        << "\t\t(0=No/1=Shuffle/2=Balanced) (0=BLX/1=ARITHMETIC) (POP.SIZE)[0,∞] (0=RandomOnly/1=RandomKeepBestCross/2=TopKeepBestCross) (0=No/1=LocalSearch) \n"
        << "\t\t{LOCALSEARCH OPTIONAL: (HowOften)[0,inf] (POP.Percentage)[0.0,1.0] (0=RandomSearch/1=OnlyBestSearch)}" << endl;
        exit(-1);
    }
    bool debuggin = false;

    /// Leemos los datos de entrada como es el archivo, barajar, tipo de cruce...
    string filename = argv[1];
    char type1 = *argv[2];
    char type2 = *argv[3];
    int streambus = atoi(argv[4]);
    long long int seed = stoll(argv[5]);
    Random::seed(seed);
    srand(seed);
    int shuffle = atoi(argv[6]);
    int CrossType = atoi(argv[7]);
    int Chromo = atoi(argv[8]);
    if(Chromo < 0)
        Chromo *= -1;
    int ChoiceMethod = atoi(argv[9]);
    int localSearch = atoi(argv[10]);
    // Default values:
    int Every = 10, perf = -1;
    float amount = 0.1;
    unsigned int localsize=ceil(amount*Chromo);
    if(localSearch==1 && argc>11){
        Every = atoi(argv[11]);
        if(argv[12] != NULL)
            amount = atof(argv[12]);
        if(argv[13] != NULL)
            perf = atoi(argv[13]);
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
                            + ((CrossType==0)?"BLX":"ARI" + to_string(ChoiceMethod));
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
    float P_c = 0.7, P_m = 0.1,alpha=0.5;
    vector<float> behaviour; behaviour.resize(2);
    int Cruzes = P_c * floor(Chromo/2.0);
    int Mutacion = P_m * (Chromo * cols);
    MatrixXd Solutions(Chromo, cols), FirstRandom(Chromo,cols);
    MatrixXd NewPopulation(Chromo, cols), GenData(Chromo,2), NewGenData(Chromo,2);

    FirstRandom = (MatrixXd::Random(Chromo,cols) + MatrixXd::Constant(Chromo,cols,1))/2.0;
    // RowVectorXd Fitness(Chromo), NewFitness(Chromo);
    RowVectorXd Cross1(cols), Cross2(cols), stored_fit(2);
    long int evaluations = 0, max_evaluations = 15000;
    unsigned int i, size, generation=0;
    unsigned int maxTilBetter = 20*allData.cols(), eval_num=0,max_eval=15000;
    RowVectorXd::Index minIndex,maxIndex;
    vector<int> bestIndex, randomIndex;
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
        getFit(data,Tlabel,*P1,GenData,alpha);
        generation = evaluations = 0;
        evaluations += P1->rows();
        while(evaluations < max_evaluations) {

            // GET BEST
            // START CROSSING
            GenData.rowwise().sum().maxCoeff(&maxIndex);
            stored_fit = GenData.row(maxIndex);
            if(ChoiceMethod==2) {
                evaluations += onlyBestCrossing(data,Tlabel,P1,NP2,GenData,CrossType,Cruzes,Mutacion);
            }else if(ChoiceMethod==1) {
                evaluations += randomCrossKeepBest(data,Tlabel,P1,NP2,GenData,CrossType,Cruzes,Mutacion);
            }else{
                evaluations += randomOnly(data,Tlabel,P1,NP2,GenData,CrossType,Cruzes,Mutacion);
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
                    getBest(GenData,bestIndex,localsize);
                    for(i=0;i<localsize;++i){
                        behaviour[0] = GenData(bestIndex[i],0);
                        behaviour[1] = GenData(bestIndex[i],1);
                        NP2->row(bestIndex[i]) =
                            LocalSearch(data,Tlabel, NP2->row(bestIndex[i]),
                                    eval_num, max_eval-evaluations,maxTilBetter,behaviour,0.5);
                        GenData(bestIndex[i],0) = behaviour[0];
                        GenData(bestIndex[i],1) = behaviour[1];
                        evaluations += eval_num;
                    }
                }
            }

            //evaluations += Solutions.rows();
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
                cout << "[BEST FITNESS]: " << GenData(maxIndex,0)/alpha << "\t" << GenData(maxIndex,1)/(1-alpha) << "\t" << GenData.row(maxIndex).sum() << "\n";
                 cout << "[WORST FITNESS]: " << GenData(minIndex,0)/alpha << "\t" << GenData(minIndex,1)/(1-alpha) << "\t" << GenData.row(minIndex).sum() << "\n";
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
        getFit(test,Ttlabel, *P1, GenData, alpha);
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
