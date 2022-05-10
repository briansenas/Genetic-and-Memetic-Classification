/**
 * @file Genetics.h
 * @version 1.0
 * @date 09/04/2022
 * @author Brian Sena Simons 3ºA-A2
 * @brief
 * Should be used with the help of mytools.h
 * @code
 * int main(){
    [...]
    Solutions = (MatrixXd::Random(Chromo,cols) + MatrixXd::Constant(Chromo,cols,1))/2.0;
    // GET INITIAL FITNESS
    Fitness = getFit(allData,label, Solutions,0.5);
    [..]
 * }
 * @endcode
 **/
#ifndef GENETICS_H
#define GENETICS_H

#include "../inc/eigen-3.4.0/Eigen/Dense"
#include "Euclidean.h"
#include "mytools.h"
#include "../inc/random.hpp"
#include <string.h>
#include <vector>

using namespace std;
using namespace Eigen;
using namespace std::chrono;
using Random = effolkronium::random_static;

/*
 * @brief Dado dos padres, añadimos sus columnas y multiplicamos cada valor
 * por un valor alpha generado aleatoriamente entre 0 y 1. Esto lo realizamos
 * dos veces para generar dos hijos.
 * @param parent1 padre número 1.
 * @param parent2 padre número 2.
 * @param res1 hijo número 1.
 * @param res2 hijo número 2.
 */
void ArithmeticCross(RowVectorXd parent1, RowVectorXd parent2, RowVectorXd& res1, RowVectorXd& res2,long int seed=-1);

/*
 * @brief Dado dos padres, para cada columna calculamos el valor máximo entre ellos
 * y el valor mínimo, con estos valores computamos la distancia y de manera que
 * tenemos el intervalo [Mínimo - alpha * Distance, Máximo + alpha * Distancia].
 * Ese intervalo se utiliza para generar un número aleatorio para esa columna.
 * @param parent1 padre número 1.
 * @param parent2 padre número 2.
 * @param res1 hijo número 1.
 * @param res2 hijo número 2.
 * @param alpha valor del blx que multiplica la distancia
 * @param seed semilla para el generado de números aleatorios
 */
void BLXCross(RowVectorXd parent1, RowVectorXd parent2,RowVectorXd& res1, RowVectorXd& res2, float alpha=0.3, long int seed=-1);

/*
 * @brief Data una matriz de datos con sus etiquetas y una matriz de pesos,
 * para cada fila de la matriz de pesos computamos el valor resultante del 1NN
 * ponderado con los parámetros de reducción y tasa de acierto multiplicados por
 * el valor alpha.
 * @param data matriz con los datos,
 * @param Tlabel vector con las etiquetas,
 * @param Solutions matriz con los pesos
 * @param alpha Ponderación entre reducción y Tasa de Aciertos.
 * @return Devolvemos un vector con la puntuación de cada fila en sus columnas.
 */
RowVectorXd getFit(MatrixXd data, vector<char> Tlabel, MatrixXd& Solutions,float alpha=0.5);
RowVectorXd getFit(MatrixXd data, vector<char> Tlabel, MatrixXd& Solutions, MatrixXd& GenData, float alpha=0.5);

RowVectorXd LocalSearch(MatrixXd allData,vector<char> label, RowVectorXd Weights,
unsigned int& eval_num, unsigned int max_eval, unsigned int maxTilBetter, vector<float>& fitness, float alpha);

#endif
