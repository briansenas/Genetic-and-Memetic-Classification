# ** _Brian Sena Simons 3ºA - A2_ **
# Práctica 2 - MetaHeurística

###- Utilize el "cmake CMakeLists.txt && make" para compilar;
###- Los algoritmos pueden ser ejecutados utilizando los scripts en ./scripts {Ej. runAll.sh}
###- Los resultados se guardan en ./results;

## Ejecución individual:
### Algoritmo Generacional:
./bin/AGGEN ./datos/ionosphere.arff {seed} [0-1] [0-1]

- Primer   0 ó 1: {0=Sin Barajar}; {1=Barajando};
- Segundo  0 ó 1: {0=Cruce BLX};   {1=Cruce Aritmetico};

## Descripción breve del Problema
La idea es comparar distintos tipos de algoritmos para clasificar datos pertenecientes
a una base de datos públicas que nos provee el profesorado. Partiremos primero
de la implementación del típico algoritmo de clasficación K-NN dónde K representa
el número de vecinos a mirar y la idea conceptual es buscar los K vecinos más
cercanos para realizar una predicción sobre que clase pertenece el objeto a predecir.

Una vez implementado el algoritmo 1NN intentaremos mejorar el porcentaje de aciertos
utilizando técnicas de ponderación de características mediante un vector de pesos.
El grueso de la práctica está en el cálculo de esos pesos. En esta parte de la
práctica compararemos los algoritmos empleados anteriormente (1NN, Greedy,
búsqueda local) con unas variaciones genéticas y meméticas.

La primera de ella consiste en generar toda una nueva población de datos a
partir de soluciones aleatorias utilizando operadores de cruces como lo
es el BLX-alpha, que calcula un intervalo de valores a generar para explorar
el vecindario, y el cruce aritmético que es una media ponderada de la característica
a calcular (Columna o gen en concreto al que se aplica). Luego aplicaremos una
mutación con una Probabilidad de 0.1 a uno o varios genes de una solución.

La segunda implemetación consiste en generar dos nuevas soluciones a partir de
dos padres aleatorios, mutarlas y que compitan para entrar de vuelta a la población
original, es decir, deben de tener mejor valor de función.

La última implementación, la versión memética, consiste en submeter esos algoritmos
a la búsqueda local cada cierto número de generación y observar el comportamiento.
Una vez desarrollado toda la práctica y obtengamos todos los datos podemos
proceder a realizar un análisis profundo de las diferencias.

Para la práctica he tenido que definir las funciones en mytools.h y mytools.cpp;
