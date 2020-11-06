# GeneDigitAna-TimTrack
 Tracks Generation, Digitation and Analysis Program for a four pad detector.

TRAGALDABAS Version
*****************************
   JA Garzon. labCAF / USC
   - Abril 2020
   2020 Abril. Sara Costa

#### GENE

 Genera ntraks trazas de una particula cargada y la
 propaga en la direccion del eje Z a traves de nplan planos
#### DIGIT
Simula la respuesta digital en nplan planos de detectores, en los que:
 - se determina las coordenadas (nx,ny) del pad por atravesado
 - Se determina el tiempo de vuelo intengrarado tint, 
#### ANA
 - Reconstruye la traza mediante el metodo TimTrack, usando la respuesta del
detetor. Por ello la traza reconstruida no coincide exactamente con la 
generada
 - Calcula la matriz de varianzas-covariances mErr
 - Al final, calcula lo que llamamos matriz de error, reducida, que contiene
   * Las incertidumbres de los parametros en la diagonal principal
   * Los coeficientes de correlacion entre parametros en la mitad superior
***************************************************************
#### Comments
Algunos criterios de programacion:
 - Los nombres de las variables siguen en general alguna norma nemotecnica
 - Los nombres de los vectores comienzan con v
 - Los nombres de las matrices comienzan con m
********************************************************************
Unidades tipicas:
 - Masa, momento y energia: MeV
 - Distancias en mm
 - Tiempo de ps
********************************************************************