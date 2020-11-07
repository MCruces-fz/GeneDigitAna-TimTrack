# GeneDigitAna-TimTrack
 Tracks Generation, Digitation and Analysis Program for a four pad detector.

TRAGALDABAS Version
*****************************
   JA Garzon. labCAF / USC
   - Abril 2020
   2020 Abril. Sara Costa

#### GENE

Generates ntracks for a charged particle and propagates it on the Z 
direction through the nplan planes.
#### DIGIT
Simulates the digital response on the nplan detector planes, in which:
 - Coordinates (nx, ny) are determined where the track hit
 - The flight time is determined integrating tint
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
Typical Units:
 - Mass, momentum y energy: MeV
 - Distances in mm
 - Time in ps
********************************************************************