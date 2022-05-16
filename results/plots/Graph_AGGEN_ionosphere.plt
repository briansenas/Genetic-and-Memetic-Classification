set terminal png size 1280, 720
set output "Ionosphere_Graphs.png"

set key off
set auto
set grid
set size 6,6
set yrange[0:1]
set ytics 0, 0.1, 1
stats "AGGEN_ionosphere170421-ARI.txt" u 1:2 i 0 name "a"
set xrange[0:a_max_x]

set title 'Algoritmo Generacional'
set  multiplot layout 2,1
    #unset xtics
    unset xlabel
    set ylabel 'fitness\_fold-1'
    plot  "AGGEN_ionosphere170421-ARI.txt" u 1:4 i 0 w l lw 0.2 , "" u 1:7 i 0 w l lw 0.1
    unset ylabel
    unset title
    #plot  "AGGEN_ionosphere170421-ARI.txt" u 1:4 i 1 w l lw 0.2 , "" u 1:7 i 1 w l lw 0.1
    #plot  "AGGEN_ionosphere170421-ARI.txt" u 1:4 i 2 w l lw 0.2 , "" u 1:7 i 2 w l lw 0.1
    #plot  "AGGEN_ionosphere170421-ARI.txt" u 1:4 i 3 w l lw 0.2 , "" u 1:7 i 3 w l lw 0.1
    #set xtics
    set xlabel 'generaciones'
    set ylabel 'fitness\_fold-5'
    plot  "AGGEN_ionosphere170421-ARI.txt" u 1:4 i 4 w l lw 0.2 , "" u 1:7 i 4 w l lw 0.1
unset multiplot

#Plot for [a=0:4] "AGGEN_ionosphere170421-ARI.txt" u 1:4 i a w lp lw 0.1 ps 0.1, for[a=0:4] "" u 1:7 i a w lp lw 0.1 ps 0.1

