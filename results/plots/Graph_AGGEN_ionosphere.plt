set terminal png size 1280, 720 enhanced font "Times-New-Roman,18"
set output "Ionosphere_GENLSVAR.png"

set key on
set key bottom right
set auto
set grid
set size 1,1
set yrange[0:1]
set ytics 0, 0.1, 1
stats "AGGEN_ionosphere150421-BLX-2LS_10-1.00.txt" u 1:4 i 0 name "a"
set xrange[0:a_max_x]

set title 'Algoritmo Generacional'
set  multiplot layout 2,1
    #unset xtics
    unset xlabel
    set ylabel 'Fitness\_fold-1'
    plot  "AGGEN_ionosphere150421-BLX-2LS_10-1.00.txt" u 1:4 i 0 w l lw 4 title "Best Fitness" , "" u 1:7 i 0 w l lw 4 title "Worst Fitness"
    unset ylabel
    unset title
    #plot  "AGEST_ionosphere150421-BLX.txt" u 1:4 i 1 w l lw 0.2 , "" u 1:7 i 1 w l lw 0.1
    #plot  "AGEST_ionosphere150421-BLX.txt" u 1:4 i 2 w l lw 0.2 , "" u 1:7 i 2 w l lw 0.1
    #plot  "AGEST_ionosphere150421-BLX.txt" u 1:4 i 3 w l lw 0.2 , "" u 1:7 i 3 w l lw 0.1
    #set xtics
    set xlabel 'Generaciones'
    set ylabel 'Fitness\_fold-5'
    plot  "AGGEN_ionosphere150421-BLX-2LS_10-1.00.txt" u 1:4 i 4 w l lw 4 title "Best Fitness", "" u 1:7 i 4 w l lw 4 title "Worst Fitness"
unset multiplot

#Plot for [a=0:4] "AGEST_ionosphere150421-BLX.txt" u 1:4 i a w lp lw 0.1 ps 0.1, for[a=0:4] "" u 1:7 i a w lp lw 0.1 ps 0.1

